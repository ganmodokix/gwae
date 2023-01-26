import math
import os
import random
import sys

import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import vaetc
from vaetc.data.utils import IMAGE_SHAPE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
from vaetc.utils import debug_print

sys.path.append("./")
import models
sys.path.pop()

@torch.no_grad()
def plot_cluster_sample(cp: vaetc.Checkpoint):

    model = cp.model

    if not isinstance(model, models.GromovWassersteinAutoEncoder):
        debug_print("The model is not GWAE")
        return

    if not isinstance(model.sampler, models.gwae.GaussianMixtureSampler):
        debug_print("The model sampler is not Gaussian mixture")
        return

    out_dir = os.path.join(cp.options["logger_path"], "cluster_samples")
    os.makedirs(out_dir, exist_ok=True)

    n = 16
    
    for i in tqdm(range(model.sampler.num_components)):

        batch_size = n ** 2
        
        m = model.sampler.component_mean[i]
        s = model.sampler.component_sqrtprecision[i]
        eps = torch.randn(size=[batch_size, model.sampler.z_dim], device=m.device)
        z = m[i] + (eps[:,None,:] * s[:,:]).sum(dim=1)
        z = model.sampler.batchnorm(z)
        x = model.decode(z)
        x = x.view(n, n, *IMAGE_SHAPE)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.detach().cpu().numpy()
        x = x[...,::-1]
        x = np.concatenate(x, axis=1)
        x = np.concatenate(x, axis=1)
        x = (x * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f"component_{i:03d}.png"), x)

def anormaly_stdgaussian(z: np.ndarray) -> np.ndarray:
    """
    E(z) = -log N(z|0,I)
    """
    return 0.5 * (z ** 2 + math.log(math.pi * 2)).sum(axis=1)

@torch.no_grad()
def anormaly_dagmm(model: vaetc.models.DAGMM, z: np.ndarray) -> np.ndarray:
    
    data_size = z.shape[0]
    batch_size = 64

    energy = []
    for ib in range(0, data_size, batch_size):
        z_batch = z[ib:ib+batch_size]
        z_batch = torch.tensor(z_batch).float().cuda()
        en = model.energy(z_batch, model.running_mean, model.running_sigma, model.running_phi)
        energy += [en.detach().cpu().numpy()]

    return np.concatenate(energy, axis=0)

def anormaly_gwae(model: models.GromovWassersteinAutoEncoder, z: np.ndarray) -> np.ndarray:

    data_size = z.shape[0]
    batch_size = 64

    energy = []
    for ib in range(0, data_size, batch_size):
        z_batch = z[ib:ib+batch_size]
        z_batch = torch.tensor(z_batch).float().cuda()
        x_batch = model.decode(z_batch)
        en = model.disc_block(x_batch, z_batch).squeeze(1)
        energy += [en.detach().cpu().numpy()]

    return np.concatenate(energy, axis=0)

def anormaly(model: vaetc.models.RLModel, z: np.ndarray) -> np.ndarray:
    """ The higher `anomaly` is, the more likely `z` is to be OoD """
    
    if isinstance(model, models.GromovWassersteinAutoEncoder):
        return anormaly_gwae(model, z)
    elif isinstance(model, vaetc.models.DAGMM):
        return anormaly_dagmm(model, z)
    else:
        return anormaly_stdgaussian(z)

def roc(ano: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    t = t.astype(int)
    debug_print("Calculating ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_true=t, y_score=ano, pos_label=1)
    debug_print("Calculating AUC...")
    auc = roc_auc_score(y_true=t, y_score=ano)

    return fpr, tpr, thresholds, auc

def ood_samples(
    cp: vaetc.Checkpoint,
    num_samples: int,
    batch_size: int = 64) -> np.ndarray:

    dataset = vaetc.data.omniglot().test_set
    # dataset = cp.dataset.test_set
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count() - 1)

    z = []
    data_gen = iter(loader)
    for _ in tqdm(range(0, num_samples, batch_size)):

        try:
            x_batch, t_batch = next(data_gen)
        except StopIteration:
            data_gen = iter(loader)
            x_batch, t_batch = next(data_gen)
        x_batch = x_batch.cuda()

        # # images with permuted n^2 patches
        # n = 4
        # assert IMAGE_WIDTH % n == 0 and IMAGE_HEIGHT % n == 0

        # patch_height = IMAGE_HEIGHT // n
        # patch_width = IMAGE_WIDTH // n
        # patch_size = (patch_height, patch_width)
        # x_batch = F.unfold(x_batch, kernel_size=patch_size, stride=patch_size)
        # perm = torch.rand(size=[batch_size, n*n], device=x_batch.device)
        # perm = perm.argsort(dim=1)
        # bidx = torch.arange(x_batch.shape[0], device=x_batch.device)
        # cidx = torch.arange(x_batch.shape[1], device=x_batch.device)
        # x_batch = x_batch[bidx[:,None,None],cidx[None,:,None],perm[:,None,:]]
        # x_batch = F.fold(x_batch, output_size=(IMAGE_HEIGHT, IMAGE_WIDTH), kernel_size=patch_size, stride=patch_size)

        # # img: np.ndarray = x_batch.detach().cpu().numpy()
        # # img = img.transpose(0, 2, 3, 1)
        # # img = np.concatenate(img, axis=1)
        # # img = (img * 255).astype(np.uint8)[...,::-1]
        # # cv2.imwrite("sandbox/patchshuffled.jpg", img)
        # # exit(-1)

        # # random noises
        # x_batch = torch.rand(size=[batch_size, *IMAGE_SHAPE], device="cuda")

        z_batch = cp.model.encode(x_batch)
        z += [z_batch.detach().cpu().numpy()]
    
    return np.concatenate(z, axis=0)

def make_ood_dataset(z_id: np.ndarray, cp: vaetc.Checkpoint) -> tuple[np.ndarray, np.ndarray]:
    
    z_ood = ood_samples(cp, num_samples=z_id.shape[0])

    ano_id = anormaly(cp.model, z_id)
    ano_ood = anormaly(cp.model, z_ood)
    t_id = np.zeros_like(ano_id)
    t_ood = np.ones_like(ano_ood)

    ano = np.concatenate([ano_id, ano_ood], axis=0)
    t = np.concatenate([t_id, t_ood], axis=0)

    return ano, t

def ood_roc_curve(ano: np.ndarray, t: np.ndarray, out_dir: str):

    fpr, tpr, thresholds, auc = roc(ano, t)

    # save as npz
    np.savez(os.path.join(out_dir, "roc_curve"), fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc)

    # save figure
    plt.figure()
    sns.set_theme(style="whitegrid")
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="dashed", linewidth=1, color="black")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    # plt.title(f"AUC: {auc:.3f}")
    plt.savefig(os.path.join(out_dir, "roc_curve.pdf"))
    plt.savefig(os.path.join(out_dir, "roc_curve.svg"))

    with open(os.path.join(out_dir, "auc.txt"), "w") as fp:
        fp.write(f"AUC: {auc}\n")

def ood_detection(cp: vaetc.Checkpoint):

    # save dir
    out_dir = os.path.join(cp.options["logger_path"], "ood_detection")
    os.makedirs(out_dir, exist_ok=True)

    # draw roc curve
    zt_test = np.load(os.path.join(cp.options["logger_path"], "zt_test.npz"))
    ano, t = make_ood_dataset(z_id=zt_test["z"], cp=cp)
    ood_roc_curve(ano, t, out_dir)

def main(cp: vaetc.Checkpoint):

    cp.model.eval()
    plot_cluster_sample(cp)
    ood_detection(cp)