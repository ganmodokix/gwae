import os
import argparse
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import vaetc
from vaetc.evaluation.metrics.generation import fid
import yaml
sys.path.append("./")
import models
sys.path.pop

def make_loader(dataset: Dataset, batch_size: int):
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count() - 1,
    )

@torch.no_grad()
def fid_generation(model: vaetc.models.VAE, dataset: vaetc.data.utils.ImageDataset, batch_size: int = 64):

    features, transform = fid.build_features_inception_v3()
    loader = make_loader(dataset.test_set, batch_size)

    model.eval()

    fs_real, fs_gen = [], []
    for x, t in tqdm(loader):

        this_batch_size = x.shape[0]

        x = x.cuda()
        
        zs = model.sample_prior(this_batch_size)
        xs = model.decode(zs)

        fs_real += [fid.features_batch(x , features, transform)]
        fs_gen  += [fid.features_batch(xs, features, transform)]
        
    f_real = torch.cat(fs_real, dim=0)
    f_gen  = torch.cat(fs_gen , dim=0)

    mean_real, cov_real = fid.mean_and_cov(f_real)
    mean_gen , cov_gen  = fid.mean_and_cov(f_gen )
    
    return fid.fid_gaussian(mean_real, cov_real, mean_gen, cov_gen)

def main(logger_path: str):

    checkpoint_path = os.path.join(logger_path, "checkpoint_best.pth")
    cp = vaetc.load_checkpoint(checkpoint_path)

    fid = fid_generation(cp.model, cp.dataset, cp.options.get("batch_size", 64))

    vaetc.utils.debug_print(f"On {logger_path}:")
    vaetc.utils.debug_print(f"FID: {fid:.1f}")

    with open(os.path.join(logger_path, "fid.yaml"), "w") as fp:
        yaml.safe_dump({
            "FID": fid,
            "num_samples": len(cp.dataset.test_set)
        }, fp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("logger_path", type=str)
    args = parser.parse_args()

    main(args.logger_path)