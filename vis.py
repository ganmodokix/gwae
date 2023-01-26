import argparse
from collections import namedtuple
import math
import sys, os

import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from tqdm import tqdm

import vaetc

sys.path.append("./")
import models

def plot_dzdx(cp: vaetc.Checkpoint):

    if not hasattr(cp.model, "decode"):
        print("No decoder", file=sys.stderr)
        return

    # prior
    num_pairs = 10000
    z = cp.model.sample_prior(num_pairs * 2)
    x = cp.model.decode(z)
    x = x.view(x.shape[0], -1)

    # sampling
    dz = ((z[:num_pairs,:] - z[num_pairs:,:]) ** 2).sum(dim=1)
    dx = ((x[:num_pairs,:] - x[num_pairs:,:]) ** 2).sum(dim=1)

    dz = dz.detach().cpu().numpy()
    dx = dx.detach().cpu().numpy()

    plt.figure()
    hist = plt.hist2d(dz, dx, bins=50, cmap="viridis")
    plt.xlabel("$\\Delta$z")
    plt.ylabel("$\\Delta$x")
    plt.colorbar(hist[3])
    plt.savefig(os.path.join(cp.options["logger_path"], "metric_interspace.svg"))
    plt.savefig(os.path.join(cp.options["logger_path"], "metric_interspace.pdf"))
    plt.close()

    plt.figure()
    hist = plt.scatter(dz, dx, alpha=0.3)
    plt.xlabel("$\\Delta$z")
    plt.ylabel("$\\Delta$x")
    plt.savefig(os.path.join(cp.options["logger_path"], "metric_interspace_scatter.svg"))
    plt.savefig(os.path.join(cp.options["logger_path"], "metric_interspace_scatter.pdf"))
    plt.close()

def plot_sampler(cp: vaetc.Checkpoint):

    data_size = 1024
    z = cp.model.sample_prior(data_size)
    z = z.detach().cpu().numpy()
    _, z_dim = z.shape

    root_dir = os.path.join(cp.options["logger_path"], "scatter_sampler")
    os.makedirs(root_dir, exist_ok=True)

    # indices = range(z_dim)
    EncodedData = namedtuple("EncodedData", ["z", "mean"])
    indices = vaetc.evaluation.visualizations.distribution.top_k_interesting_latents(EncodedData(z=z, mean=None), k=10)

    ij = [(i, j) for i in indices for j in indices if i < j]
    for i, j in tqdm(ij):

        plt.figure(figsize=(7, 6))
        sns.set(style="whitegrid")
        plt.scatter(z[:,i], z[:,j])
        plt.xlabel(f"$z_{{{i}}}$")
        plt.ylabel(f"$z_{{{j}}}$")
        file_name = os.path.join(root_dir, f"z{i:03d}_z{j:03d}")
        plt.savefig(file_name + ".pdf")
        plt.savefig(file_name + ".svg")
        plt.close()

def plot_causal(cp: vaetc.Checkpoint):

    if not hasattr(cp.model, "sampler") or not hasattr(cp.model.sampler, "dag"):
        return

    g = cp.model.sampler.dag()
    g = g.detach().cpu().numpy()

    plt.figure()
    sns.set(style="whitegrid")
    plt.imshow(g, interpolation="nearest", vmin=0, vmax=1, cmap="coolwarm")
    plt.colorbar()
    out_name = os.path.join(cp.options["logger_path"], "causal_mask")
    plt.savefig(out_name + ".svg")
    plt.close()

def cluster(cp: vaetc.Checkpoint, mink=2, maxk=12):

    data_valid = np.load(os.path.join(cp.options["logger_path"], "zt_valid.npz"))
    data_test = np.load(os.path.join(cp.options["logger_path"], "zt_test.npz"))

    valid_size = data_valid["z"].shape[0]
    test_size = data_test["z"].shape[0]
    z_dim = data_test["z"].shape[1]
    t_dim = data_test["t"].shape[1]
    
    x = []
    y = []
    batch_size = cp.options["batch_size"]
    for n_clusters in tqdm(range(mink, maxk+1)):
        
        clf = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=batch_size)

        for ib in range(0, valid_size, batch_size):
            ie = ib + batch_size
            clf.partial_fit(data_valid["z"][ib:ie,:])

        t_pred = clf.predict(data_test["z"])
        score = calinski_harabasz_score(data_test["z"], t_pred)
        x += [n_clusters]
        y += [score]

    plt.figure()
    sns.set(style="whitegrid")
    plt.plot(x, y)
    plt.xlabel("The number of clusters")
    plt.ylabel("Variance Ratio Criterion")
    out_name = os.path.join(cp.options["logger_path"], "cluster_analysis")
    plt.xticks(x)
    plt.gca().yaxis.get_major_locator().set_params(integer=True)
    plt.savefig(out_name + ".svg")
    plt.savefig(out_name + ".pdf")
    plt.close()

def main(checkpoint: vaetc.Checkpoint):

    with torch.no_grad():
        checkpoint.model.eval()
        plot_dzdx(checkpoint)
        plot_sampler(checkpoint)
        plot_causal(checkpoint)
        cluster(checkpoint)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("logger_path", type=str)
    parser.add_argument("--evaluate", "-e", action="store_true", default=False)

    args = parser.parse_args()

    cp = vaetc.load_checkpoint(os.path.join(args.logger_path, "checkpoint_best.pth"))

    if args.evaluate:
        vaetc.evaluate(cp, cp)

    main(cp)