# Code for Gromov-Wasserstein Autoencoders (GWAE)

This repository contains the official implementation for ["Gromov-Wasserstein Autoencoders" (ICLR 2023)](https://openreview.net/forum?id=sbS10BCtc7).
The 

## Paper Abstract
 Variational Autoencoder (VAE)-based generative models offer flexible representation learning by incorporating meta-priors, general premises considered beneficial for downstream tasks. However, the incorporated meta-priors often involve ad-hoc model deviations from the original likelihood architecture, causing undesirable changes in their training. In this paper, we propose a novel representation learning method, Gromov-Wasserstein Autoencoders (GWAE), which directly matches the latent and data distributions using the variational autoencoding scheme. Instead of likelihood-based objectives, GWAE models minimize the Gromov-Wasserstein (GW) metric between the trainable prior and given data distributions. The GW metric measures the distance structure-oriented discrepancy between distributions even with different dimensionalities, which provides a direct measure between the latent and data spaces. By restricting the prior family, we can introduce meta-priors into the latent space without changing their objective. The empirical comparisons with VAE-based models show that GWAE models work in two prominent meta-priors, disentanglement and clustering, with their GW objective unchanged.

# Installation
We developed and tested this code in the environment as follows:

- Ubuntu 20.04
- Python3.9
- CUDA 11.3
- 1x GeForce® RTX 2080 Ti
- 31.2GiB (32GB) RAM

We have also confirmed that this code successfully works on Python3.10.

We recommend to run this code under the `venv` envirionment of Python 3.9.
The requirements can be easily installed using `pip`.
```
$ python3.9 -m venv .env
$ source .env/bin/activate
(.env) $ pip install -U pip
(.env) $ pip install wheel
(.env) $ pip install -r requirements.txt
```
In `requirements.txt`, a third-party representation learning package is specified, which is downloaded from `github.com` and installed via `pip`.

# How to Train
Run `train.py` with a setting file to train models.
```
(.env) $ python train.py setting/gwae.yaml
(.env) $ python train.py setting/vae.yaml
(.env) $ python train.py setting/geco.yaml
(.env) $ python train.py setting/ali.yaml
...
```
The results are saved in the `logger_path` directory specified in the setting YAML file.

# How to Evaluate GWAEs
Run `vis.py` with the `logger_path` directory specified in the settings.
```
(.env) $ python vis.py runs/celeba_gwae
```

# How to Evaluate Models with FID
To compute the generation FID, run `fid.py` with the `logger_path` directory path.
```
(.env) $ python fid.py runs/celeba_gwae
```
*Note*: The script `fid.py` calculates the FID score of the images sampled from *the generative model*. On the other hand, the FID values computed by [the `vaetc` package](https://github.com/ganmodokix/vaetc) is *reconstruction* FID, using the test set images for generating *reconstructed* images.