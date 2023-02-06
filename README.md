# Code for Gromov-Wasserstein Autoencoders (GWAE)

This repository contains the code for ["Gromov-Wasserstein Autoencoders" (ICLR 2023)](https://openreview.net/forum?id=sbS10BCtc7).

# Installation
We developed and tested this code in the environment as follows:
- Ubuntu 20.04
- Python3.9
- CUDA 11.3
- 1x GeForce® RTX 2080 Ti
- 31.2GiB (32GB) RAM
We have also confirmed that this code successfully works on Python3.10.

We encourage that the code is run under the `venv` envirionment of Python 3.9.
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

# How to Evaluate GWAE
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