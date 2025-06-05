## Description

This repository contains the official implementation of the Generalised Scalable Latent Variable Multi-Output Gaussian Process (GS-LVMOGP) model, an extension of [LVMOGP NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/1680e9fa7b4dd5d62ece800239bb53bd-Abstract.html) with support of mini-batch training and $Q>1$ coregionalisation matrices.

## Requirements

```
torch 2.5.1
gpytorch 1.13
```

## Citation
If you find this work helpful, please consider citing our TMLR paper:
```
@article{
jiang2025scalable,
title={Scalable Multi-Output Gaussian Processes with Stochastic Variational Inference},
author={Xiaoyu Jiang and Sokratia Georgaka and Magnus Rattray and Mauricio A {\'A}lvarez},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=kK0WrBZAli},
note={}
}
```
