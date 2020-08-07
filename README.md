# Extended Neural Estimator for Entropy Production (NEEP)

[![Paper](http://img.shields.io/badge/paper-arxiv.2003.04166-B31B1B.svg)](https://arxiv.org/abs/2003.04166)
[![LICENSE](https://img.shields.io/github/license/kdkyum/neep.svg)](https://github.com/kdkyum/neep/blob/master/LICENSE)

## Introduction

This repository extends on the source code that was originally used to 
generate the results presented in 
[Learning entropy production via neural networks](https://arxiv.org/abs/2003.04166).

The neural estimator is extended for calculating entropy production rates 
of higher dimensional systems including hard disk monte carlo simulations,
lennard jones molecular dynamics simulations, and active matter simulations.

## Installation
```bash
git clone https://github.com/rganti/neep
cd neep
conda env create -f environment.yml
conda activate neep
python -m ipykernel install --name neep
python setup.py develop
```

## Quickstart

```bash
jupyter notebook
```

See the following notebooks for the runs in the paper.
### Bead-spring model
* [`notebooks/bead-spring.ipynb`](notebooks/bead-spring.ipynb)

### Discrete flashing ratchet
* [`notebooks/ratchet.ipynb`](notebooks/ratchet.ipynb)

### RNEEP for Non-Markovian process
* [`notebooks/partial-ratchet-RNEEP.ipynb`](notebooks/partial-ratchet-RNEEP.ipynb)

## Author
Original Authors: Dong-Kyum Kim, Youngkyoung Bae, Sangyun Lee and Hawoong Jeong

Raman Ganti

## Bibtex
Cite following bibtex.
```bibtex
@article{kim2020learning,
  title={Learning entropy production via neural networks},
  author={Dong-Kyum Kim and Youngkyoung Bae and Sangyun Lee and Hawoong Jeong},
  journal={arXiv preprint arXiv:2003.04166},
  year={2020}
}
```

## License

This project following MIT License as written in LICENSE file.
