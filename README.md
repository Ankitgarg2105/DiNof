
# College Research Project: Diffusion Models with Deterministic Normalizing Flow Priors
This repository contains our college research project implementing the paper **[Diffusion Models with Deterministic Normalizing Flow Priors](https://openreview.net/forum?id=ACMNVwcR6v)** by Mohsen Zand, Ali Etemad, and Michael Greenspan. 
## Project Overview
For our research project, we were tasked with implementing the concepts presented in the base paper, which introduces deterministic normalizing flows as priors for diffusion models. This repository contains the code for reproducing those results and exploring the architecture.
## Development Environment
- **OS**: Ubuntu 20.04
- **CUDA**: 11.6
- **GPU**: NVIDIA A100 (or equivalent)
- **Python**: 3.11.1
## Datasets
- **CIFAR-10**: Downloaded automatically during execution.
- **CelebA-HQ-256**: To use this dataset, please follow the download instructions in [Vahdat and Kautz](https://github.com/NVlabs/NVAE).
*(Note: FID statistics files are required to compute evaluation metrics but are not included due to file size limits. You can generate them using `fid_score.py` and `precompute_fid_statistics.py` located in the `fids` folder).*
## Running the Code
Model architectures are defined in the `models` folder.
To train or evaluate a specific model, run `main.py` with the appropriate flags for your paths and directories:
- Set `--mode` to `train` or `eval` for training or evaluation.
- Select one of the provided configuration files from the `configs` folder (and its corresponding SDE: ve, vp, or subvp).
All config, training, and sampling files are self-explanatory.
## Base Paper Citation
If you are interested in the original research our project is based on, please refer to:
```bibtex
@article{
  zand2024diffusion,
  title={Diffusion Models with Deterministic Normalizing Flow Priors},
  author={Mohsen Zand and Ali Etemad and Michael Greenspan},
  journal={Transactions on Machine Learning Research},
  year={2024},
  url={https://openreview.net/forum?id=ACMNVwcR6v}
}
```
