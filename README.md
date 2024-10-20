# AEM: Attention Entropy Maximization for Multiple Instance Learning based Whole Slide Image Classification

This repository contains the PyTorch implementation of our paper "[AEM: Attention Entropy Maximization for Multiple Instance Learning based Whole Slide Image Classification](https://arxiv.org/pdf/2406.15303)". The code is built upon the [ACMIL](https://github.com/dazhangyu123/ACMIL) framework.


[![arXiv](https://img.shields.io/badge/arXiv-2406.15303-b31b1b.svg)](https://arxiv.org/pdf/2406.15303)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/datasets/dazhangyu123/AEM-dataset)


## Overview

Attention Entropy Maximization (AEM) is a novel plug-and-play regularization technique designed to address attention concentration in Multiple Instance Learning (MIL) frameworks. Key features:

- Simple yet effective solution for mitigating overfitting in whole slide image classification tasks
- Requires no additional modules
- Features just one hyperparameter
- Demonstrates excellent compatibility with various MIL frameworks and techniques

## Dataset Preparation
We provide pre-extracted features to facilitate the reproduction of our results. These features are available through two platforms:

1. **Hugging Face**: A comprehensive dataset containing all features
   - [AEM-dataset on Hugging Face](https://huggingface.co/datasets/dazhangyu123/AEM-dataset)

2. **Quark Pan**: Individual feature sets for specific models and datasets

### CAMELYON16 Dataset Features

| Model | Feature Set |
|-------|-------------|
| ImageNet supervised ResNet18 | [Download](https://pan.quark.cn/s/dd77e6a476a0) |
| SSL ViT-S/16 | [Download](https://pan.quark.cn/s/6ea54bfa0e72) |
| PathGen-CLIP ViT-L (336 × 336 pixels) | [Download](https://pan.quark.cn/s/62fe3dc65291) |

### CAMELYON17 Dataset Features

| Model | Feature Set |
|-------|-------------|
| ImageNet supervised ResNet18 | [Download](https://pan.quark.cn/s/22acfa46905e) |
| SSL ViT-S/16 | [Download](https://pan.quark.cn/s/4883fff37071) |
| PathGen-CLIP ViT-L (336 × 336 pixels) | [Download](https://pan.quark.cn/s/0f8730bbbdf1) |

These pre-extracted features are designed to help researchers quickly implement and validate our results. Choose the dataset and model that best suits your research needs.

For custom datasets, modify and run [Step1_create_patches_fp.py](Step1_create_patches_fp.py) and [Step2_feature_extract.py](Step2_feature_extract.py). More details can be found in the [CLAM repository](https://github.com/mahmoodlab/CLAM/).

Note: We recommend extracting features using SSL pretrained methods. Our code uses checkpoints provided by [Benchmarking Self-Supervised Learning on Diverse Pathology Datasets](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.html).

### Pretrained Checkpoints

When running [Step2_feature_extract.py](Step2_feature_extract.py), you can choose from various feature encoders. Links to obtain their checkpoints are provided below:

| Model | Website Link |
|-------|--------------|
| Lunit | [Website](https://github.com/lunit-io/benchmark-ssl-pathology) |
| UNI | [Website](https://github.com/mahmoodlab/UNI) |
| Gigapath | [Website](https://github.com/prov-gigapath/prov-gigapath) |
| Virchow | [Website](https://huggingface.co/paige-ai/Virchow) |
| PLIP | [Website](https://github.com/PathologyFoundation/plip) |
| Quilt-net | [Website](https://github.com/wisdomikezogwo/quilt1m) |
| Biomedclip | [Website](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| PathGen-CLIP | [Website](https://github.com/superjamessyx/PathGen-1.6M) |

## Training

### Baseline (ABMIL)
To run the baseline ABMIL, use the following command with `lamda = 0`:

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --seed 4 --wandb_mode online --lamda 0 --pretrain medical_ssl --config config/camelyon17_config.yml
```
### AEM
To run AEM, use the following command with `lamda > 0`:
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --seed 4 --wandb_mode online --lamda 0.1 --pretrain medical_ssl --config config/camelyon17_config.yml
```


## BibTeX
If you find our work useful for your project, please consider citing the following paper.


```
@misc{zhang2023attentionchallenging,
      title={AEM: Attention Entropy Maximization for Multiple Instance Learning based Whole Slide Image Classification}, 
      author={Yunlong Zhang and Honglin Li and Yuxuan Sun and Jingxiong Li and Chenglu Zhu and Lin Yang},
      year={2024},
      eprint={2406.15303},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
