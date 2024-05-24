# SSL-EF

This repository provides the official PyTorch implementation of our paper "Short-Term Earthquake Forecasting via Self-Supervised Learning".

## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- python 3.7.16
- cudatoolkit 11.1.1
- torch 1.13.1
- torchvision 0.9.0
- numpy 1.21.5
- scikit-learn 1.0.2

## Quick Example

- Download the preprocessed [test set](https://drive.google.com/file/d/1L2mynxrl7gvEsye7YofKbLMnJJPq8Bkj/view?usp=sharing) and put it in the `datasets/` directory.

- Download the pre-trained [model](https://drive.google.com/file/d/1fiJU8tGqdX9yBAgVsi05TkRg2Zx4BP4D/view?usp=drive_link) and put it in the `results/` directory.

- To do the quick test, run:

```bash
python downstream_test.py
```

## Datasets

- Download [AETA](https://platform.aeta.cn/zh-CN/competitionpage/download) dataset and [earthquake directory](https://news.ceic.ac.cn/index.html?time=1704271080).

- All downloaded electromagnetic data are stored in the `datasets/` directory, comprising 159 CSV files. Each file within this directory contains the electromagnetic data from a single observation station.

## Data Preprocessing

- The data preprocessing includes several crucial steps: station selection, data cleaning, missing data imputation, data normalization, and dataset construction.

- Perform data preprocessing using the following script.

```bash
cd data_preporcessing
bash magn.sh
```

## Pretext Task

- We design the prediction task as a pretext task, leveraging the past week's observational data to predict the coming week's data.

- To do the pretext prediction task on a large-scale dataset composed of all samples, run:

```bash
cd scripts
bash pre.sh
```

## Downstream Task

- We set the classification task as a downstream task, focusing on whether a major earthquake occurs in the coming week.

- To do the downstream classification task on a small-scale yet balanced dataset built through undersampling, run:

```bash
cd scripts
bash cls.sh
```