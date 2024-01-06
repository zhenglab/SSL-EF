# SSL-EF

This repository provides the official PyTorch implementation of our paper "Short-Term Earthquake Forecasting via Self-Supervised Learning".

## Prerequisites

- Linux
- Python 3.7
- NVIDIA GPU + CUDA CuDNN

## Datasets

- Download [AETA](https://platform.aeta.cn/zh-CN/competitionpage/download) dataset and [earthquake catalog](https://news.ceic.ac.cn/index.html?time=1704271080).

- All downloaded electromagnetic data are stored in the `datasets/magn_all` directory, comprising 159 CSV files. Each file within this directory contains the electromagnetic data from a single observation station.

## Data Preprocessing

- The data preprocessing includes several crucial steps: station selection, data cleaning, missing data imputation, data normalization, and dataset construction.

- Perform data preprocessing using the following script.

```bash
bash data_preporcessing/magn.sh
```

## Pretext Task

- We design the prediction task as a pretext task, leveraging the past week's observational data to predict the coming week's data.

- To do the pretext prediction task on a large-scale dataset composed of all samples, run:

```bash
python main_pretext.py
```

## Downstream Task

- We set the classification task as a downstream task, focusing on whether a major earthquake occurs in the coming week.

- To do the downstream classification task on a small-scale yet balanced dataset built through undersampling, run:

```bash
python main_downstream.py
```