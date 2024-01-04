# SSL-EF

This repository provides the official PyTorch implementation of our paper "Short-Term Earthquake Forecasting via Self-Supervised Learning".

## Prerequisites

- Linux
- Python 3.7
- NVIDIA GPU + CUDA CuDNN

## Datasets

- Download [AETA](https://platform.aeta.cn/zh-CN/competitionpage/download) dataset and and [earthquake directory](https://news.ceic.ac.cn/index.html?time=1704271080).

- All downloaded electromagnetic data is placed in the `datasets/magn_all`, and a CSV file contains all the data of a station.

## Data Preprocessing

- The data preprocessing includes several crucial steps: station selection, data cleaning, missing data imputation, data normalization, and dataset construction.

```bash
bash data_preporcessing/magn.sh
```

## Pretext Task

- We design the prediction task as a pretext task, leveraging the past week's observational data to predict the coming week's data.

```bash
python main_pretext.py
```

## Downstream Task

- We set the classification task as a downstream task, focusing on whether a major earthquake occurs in the coming week.

```bash
python main_downstream.py
```