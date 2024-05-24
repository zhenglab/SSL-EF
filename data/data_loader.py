import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler, stamp2date
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_AETA(Dataset):
    def __init__(self, root_path, data_type='magn', fea_use='all', flag='train', seq_len=1008, label_len=252, pred_len=1008,
                 features='M', scale=False, inverse=False, timeenc=1, freq='t'):

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        # init
        assert flag in ['train', 'test']
        
        self.flag = flag
        self.data_type = data_type
        self.fea_use = fea_use
        
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        data_path = os.path.join(self.root_path, self.flag, 'data')
        label_path = os.path.join(self.root_path, self.flag, 'label')

        
        label_files = sorted(os.listdir(label_path))
        src_list = []
        lab_list = []
        trg_list = []
        src_stamp_list = []
        lab_stamp_list = []
        trg_stamp_list = []
        if self.flag == 'train':
            for file in label_files:
                label_file = pd.read_csv(os.path.join(label_path, file))
                label_file.drop('Unnamed: 0', axis=1, inplace=True)
                for i in range(len(label_file.values)-8): 
                    src_all = pd.read_csv(os.path.join(data_path, label_file.iloc[i,0]))
                    src_all.drop('Unnamed: 0', axis=1, inplace=True)
                    src = choose_input_fea(self.data_type, self.fea_use, src_all)
                    src_stamp = src['TimeStamp']
                    src_stamp = src_stamp.map(stamp2date)
                    src_stamp2date = pd.to_datetime(src_stamp)
                    src_data_stamp = time_features(src_stamp2date, timeenc=self.timeenc, freq=self.freq)

                    src_list.append(src.iloc[:, 2:].values)
                    lab_list.append(src.iloc[-self.label_len:, 2:].values)
                    src_stamp_list.append(src_data_stamp)
                    lab_stamp_list.append(src_data_stamp[-self.label_len:, :])

                    trg_all = pd.read_csv(os.path.join(data_path, label_file.iloc[i+7,0]))
                    trg_all.drop('Unnamed: 0', axis=1, inplace=True)
                    trg = choose_input_fea(self.data_type, self.fea_use, trg_all)
                    
                    trg_stamp = trg['TimeStamp']
                    trg_stamp = trg_stamp.map(stamp2date)
                    trg_stamp2date = pd.to_datetime(trg_stamp)
                    trg_data_stamp = time_features(trg_stamp2date, timeenc=self.timeenc, freq=self.freq)
                    
                    trg_list.append(trg.iloc[:, 2:].values)
                    trg_stamp_list.append(trg_data_stamp)
        
        elif self.flag == 'test':
            for file in label_files:
                label_file = pd.read_csv(os.path.join(label_path, file))
                label_file.drop('Unnamed: 0', axis=1, inplace=True)
                for i in range(len(label_file.values)-1): 
                    src_all = pd.read_csv(os.path.join(data_path, label_file.iloc[i,0]))
                    src_all.drop('Unnamed: 0', axis=1, inplace=True)
                    src = choose_input_fea(self.data_type, self.fea_use, src_all)
                    src_stamp = src['TimeStamp']
                    src_stamp = src_stamp.map(stamp2date)
                    src_stamp2date = pd.to_datetime(src_stamp)
                    src_data_stamp = time_features(src_stamp2date, timeenc=self.timeenc, freq=self.freq)

                    src_name = label_file.iloc[i,0]
                    trg_name = label_file.iloc[i+1,0]

                    if int(src_name.split('.')[0].split('_')[1]) + 1 == int(trg_name.split('.')[0].split('_')[1]):
                        trg_all = pd.read_csv(os.path.join(data_path, label_file.iloc[i+1,0]))
                        trg_all.drop('Unnamed: 0', axis=1, inplace=True)
                        trg = choose_input_fea(self.data_type, self.fea_use, trg_all)
                        
                        trg_stamp = trg['TimeStamp']
                        trg_stamp = trg_stamp.map(stamp2date)
                        trg_stamp2date = pd.to_datetime(trg_stamp)
                        trg_data_stamp = time_features(trg_stamp2date, timeenc=self.timeenc, freq=self.freq)
                        
                        src_list.append(src.iloc[:, 2:].values)
                        lab_list.append(src.iloc[-self.label_len:, 2:].values)
                        src_stamp_list.append(src_data_stamp)
                        lab_stamp_list.append(src_data_stamp[-self.label_len:, :])

                        trg_list.append(trg.iloc[:, 2:].values)
                        trg_stamp_list.append(trg_data_stamp)

        self.src_list = src_list
        self.lab_list = lab_list
        self.trg_list = trg_list
        self.src_stamp_list = src_stamp_list
        self.lab_stamp_list = lab_stamp_list
        self.trg_stamp_list = trg_stamp_list

    def __getitem__(self, index):
        seq_x = self.src_list[index]
        seq_x_mark = self.src_stamp_list[index]

        seq_y = np.concatenate((self.lab_list[index], self.trg_list[index]), axis=0)
        seq_y_mark = np.concatenate((self.lab_stamp_list[index], self.trg_stamp_list[index]), axis=0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.src_list)


def choose_input_fea(data_type, fea_use, data):
    if data_type == 'magn':
        if fea_use == 'Fourier_power_0_15':
            fea_data = pd.concat([data.iloc[:, :2], data['magn@power_0_5'], data['magn@power_5_10'], data['magn@power_10_15']], axis=1)
        return fea_data