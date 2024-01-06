import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import stamp2date
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_AETA_cls(Dataset):
    def __init__(self, root_path, data_type='magn', fea_use='all', flag='train', seq_len=1008, label_len=144, pred_len=1008,
                 features='M', scale=False, inverse=False, timeenc=1, freq='t', sample='undersampling'):

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        assert flag in ['train', 'test']
        
        self.flag = flag
        self.data_type = data_type
        self.fea_use = fea_use
        
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path

        self.sample = sample
        self.__read_data__()

    def __read_data__(self):
        data_path = os.path.join(self.root_path, self.flag, 'data')
        label_path = os.path.join(self.root_path, self.flag, 'label')

        label_files = sorted(os.listdir(label_path))
        src_list = []
        lab_list = []
        src_stamp_list = []
        lab_stamp_list = []
        label_bin_list = []
        if self.flag == 'train':
            label_files = sorted(os.listdir(label_path))
            label_all = pd.DataFrame()
            for file in label_files:
                label_file = pd.read_csv(os.path.join(label_path, file))
                label_file.drop('Unnamed: 0', axis=1, inplace=True)
                label_all = pd.concat([label_all, label_file], axis=0)
            
            if self.sample == 'undersampling':
                label_eq = label_all[label_all['label'] == 1]
                label_noEq = label_all[label_all['label'] == 0].sample(n=len(label_eq.index.values), replace=False)
                label_train_all = pd.concat([label_eq, label_noEq])
            
            for i in range(len(label_train_all)):
                # Obtain binary labels for each sample
                label = label_train_all.iloc[i, 1]
                
                src_name = label_train_all.iloc[i, 0]
                trg_name_num = int(src_name.split('.')[0].split('_')[1]) + 144
                trg_name = src_name.split('_')[0] + '_' + str(trg_name_num) + '.csv'
                # Obtain the source sequence and its corresponding timestamp encoding
                src_all = pd.read_csv(os.path.join(data_path, src_name))
                src_all.drop('Unnamed: 0', axis=1, inplace=True)
                src = choose_input_fea(self.data_type, self.fea_use, src_all)
                src_stamp = src['TimeStamp']
                src_stamp = src_stamp.map(stamp2date)
                src_stamp2date = pd.to_datetime(src_stamp)
                src_data_stamp = time_features(src_stamp2date, timeenc=self.timeenc, freq=self.freq)

                
                label_bin_list.append(label)
                src_list.append(src.iloc[:, 2:].values)
                lab_list.append(src.iloc[-self.label_len:, 2:].values)
                src_stamp_list.append(src_data_stamp)
                lab_stamp_list.append(src_data_stamp[-self.label_len:, :])


        
        elif self.flag == 'test':
            for file in label_files:
                label_file = pd.read_csv(os.path.join(label_path, file))
                label_file.drop('Unnamed: 0', axis=1, inplace=True)
                for i in range(len(label_file.values)): 
                    # Obtain binary labels for each sample
                    label = label_file.iloc[i, 1]
                    # Obtain the source sequence and its corresponding timestamp encoding
                    src_all = pd.read_csv(os.path.join(data_path, label_file.iloc[i,0]))
                    src_all.drop('Unnamed: 0', axis=1, inplace=True)
                    src = choose_input_fea(self.data_type, self.fea_use, src_all)
                    src_stamp = src['TimeStamp']
                    src_stamp = src_stamp.map(stamp2date)
                    src_stamp2date = pd.to_datetime(src_stamp)
                    src_data_stamp = time_features(src_stamp2date, timeenc=self.timeenc, freq=self.freq)

     
                    label_bin_list.append(label)
                    src_list.append(src.iloc[:, 2:].values)
                    lab_list.append(src.iloc[-self.label_len:, 2:].values)
                    src_stamp_list.append(src_data_stamp)
                    lab_stamp_list.append(src_data_stamp[-self.label_len:, :])


        self.src_list = src_list
        self.lab_list = lab_list
        self.src_stamp_list = src_stamp_list
        self.lab_stamp_list = lab_stamp_list
        self.label_bin_list = label_bin_list
        

    def __getitem__(self, index):
        seq_x = self.src_list[index]
        seq_x_mark = self.src_stamp_list[index]

        label = self.label_bin_list[index]

        return seq_x, seq_x_mark, label
    
    def __len__(self):
        return len(self.src_list)



def choose_input_fea(data_type, fea_use, data):
    if data_type == 'magn':
        if fea_use == 'Fourier_power_0_15':
            fea_data = pd.concat([data.iloc[:, :2], data['magn@power_0_5'], data['magn@power_5_10'], data['magn@power_5_10']], axis=1)
    return fea_data
