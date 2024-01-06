import os
import argparse

import numpy as np
import pandas as pd

from config import *
from toolkit import *



def minmax_norm(df):
    return (df - df.min()) / ( df.max() - df.min())


def mean_norm(df):
    return (df - df.mean()) / df.std()


def comp_boundary_seg(data, theta):
    quartile = np.nanpercentile(data, (25, 50, 75), axis=0, interpolation='midpoint')
    IQR = quartile[2] - quartile[0]
    lower_boundary = quartile[0] - 1.5 * IQR
    upper_boundary = quartile[2] + 1.5 * IQR
    # Determine the boundary of outliers
    ultra_lower_bound = quartile[0] - theta * IQR
    ultra_upper_bound = quartile[2] + theta * IQR
    return lower_boundary, upper_boundary, ultra_lower_bound, ultra_upper_bound


def mapping(lower_boun, upper_boun, data_max, data_min, data):
    return ((upper_boun - lower_boun) / (data_max - data_min)) * (data - data_min) + lower_boun


def quartile_oneFea_seg_norm(df):
    theta = 1.8
    data = np.array(df)
    lower_boun, upper_boun, ultra_lower_boun, ultra_upper_boun = comp_boundary_seg(data, theta)

    # Identify values above the upper limit in the raw data
    tmp_upper = df[df > upper_boun]
    tmp_upper_max = tmp_upper.max()
    tmp_upper_min = tmp_upper.min()
    # Identify values below the lower limit in the raw data
    tmp_lower = df[df < lower_boun]
    tmp_lower_max = tmp_lower.max()
    tmp_lower_min = tmp_lower.min()
    # Identify values within the normal range in the raw data
    tmp_normal = df[(df > lower_boun) & (df < upper_boun)]
    tmp_normal_max = tmp_normal.max()
    tmp_normal_min = tmp_normal.min()

    # The raw data is divided into three segments and then mapped respectively.
    if tmp_upper.size != 0:
        tmp_upper_norm = mapping(upper_boun, ultra_upper_boun, tmp_upper_max, tmp_upper_min, tmp_upper)
    else:
        tmp_upper_norm = pd.Series([])
    if tmp_lower.size != 0:
        tmp_lower_norm = mapping(ultra_lower_boun, lower_boun, tmp_lower_max, tmp_lower_min, tmp_lower)
    else:
        tmp_lower_norm = pd.Series([])
    if tmp_normal.size != 0:
        tmp_normal_norm = mapping(lower_boun, upper_boun, tmp_normal_max, tmp_normal_min, tmp_normal)
    else:
        tmp_normal_norm = pd.Series([])
    combined = pd.concat([tmp_upper_norm, tmp_lower_norm, tmp_normal_norm])
    data_norm = combined.sort_index()

    return data_norm


def norm_oneSta_oneFea(data, norm_type):
    features_name = data.columns.values[2:]
    data_norm = data.iloc[:, :2]
    for i in range(len(features_name)):
        tmp = data[features_name[i]]
        if norm_type == 'min_max':
            tmp_norm = minmax_norm(tmp)
        elif norm_type == 'z_score':
            tmp_norm = mean_norm(tmp)
        elif norm_type == 'quartile_seg':
            tmp_norm = quartile_oneFea_seg_norm(tmp)
        tmp_norm_df = pd.DataFrame(tmp_norm, columns=[features_name[i]])
        data_norm = pd.concat([data_norm, tmp_norm_df], axis=1)

    return data_norm



### Data normalization ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='merge')
    parser.add_argument('--cleaning', type=str, default='fill_0') 
    parser.add_argument('--filling', type=str, default='linear_interpolate') 
    parser.add_argument('--threshold_time', type=int, default=72) 
    parser.add_argument('--norm_data', type=str, default='oneSta_oneFea')
    parser.add_argument('--norm_type', type=str, default='quartile_seg')
    args = parser.parse_args()
    print(args)

    if args.data_type == 'magn':
        magn_fill_files = sorted(os.listdir(os.path.join(Magn_Imputation_Path, args.cleaning, args.filling)))
    
    if args.data_type == 'magn':
        if args.norm_data == 'oneSta_oneFea':
            for file in magn_fill_files:
                magn_data = pd.read_csv(os.path.join(Magn_Imputation_Path, args.cleaning, args.filling, file))
                magn_data.drop('Unnamed: 0', axis=1, inplace=True)
                
                # Data normalization
                magn_norm = norm_oneSta_oneFea(magn_data, args.norm_type)

                save_path = os.path.join(Magn_Norm_Path, args.cleaning, args.filling, args.norm_data, args.norm_type)
                mkdir(save_path)
                magn_norm.to_csv(os.path.join(save_path, file))
                print(file)
        
