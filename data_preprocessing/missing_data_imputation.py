import os
import argparse

import numpy as np
import pandas as pd

from config import *
from toolkit import *


def linear_interpolate(data, missing, threshold_time):
    for i in range(len(missing.values)-1, -1, -1):
        left_time = missing.iloc[i, 0]
        fill_num = int(missing.iloc[i, 2] / 10 - 1)

        if fill_num < threshold_time:
            # Create an empty dataframe with the same number of rows as the one to be added
            fill_nan = pd.DataFrame(np.full([fill_num, len(data.columns.values)], np.nan), columns=data.columns.values)
            # Determine the starting position of the insertion point and subsequently insert data
            left_data = data[data['TimeStamp'] == left_time]
            left_index = left_data.index.values[0]
            above = data.iloc[:left_index+1, :]
            below = data.iloc[left_index+1:, :]
            fill_data_nan = pd.concat([above, fill_nan, below], axis=0)
            data = fill_data_nan

    data_filling = data.interpolate()
    data_filling = data_filling.reset_index(drop=True)

    return data_filling


### Missing data imputation, rectifying data that is missing continuously for less than 12 hours ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='magn') 
    parser.add_argument('--cleaning', type=str, default='fill_0') 
    parser.add_argument('--filling', type=str, default='linear_interpolate') 
    parser.add_argument('--threshold_time', type=int, default=72) 
    args = parser.parse_args()
    print(args)

    # Obtain cleaned data
    if args.data_type == 'magn':
        magn_clean_files = sorted(os.listdir(os.path.join(Magn_Cleaning_Path, args.cleaning)))

    if args.data_type == 'magn':
        for file in magn_clean_files:
            magn_data = pd.read_csv(os.path.join(Magn_Cleaning_Path, args.cleaning, file))
            magn_data.drop('Unnamed: 0', axis=1, inplace=True)
            magn_missing = pd.read_csv(os.path.join(Magn_Missing_Path, file))
            
            # Missing data imputation
            if args.filling == 'linear_interpolate':
                filling_data = linear_interpolate(magn_data, magn_missing, args.threshold_time)

            save_path = os.path.join(Magn_Imputation_Path, args.cleaning, args.filling)
            mkdir(save_path)
            filling_data.to_csv(os.path.join(save_path, file))
            print(file)

