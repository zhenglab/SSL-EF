import os
import pandas as pd
import argparse

from config import *
from toolkit import *


### Data cleaning, removing NaN from the data ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='magn') 
    parser.add_argument('--cleaning', type=str, default='fill_0') 
    args = parser.parse_args()
    print(args)

    # Only select the stations left after screening
    magn_all_files = pd.read_csv(os.path.join(Remaining_Sation_Path, 'magn.csv'))

    if args.data_type == 'magn':
        sta_name = magn_all_files['Station'].values
        for file in sta_name:
            magn_data = pd.read_csv(os.path.join(Magn_All_Save_Path, file))

            if args.cleaning == 'fill_0':
                magn_data_na = magn_data.fillna(0.)

            save_path = os.path.join(Magn_Cleaning_Path, args.cleaning)
            mkdir(save_path)
            magn_data.to_csv(os.path.join(save_path, file))
            print(file)



