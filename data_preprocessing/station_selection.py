import os
import pandas as pd

from config import *
from toolkit import *


if __name__ == '__main__':
    magn_all_files = sorted(os.listdir(Magn_All_Save_Path))

    # Deleted stations
    magn_delete_sta = ['119_magn.csv', '59_magn.csv', '229_magn.csv', '82_magn.csv',
                       '32_magn.csv', '84_magn.csv', '132_magn.csv']
    
    for sta in magn_delete_sta:
        magn_all_files.remove(sta)
    magn_all_files_df = pd.DataFrame(magn_all_files, columns=['Station'])
    mkdir(Remaining_Sation_Path)
    magn_all_files_df.to_csv(os.path.join(Remaining_Sation_Path, 'magn.csv'))






