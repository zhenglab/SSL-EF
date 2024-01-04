import os
import argparse

import numpy as np
import pandas as pd

from config import *
from toolkit import *

from collections import Counter

def feature_selection(data, data_type, fea_select):
    if data_type == 'magn':
        if fea_select == 'Fourier_power_0_15':
            data_sel = pd.concat([data.iloc[:, :2], data['magn@power_0_5'], data['magn@power_5_10'], data['magn@power_5_10']], axis=1)

    return data_sel


def slide_input(data, window_size, step_size, input_path, cleaning_type, filling_type, norm_data,
                norm_type, input_length, fea_select, input_sel_type, class_type, train_phase, station_id):
    timestamp = []
    for i in range(0, len(data.index.values) - window_size + 1, step_size):
        tmp = data[i:i + window_size]
        # Save data, save each set of data to a CSV file
        save_path = os.path.join(input_path, cleaning_type, filling_type, norm_data, norm_type, fea_select,
                                 'Input_%s_%s_Output_%s' % (input_length, input_sel_type, class_type), train_phase,
                                 'data')
        mkdir(save_path)
        tmp.to_csv(os.path.join(save_path, '%d_%d.csv' % (station_id, i)))

        # Save the first and last timestamps of each group
        timestamp.append(['%d_%d.csv' % (station_id, i), tmp['TimeStamp'].iloc[0], tmp['TimeStamp'].iloc[-1]])  # 保存每一组最后一个时间戳，据此对未来一段时间天内的地震情况做标签
        print(i)
    return timestamp


# Perform binary labeling - only label earthquakes with magnitudes 5 and above
def label_binary(max_mag, distance):
    if (max_mag >=5) & (max_mag <= 6) & (distance < 100):
        label = 1
    elif (max_mag > 6) & (distance < 200):
        label = 1
    else:
        label = 0
    return label


def get_label_pred(station, eq_list, last_timestamp, class_type, predict_size):
    label_list = []
    label_only = []
    for i in range(len(last_timestamp)):
        eq_range_left = last_timestamp[i][1]
        eq_range_right = last_timestamp[i][1] + predict_size * 24 * 60 * 60
        eq_sel = eq_list[(eq_list['TimeStamp'] > eq_range_left) & (eq_list['TimeStamp'] < eq_range_right)]

        # How to label if an earthquake occurs within the next week
        if len(eq_sel.values) > 0:
            eq_max_mag = np.max(eq_sel['Magnitude'].to_numpy()) 
            eq_sel_mag = eq_sel[eq_sel['Magnitude'] == eq_max_mag]  

            # Calculate the distance between the location of the maximum magnitude and the station, and take the minimum epicenter distance as the final result
            distance_min = get_distance(station['Longitude'], station['Latitude'], eq_sel_mag['Longitude'].iloc[0],
                                        eq_sel_mag['Latitude'].iloc[0])  
            # When there are multiple earthquakes with the same maximum magnitude, choose the earthquake with the smallest epicenter distance
            if len(eq_sel_mag.values) > 1:
                for j in range(1, len(eq_sel_mag.values)):
                    distance = get_distance(station['Longitude'], station['Latitude'], eq_sel_mag['Longitude'].iloc[j],
                                            eq_sel_mag['Latitude'].iloc[j])
                    if distance < distance_min:
                        distance_min = distance

            # Label based on epicenter distance and magnitude
            if class_type == 'binary_cls':
                label = label_binary(eq_max_mag, distance_min)  

            label_list.append([last_timestamp[i][0], label])
            label_only.append(label)
            print(i)
            
        # How to label if there is no earthquake in the next week
        else:
            label_list.append([last_timestamp[i][0], 0])
            label_only.append(0)
            print(i)

    return pd.DataFrame(label_list, columns=['file_name', 'label']), label_only



def get_test_data(tmp, test_data, input_path, cleaning_type, filling_type, norm_data, norm_type, fea_select,
                  input_length, input_sel_type, class_type, station_id):
    timestamp = []
    for i in range(len(tmp.index.values) - 1):
        normal_timestamp = np.array(tmp['Normal_TimeStamp'])
        date_1 = tmp['Normal_TimeStamp'].iloc[i]
        date_2 = tmp['Normal_TimeStamp'].iloc[i + 1]

        if (date_1 in normal_timestamp) & (date_2 in normal_timestamp):
            week_data_tmp = test_data[(test_data['TimeStamp'] >= date_1) & (test_data['TimeStamp'] < date_2)]
            # Check for missing data and only retain weeks without missing data
            if len(week_data_tmp.index.values) == 1008:
                # Save data, save each set of data to a CSV file
                save_path = os.path.join(input_path, cleaning_type, filling_type, norm_data, norm_type, fea_select,
                                         'Input_%s_%s_Output_%s' % (input_length, input_sel_type, class_type),
                                         'test', 'data')
                mkdir(save_path)
                week_data_tmp.to_csv(os.path.join(save_path, '%d_%d.csv' % (station_id, i)))
                timestamp.append(['%d_%d.csv' % (station_id, i), week_data_tmp['TimeStamp'].iloc[0], week_data_tmp['TimeStamp'].iloc[-1]])

    return timestamp


def data_input(data, data_type, dataset_split_time, window_size, step_size, eq_list, sta_info, predict_size, input_path,
               cleaning_type, filling_type,
               norm_data, norm_type, input_length, fea_select, input_sel_type, class_type, train_phase):
    # Determine which station to process at this time and the corresponding station information
    station_id = data['StationID'].iloc[0]
    station = sta_info[sta_info['StationID'] == station_id]

    # Add a new column to convert the time in the earthquake directory to a timestamp
    eq_list['TimeStamp'] = eq_list['Time'].apply(lambda x: string2stamp(x))

    # Choose which features to use
    data = feature_selection(data, data_type, fea_select)

    # Split the dataset into training and testing sets according to the set time
    dataset_split_timeStamp = string2stamp(dataset_split_time)
    train_data = data[data['TimeStamp'] < dataset_split_timeStamp]
    test_data = data[data['TimeStamp'] >= dataset_split_timeStamp]

    if train_phase == 'train':
        if input_sel_type == 'Slide':
            train_timestamp = slide_input(train_data, window_size, step_size, input_path, cleaning_type,
                                               filling_type, norm_data, norm_type, input_length, fea_select,
                                               input_sel_type, class_type, train_phase, station_id)
            
        tarin_label, train_label_only = get_label_pred(station, eq_list, train_timestamp, class_type, predict_size)
        # a = Counter(train_label_only)

        save_path = os.path.join(input_path, cleaning_type, filling_type, norm_data, norm_type, fea_select,
                                 'Input_%s_%s_Output_%s' % (input_length, input_sel_type, class_type), train_phase,
                                 'label')
        mkdir(save_path)
        tarin_label.to_csv(os.path.join(save_path, '%d.csv' % station_id))

    elif train_phase == 'test':
        # How many weeks did this station last from January 1, 2022 to the end of the recording period
        if len(test_data.index.values) > 0:
            first_timestamp = test_data['TimeStamp'].iloc[0]
            last_timestamp = test_data['TimeStamp'].iloc[-1]
            weeks = (last_timestamp - first_timestamp) / (86400 * 7)

            # Obtain the timestamp of midnight every Saturday and predict the earthquake situation for the next week
            tmp_df = pd.DataFrame(np.arange(0, weeks + 1, 1), columns=['weeks'])
            tmp_df['TimeStamp'] = first_timestamp
            tmp_df['Normal_TimeStamp'] = tmp_df['TimeStamp'] + (tmp_df['weeks'] * 86400 * 7)

            # Retrieve data from midnight every Saturday to midnight next Saturday, and return the timestamp of 23:50 next Friday
            test_last_timestamp = get_test_data(tmp_df, test_data, input_path, cleaning_type, filling_type, norm_data,
                                                norm_type,
                                                fea_select, input_length, input_sel_type, class_type, station_id)
            # Use the last timestamp of each group for labeling operations
            test_label, test_label_only = get_label_pred(station, eq_list, test_last_timestamp, class_type, predict_size)
            # a = Counter(test_label_only)

            save_path = os.path.join(input_path, cleaning_type, filling_type, norm_data, norm_type, fea_select,
                                     'Input_%s_%s_Output_%s' % (input_length, input_sel_type, class_type), train_phase,
                                     'label')
            mkdir(save_path)
            test_label.to_csv(os.path.join(save_path, '%d.csv' % station_id))


### Build datasets ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='magn')  
    parser.add_argument('--cleaning', type=str, default='fill_0') 
    parser.add_argument('--filling', type=str,
                        default='linear_interpolate')  
    parser.add_argument('--threshold_time', type=int, default=72)  
    parser.add_argument('--norm_data', type=str, default='oneSta_oneFea')
    parser.add_argument('--norm_type', type=str, default='quartile_seg')

    parser.add_argument('--fea_select', type=str, default='Fourier_power_0_15')  
    parser.add_argument('--dataset_split_time', type=str, default='2022-01-01 00:00:00')  
    parser.add_argument('--input_length', type=str, default='7days')  
    parser.add_argument('--input_sel_type', type=str, default='Slide')  
    parser.add_argument('--input_window_size', type=int, default=144 * 7)  
    parser.add_argument('--step_size', type=int, default=144)  
    parser.add_argument('--predict_size', type=int, default=7)  
    parser.add_argument('--class_type', type=str, default='binary_cls')  
    parser.add_argument('--train_phase', type=str, default='train')  # 'train' or 'test'
    args = parser.parse_args()
    print(args)

    eq_list = pd.read_csv(Eq_Path)
    sta_info = pd.read_csv(Station_Path)

    if args.data_type == 'magn':
        magn_all_files = sorted(
            os.listdir(os.path.join(Magn_Norm_Path, args.cleaning, args.filling, args.norm_data, args.norm_type)))
        for file in magn_all_files:
            magn = pd.read_csv(
                os.path.join(Magn_Norm_Path, args.cleaning, args.filling, args.norm_data, args.norm_type, file))
            magn.drop('Unnamed: 0', axis=1, inplace=True)
            # magn.dropna(axis=0, how='any', inplace=True)

            data_input(magn, args.data_type, args.dataset_split_time, args.input_window_size, args.step_size, eq_list,
                       sta_info, args.predict_size, Magn_Input_Path, args.cleaning,
                       args.filling, args.norm_data, args.norm_type, args.input_length, args.fea_select,
                       args.input_sel_type, args.class_type, args.train_phase)



