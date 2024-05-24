#!/usr/bin/env bash

python ../main_pretext.py --gpu 0 --seed 77 --data_type magn --model Eq_Fore --cleaning fill_0 --filling linear_interpolate --norm_data oneSta_oneFea --norm_type quartile_seg --fea_select all --fea_use Fourier_power_0_15 --input_window_size 1008 --input_sel_type Slide --train_epochs 30 --batch_size 16 --checkpoints ../checkpoints --enc_in 3 --dec_in 3 --c_out 3

