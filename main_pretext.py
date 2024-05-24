import argparse
import os
import torch

from exp.exp_eq_fore import Exp_Eq_Fore
from utils.tools import mkdir
import random
import numpy as np

parser = argparse.ArgumentParser(description='Earthquake Forecasting')

parser.add_argument('--model', type=str, default='Eq_Fore',help='model of experiment')

parser.add_argument('--dataroot', type=str, default='./datasets/magn_all', help='path of data')
parser.add_argument('--data_type', type=str, default='magn') 
parser.add_argument('--cleaning', type=str, default='fill_0') 
parser.add_argument('--filling', type=str, default='linear_interpolate') 
parser.add_argument('--threshold_time', type=int, default=72) 
parser.add_argument('--norm_data', type=str, default='oneSta_oneFea') 
parser.add_argument('--norm_type', type=str, default='quartile_seg') 
parser.add_argument('--fea_select', type=str, default='all') 
parser.add_argument('--fea_use', type=str, default='Fourier_power_0_15')
parser.add_argument('--dataset_split_time', type=str, default='2022-01-01 00:00:00') 
parser.add_argument('--input_length', type=str, default='7days') 
parser.add_argument('--input_sel_type', type=str, default='Slide') 
parser.add_argument('--input_window_size', type=int, default=1008)
parser.add_argument('--predict_size', type=int, default=7) 
parser.add_argument('--class_type', type=str, default='binary_cls') 
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--sample', type=str, default='undersampling') 
parser.add_argument('--train_phase', type=str, default='train') 

parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=1008, help='input sequence length of encoder')
parser.add_argument('--label_len', type=int, default=144, help='start token length of decoder')
parser.add_argument('--pred_len', type=int, default=1008, help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
parser.add_argument('--c_out', type=int, default=3, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=2, help='train epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=3, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--seed', type=int, default=77, help='The random seed')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
print(random.random())

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Eq_Fore

for ii in range(args.itr):
    data_path = os.path.join(args.dataroot, args.data_type, args.cleaning, args.filling, args.norm_data, args.norm_type, args.fea_select,
                                 'Input_%s_%s_Output_%s' % (args.input_length, args.input_sel_type, args.class_type))
    checkpoint_path = os.path.join(args.checkpoints, args.data_type, args.cleaning, args.filling, args.norm_data,
                                   args.norm_type, args.fea_use,
                                   'Input_%d_%s_Output_%d' % (args.seq_len, args.input_sel_type, args.pred_len), args.model)
    mkdir(checkpoint_path)
    exp = Exp(args, data_path) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(checkpoint_path))
    exp.train(checkpoint_path)

    torch.cuda.empty_cache()
