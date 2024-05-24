import os
import pandas as pd
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile
from collections import Counter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from data.dataset import *
from utils.tools import *
from models.model import Eq_Fore, BiLSTM, Info_Cls
from sklearn.metrics import accuracy_score, roc_curve, auc


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets', help='path of data')
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

parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.00001, help='initial (base) learning rate')
parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='path for saving result models')
parser.add_argument('--results', type=str, default='./results', help='path for saving result models')
parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
parser.add_argument('--hidden_nc', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--optimizer', type=str, default='Adam', help='the optimizer: SGD|Adam')
parser.add_argument('--model_pre', type=str, default='Eq_Fore', help='network: Eq_Fore')
parser.add_argument('--model_cls', type=str, default='BiLSTM', help='network: MLP | BiLSTM')
parser.add_argument('--model_pred_state', type=str, default='resume')


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
parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--seed', type=int, default=77, help='The random seed')



def main():
    args = parser.parse_args()
    print(args.data_type)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(random.random())

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

   
    data_path = os.path.join(args.dataroot, args.data_type, args.cleaning, args.filling, args.norm_data, args.norm_type, args.fea_select,
                                 'Input_%s_%s_Output_%s' % (args.input_length, args.input_sel_type, args.class_type))
    # results_path = os.path.join(args.results, args.data_type, args.cleaning, args.filling, args.norm_data,
    #                                args.norm_type, args.fea_use,
    #                                'Input_%d_%s_Output_%s_%s' % (args.seq_len, args.input_sel_type, args.class_type, args.sample), args.model_cls, args.model_pred_state)
    results_path = args.results

    test_data, test_loader = get_data(args, data_path, 'test')

    if args.model_pre == 'Eq_Fore':
        device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
        model_pre = Eq_Fore(args.enc_in, args.dec_in, args.c_out, args.seq_len, args.label_len, args.pred_len, 
                         args.factor, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, 
                         args.dropout, args.attn, args.embed, args.freq, args.activation, args.output_attention, 
                         args.distil, args.mix, device).float()

    if args.model_cls == 'BiLSTM':
        model_cls = BiLSTM(args.d_model, args.hidden_nc, args.num_layers, args.num_classes, args)    
    
    model = Info_Cls(model_pre, model_cls)

    result = torch.load(os.path.join(results_path, 'results.pth'))
    model.load_state_dict(result)
    

    # set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        print('Memory_origin:', torch.cuda.memory_allocated(args.gpu))
    print(model)



    true_label, predictions, model_output, test_time  = test(test_loader, model, args)
    accuracy, fp, fn, auc_score = evaluate(true_label, predictions, model_output)


    print('Accuracy: %.3f' % accuracy, 'AUC: %.3f' % auc_score, 'FNR: %.3f' % fn, 'FPR: %.3f' % fp)
    print(111)

    
def get_data(args, data_path, flag):
        timeenc = 0 if args.embed!='timeF' else 1

        data_set = Dataset_AETA_cls(data_path, args.data_type, args.fea_use, flag, args.seq_len, args.label_len,
                                    args.pred_len, args.features, False, False, timeenc, 
                                    args.freq, args.sample)

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq


        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


def test(test_loader, model, args):
    model.eval()

    test_time_all = []
    true_label = []
    predictions = []
    model_output = []
    with torch.no_grad():
        for i, (batch_x, batch_x_mark, labels) in enumerate(test_loader, 0):
            test_time_start = time.time()
            
            batch_x = batch_x.to(torch.float32)
            batch_x_mark = batch_x_mark.to(torch.float32)
            labels = labels.to(torch.long)

            
            if args.gpu is not None:
                batch_x = batch_x.cuda(args.gpu)
                batch_x_mark = batch_x_mark.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            outputs = model(batch_x, batch_x_mark)

            test_time_end = time.time() - test_time_start
            test_time_all.append(test_time_end)

            
            pred = torch.argmax(outputs, dim=1)

            true_label.append(labels.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            model_output.append(outputs.cpu().numpy())


    true_label = np.concatenate(true_label, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    model_output = np.concatenate(model_output, axis=0)
    correct = torch.sum(torch.tensor(predictions) == torch.tensor(true_label)).item()  
    total = len(true_label)
    print('Accuracy: %.2f %%' % (100 * correct / total))
    test_time = np.mean(np.array(test_time_all))
    print('test time:{}'.format(test_time))

    return true_label, predictions, model_output, test_time


def evaluate(label, pred, output):
    accuracy = accuracy_score(label, pred)
    fp = false_positive_rate(label, pred) 
    fn = false_negative_rate(label, pred) 
    fpr, tpr, thresholds = roc_curve(label, output[:, 1], pos_label=1) 
    auc_score = auc(fpr, tpr)
    return accuracy, fp, fn, auc_score


if __name__ == '__main__':
    main()


