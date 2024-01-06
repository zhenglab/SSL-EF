import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

    
def stamp2date(stamp):
    timeArray = time.localtime(stamp)
    date = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return date

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def plot_loss(data, path, name):
    plt.figure(figsize=(15, 7))
    plot_x = np.linspace(1, len(data), len(data))
    plt.plot(plot_x, data, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    mkdir(path)
    plt.savefig(os.path.join(path, name))

def plot_metrics_seq2seq(data, path, name, metric):
    plt.figure(figsize=(15, 7))
    plot_x = np.linspace(1, len(data), len(data))
    plt.plot(plot_x, data, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    mkdir(path)
    plt.savefig(os.path.join(path, name))

def save_data(data, path, name):
    df = pd.DataFrame(data)
    mkdir(path)
    df.to_csv(os.path.join(path, name))

def save_seq2seq_gpu(path, name, args, train_num, test_num, params, flops):
    with open(os.path.join(path, name), 'a') as file:
        for arg, value in sorted(vars(args).items()):
            file.writelines('Argument %s: %r \n' % (arg, value))
        file.writelines('Train num: %d \n' % train_num)
        file.writelines('Test num: %d \n' % test_num)
        file.writelines('FLOPs: %.2fM \n' % (flops / 1e6))
        file.writelines('Params: %.5fM \n' % (params / 1e6))

        file.writelines('----------End--------------- \n')

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    return fpr

def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)
    return fnr

def save_txt_gpu_test(path, name, args, params, flops, train_label, test_label, memory_origin, memory_allocate, memory_reserved, memory_usage):
    with open(os.path.join(path, name), 'a') as file:
        for arg, value in sorted(vars(args).items()):
            file.writelines('Argument %s: %r \n' % (arg, value))
        file.writelines('Train_Num_eq: %d \n' % int(train_label[1]))
        file.writelines('Train_Num_noEq: %d \n' % int(train_label[0]))
        # test_label_value = test_label.values()
        file.writelines('Test_Num_eq: %d \n' % int(test_label[1]))
        file.writelines('Test_Num_noEq: %d \n' % int(test_label[0]))
        file.writelines('FLOPs: %.2fM \n' % (flops / 1e6))
        file.writelines('Params: %.5fM \n' % (params / 1e6))
        file.writelines('Memory_origin: %.5fM \n' % (memory_origin / 1e6))
        file.writelines('Memory_allocate: %.5fM \n' % (memory_allocate / 1e6))
        file.writelines('Memory_reserved: %.5fM \n' % (memory_reserved / 1e6))
        file.writelines('Memory_usage: %.5fM \n' % (memory_usage / 1e6))
        file.writelines('----------End--------------- \n')

def plot_metrics_one(data, path, name, metric):
    plt.figure(figsize=(15, 7))
    plot_x = np.linspace(1, len(data), len(data))
    plot_y = []
    for i in range(len(data)):
        plot_y.append(data[i][1])
    plt.plot(plot_x, plot_y, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    mkdir(path)
    plt.savefig(os.path.join(path, name))

def plot_metrics_two(data, path, name, metric):
    noEq = []
    Eq = []
    for i in range(len(data)):
        noEq.append(data[i][1])
        Eq.append(data[i][2])

    plt.figure(figsize=(30, 7))

    plt.subplot(121)
    plot_noEq = np.linspace(1, len(noEq), len(noEq))
    plt.plot(plot_noEq, noEq, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(metric + '-' + 'Aseismic')

    plt.subplot(122)
    plot_Eq = np.linspace(1, len(Eq), len(Eq))
    plt.plot(plot_Eq, Eq, marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(metric + '-' + 'Earthquake')

    mkdir(path)
    plt.savefig(os.path.join(path, name))