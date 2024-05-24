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

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    
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

def save_seq2seq_gpu(path, name, args, train_num, test_num, params, flops, num_params, iter_time, test_time):
    with open(os.path.join(path, name), 'a') as file:
        for arg, value in sorted(vars(args).items()):
            file.writelines('Argument %s: %r \n' % (arg, value))
        file.writelines('Train num: %d \n' % train_num)
        file.writelines('Test num: %d \n' % test_num)
        file.writelines('FLOPs: %.5fM \n' % (flops / 1e6))
        file.writelines('Params: %.5fM \n' % (params / 1e6))
        file.writelines('Params_Nums: %f \n' % num_params)
        file.writelines('One Iteration Time: %fs \n' % (iter_time))
        file.writelines('One Sample Test Time: %fs \n' % (test_time))


        file.writelines('----------End--------------- \n')


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    return fpr


def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)
    return fnr

def save_txt_gpu_test(path, name, args, train_label, test_label, params, flops, num_params, memory_origin, memory_allocate, memory_reserved, memory_usage, iter_time, test_time):
    with open(os.path.join(path, name), 'a') as file:
        for arg, value in sorted(vars(args).items()):
            file.writelines('Argument %s: %r \n' % (arg, value))
        file.writelines('Train Num_eq: %d \n' % int(train_label[1]))
        file.writelines('Train Num_noEq: %d \n' % int(train_label[0]))
        file.writelines('Test Num_eq: %d \n' % int(test_label[1]))
        file.writelines('Test Num_noEq: %d \n' % int(test_label[0]))
        file.writelines('FLOPs: %.5fM \n' % (flops / 1e6))
        file.writelines('Params: %.5fM \n' % (params / 1e6))
        file.writelines('Num Params: %f \n' % num_params)

        file.writelines('Memory_origin: %.5fM \n' % (memory_origin / 1e6))
        file.writelines('Memory_allocate: %.5fM \n' % (memory_allocate / 1e6))
        file.writelines('Memory_reserved: %.5fM \n' % (memory_reserved / 1e6))
        file.writelines('Memory_usage: %.5fM \n' % (memory_usage / 1e6))

        file.writelines('One Iteration Time: %fs \n' % iter_time)
        file.writelines('One Sample Test Time: %fs \n' % test_time)
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