from data.data_loader import Dataset_AETA
from exp.exp_basic import Exp_Basic
from models.model import Eq_Fore

from utils.tools import *
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from thop import profile
import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Eq_Fore(Exp_Basic):
    def __init__(self, args, data_path):
        super(Exp_Eq_Fore, self).__init__(args)
        self.data_path = data_path

    def _build_model(self):
        model_dict = {
            'Eq_Fore':Eq_Fore,
        }
        model = model_dict[self.args.model](
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.label_len,
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        """ number of parameters """
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(self.num_params))
        
        return model

    def _get_data(self, flag):
        args = self.args
        timeenc = 0 if args.embed!='timeF' else 1

        data_set = Dataset_AETA(self.data_path, self.args.data_type, self.args.fea_use, flag, self.args.seq_len, self.args.label_len, self.args.pred_len, 
                            self.args.features, False, False, timeenc, self.args.freq)

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
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

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

            total_loss = np.average(total_loss)
            preds = np.vstack(preds)
            trues = np.vstack(trues)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))

            self.model.train()
        return total_loss, mae, mse, rmse, mape, mspe

    def train(self, checkpoint_path):
        train_data, train_loader = self._get_data(flag = 'train')
        test_data, test_loader = self._get_data(flag = 'test')

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_loss_list = []
        test_loss_list = []

        train_mae_list = []
        train_mse_list = []
        train_rmse_list = []
        train_mape_list = []
        train_mspe_list = []

        test_mae_list = []
        test_mse_list = []
        test_rmse_list = []
        test_mape_list = []
        test_mspe_list = []

        best_mse = 1e9

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_preds = []
            train_trues = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                train_preds.append(pred.detach().cpu().numpy())
                train_trues.append(true.detach().cpu().numpy())
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            adjust_learning_rate(model_optim, epoch+1, self.args)

            train_loss = np.average(train_loss)
            train_loss_list.append(train_loss)
            train_preds = np.array(train_preds)
            train_trues = np.array(train_trues)
            train_preds = train_preds.reshape(-1, train_preds.shape[-2], train_preds.shape[-1])
            train_trues = train_trues.reshape(-1, train_trues.shape[-2], train_trues.shape[-1])
            train_mae, train_mse, train_rmse, train_mape, train_mspe = metric(train_preds, train_trues)
            print('train_mae:{}, train_mse:{}'.format(train_mae, train_mse))
            train_mae_list.append(train_mae)
            train_mse_list.append(train_mse)
            train_rmse_list.append(train_rmse)
            train_mape_list.append(train_mape)
            train_mspe_list.append(train_mspe)

            test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe = self.vali(test_data, test_loader, criterion)
            test_loss_list.append(test_loss)
            test_mae_list.append(test_mae)
            test_mse_list.append(test_mse)
            test_rmse_list.append(test_rmse)
            test_mape_list.append(test_mape)
            test_mspe_list.append(test_mspe)

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))

            # remember best mse and save checkpoint
            is_best = test_mse < best_mse
            best_mse = min(test_mse, best_mse)
            if is_best:
                best_model_path = checkpoint_path +'/'+'checkpoint.pth'
                torch.save(self.model.state_dict(), best_model_path)

        self.model.load_state_dict(torch.load(best_model_path))

        plot_loss(train_loss_list, os.path.join(checkpoint_path, 'figs'), 'train_loss.png')
        plot_loss(test_loss_list, os.path.join(checkpoint_path, 'figs'), 'test_loss.png')
        plot_metrics_seq2seq(train_mae_list, os.path.join(checkpoint_path, 'figs'), 'train_mae.png', 'train_mae')
        plot_metrics_seq2seq(train_mse_list, os.path.join(checkpoint_path, 'figs'), 'train_mse.png', 'train_mse')
        plot_metrics_seq2seq(train_rmse_list, os.path.join(checkpoint_path, 'figs'), 'train_rmse.png', 'train_rmse')
        plot_metrics_seq2seq(train_mape_list, os.path.join(checkpoint_path, 'figs'), 'train_mape.png', 'train_mape')
        plot_metrics_seq2seq(train_mspe_list, os.path.join(checkpoint_path, 'figs'), 'train_mspe.png', 'train_mspe')
        plot_metrics_seq2seq(test_mae_list, os.path.join(checkpoint_path, 'figs'), 'test_mae.png', 'test_mae')
        plot_metrics_seq2seq(test_mse_list, os.path.join(checkpoint_path, 'figs'), 'test_mse.png', 'test_mse')
        plot_metrics_seq2seq(test_rmse_list, os.path.join(checkpoint_path, 'figs'), 'test_rmse.png', 'test_rmse')
        plot_metrics_seq2seq(test_mape_list, os.path.join(checkpoint_path, 'figs'), 'test_mape.png', 'test_mape')
        plot_metrics_seq2seq(test_mspe_list, os.path.join(checkpoint_path, 'figs'), 'test_mspe.png', 'test_mspe')

        save_data(train_loss_list, os.path.join(checkpoint_path, 'files'), 'train_loss.csv') 
        save_data(test_loss_list, os.path.join(checkpoint_path, 'files'), 'test_loss.csv') 
        save_data(train_mae_list, os.path.join(checkpoint_path, 'files'), 'train_mae.csv')
        save_data(train_mse_list, os.path.join(checkpoint_path, 'files'), 'train_mse.csv')
        save_data(train_rmse_list, os.path.join(checkpoint_path, 'files'), 'train_rmse.csv')
        save_data(train_mape_list, os.path.join(checkpoint_path, 'files'), 'train_mape.csv')
        save_data(train_mspe_list, os.path.join(checkpoint_path, 'files'), 'train_mspe.csv')
        save_data(test_mae_list, os.path.join(checkpoint_path, 'files'), 'test_mae.csv')
        save_data(test_mse_list, os.path.join(checkpoint_path, 'files'), 'test_mse.csv')
        save_data(test_rmse_list, os.path.join(checkpoint_path, 'files'), 'test_rmse.csv')
        save_data(test_mape_list, os.path.join(checkpoint_path, 'files'), 'test_mape.csv')
        save_data(test_mspe_list, os.path.join(checkpoint_path, 'files'), 'test_mspe.csv')

        
        input_x = torch.randn_like(batch_x)[0].unsqueeze(0).float().to(self.device)
        input_x_mark = torch.randn_like(batch_x_mark)[0].unsqueeze(0).float().to(self.device)
        input_y = torch.randn_like(batch_y)[0].unsqueeze(0).float().to(self.device)
        input_y_mark = torch.randn_like(batch_y_mark)[0].unsqueeze(0).float().to(self.device)
        flops, params = profile(self.model, inputs=(input_x, input_x_mark, input_y, input_y_mark))
        print("FLOPs: %.2fM" % (flops / 1e6), "Params: %.5fM" % (params / 1e6))

        if self.args.gpu is not None:
            save_seq2seq_gpu(os.path.join(checkpoint_path, 'files'), 'parameters.txt', self.args, len(train_data), len(test_data), params, flops)

        return self.model

    def test(self, checkpoint_path):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])


        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(checkpoint_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(checkpoint_path+'pred.npy', preds)
        np.save(checkpoint_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y
