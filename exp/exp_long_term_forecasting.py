import math
import os
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, draw_figure

warnings.filterwarnings('ignore')


# noinspection DuplicatedCode
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args, try_model=False, save_process=True):
        super(Exp_Long_Term_Forecast, self).__init__(args, try_model, save_process)

    def train(self, setting, check_folder=False, only_init=False):
        if check_folder:
            self._check_folders([self.args.checkpoints, self.root_process_path])

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and not self.try_model:
            os.makedirs(path)
        
        process_path = self.root_process_path + f'/{setting}/'
        if not os.path.exists(process_path) and not self.try_model:
            os.makedirs(process_path)
        self.process_file_path = process_path + f'{self.args.task_name}.txt'

        train_data, train_loader = self._get_data(flag='train', _try_model=self.try_model)
        vali_data, vali_loader = self._get_data(flag='val', _try_model=self.try_model)
        test_data, test_loader = self._get_data(flag='test', _try_model=self.try_model)

        if only_init:
            return

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.checkpoints_file_path, patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        stop_epochs = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # try model if needed
                if self.try_model:
                    # noinspection PyBroadException
                    try:
                        self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        return True
                    except:
                        return False

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    _ = "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    self.print_content(_)
                    speed = (time.time() - time_now) / iter_count
                    # left time for all epochs
                    # left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # left time for current epoch
                    left_time = speed * (train_steps - i)
                    if left_time > 60 * 60:
                        _ = '\tspeed: {:.4f} s/iter; left time: {:.4f} hour'.format(speed, left_time / 60.0 / 60.0)
                    elif left_time > 60:
                        _ = '\tspeed: {:.4f} s/iter; left time: {:.4f} min'.format(speed, left_time / 60.0)
                    else:
                        _ = '\tspeed: {:.4f} s/iter; left time: {:.4f} second'.format(speed, left_time)
                    self.print_content(_)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            current_epoch_time = time.time() - epoch_time
            if current_epoch_time > 60 * 60:
                _ = "Epoch: {}; cost time: {:.4f} hour".format(epoch + 1, current_epoch_time / 60.0 / 60.0)
            elif current_epoch_time > 60:
                _ = "Epoch: {}; cost time: {:.4f} min".format(epoch + 1, current_epoch_time / 60.0)
            else:
                _ = "Epoch: {}; cost time: {:.4f} second".format(epoch + 1, current_epoch_time)
            self.print_content(_)

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            _ = ("Epoch: {0}, Steps: {1} --- Train Loss: {2:.7f}; Vali Loss: {3:.7f}; Test Loss: {4:.7f};".
                 format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            self.print_content(_)

            _ = early_stopping(vali_loss, self.model, path)
            if _ is not None:
                self.print_content(_)

            if early_stopping.early_stop:
                self.print_content("Early stopping")
                stop_epochs = epoch + 1
                break

            _ = adjust_learning_rate(model_optim, epoch + 1, self.args)
            if _ is not None:
                self.print_content(_)

        self.print_content("", True)

        best_model_path = path + '/' + self.checkpoints_file_path
        if os.path.exists(best_model_path):
            if self.device == torch.device('cpu'):
                self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
            else:
                self.model.load_state_dict(torch.load(best_model_path))

        return stop_epochs

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
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
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss_values = [tensor_item.item() for tensor_item in total_loss]
        total_loss = np.average(total_loss_values)
        self.model.train()
        return total_loss

    def test(self, setting, test=False, check_folder=False):
        test_data, test_loader = self._get_data(flag='test', _try_model=self.try_model)
        if test:
            self.print_content('loading model')
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + self.checkpoints_file_path
            if os.path.exists(best_model_path):
                if self.device == torch.device('cpu'):
                    self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
                else:
                    self.model.load_state_dict(torch.load(best_model_path))
            else:
                raise FileNotFoundError('You need to train this model before testing it!')

        if check_folder:
            self._check_folders([self.root_test_results_path, self.root_results_path])

        preds = []
        trues = []

        folder_path = self.root_test_results_path + f'/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        probability_step = 0
        select_position = -test_data.pred_len + probability_step

        length = len(test_data)
        batch_size = test_loader.batch_size
        pred_value = torch.zeros(length).to(self.device)
        true_value = torch.zeros(length).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred_value[i * batch_size: (i + 1) * batch_size] = outputs[:, select_position, -1].squeeze()
                true_value[i * batch_size: (i + 1) * batch_size] = batch_y[:, select_position, -1].squeeze()

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs  # [256, 96, 1]
                true = batch_y  # [256, 96, 1]

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    _input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = _input.shape
                        _input = test_data.inverse_transform(_input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((_input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((_input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        self.print_content(f'test shape: {preds.shape} {trues.shape}')
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])  # (5684, 96, 14)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])  # (5684, 96, 14)
        self.print_content(f'test shape: {preds.shape} {trues.shape}')

        # result save
        folder_path = self.root_results_path + f'/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.print_content('mse:{}, mae:{}'.format(mse, mae))

        # save results in txt
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        self.print_content("", True)

        folder_path = self.root_prob_results_path + f'/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # move to cpu and covert to numpy for plotting
        pred_value = pred_value.detach().cpu().numpy()  # [15616]
        true_value = true_value.detach().cpu().numpy()  # [15616]

        # convert to shape: (sample, feature) for inverse transform
        new_shape = (length, self.args.enc_in)
        _ = np.zeros(new_shape)
        _[:, -1] = pred_value
        pred_value = _
        _ = np.zeros(new_shape)
        _[:, -1] = true_value
        true_value = _

        # perform inverse transform
        dataset = test_data
        pred_value = dataset.inverse_transform(pred_value)
        true_value = dataset.inverse_transform(true_value)

        # get the original data
        pred_value = pred_value[:, -1].squeeze()  # [15616]
        true_value = true_value[:, -1].squeeze()  # [15616]

        # draw figures
        draw_figure(range(length), pred_value, true_value, None, None, None,
                    os.path.join(folder_path, 'prediction all.png'))

        interval = 128
        num = math.floor(length / interval)
        for i in range(num):
            if (i + 1) * interval >= length:
                continue
            draw_figure(range(interval), pred_value[i * interval: (i + 1) * interval],
                        true_value[i * interval: (i + 1) * interval],
                        None, None, None, os.path.join(folder_path, f'prediction {i}.png'))

        return {
            'mse': mse,
            'mae': mae,
        }
