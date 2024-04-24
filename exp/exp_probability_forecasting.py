import math
import os
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.pf_utils import init_metrics, update_metrics, final_metrics
from utils.tools import EarlyStopping, adjust_learning_rate, draw_figure, draw_attention_map

warnings.filterwarnings('ignore')


# noinspection DuplicatedCode
class Exp_Probability_Forecast(Exp_Basic):
    def __init__(self, args, try_model=False, save_process=True):
        super(Exp_Probability_Forecast, self).__init__(args, try_model, save_process)

    def train(self, setting, check_folder=False, only_init=False, adjust_lr=False):
        if check_folder:
            self._check_folders([self.args.checkpoints, self.root_process_path])

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and not self.try_model:
            os.makedirs(path)

        process_path = self.root_process_path + f'/{setting}/'
        if not os.path.exists(process_path) and not self.try_model:
            os.makedirs(process_path)
        self.process_file_path = process_path + f'{self.args.task_name}.txt'

        if only_init:
            return

        train_data, train_loader = self._get_data(data_flag='train', enter_flag='train', _try_model=self.try_model)
        vali_data, vali_loader = self._get_data(data_flag='val', enter_flag='train', _try_model=self.try_model)
        test_data, test_loader = self._get_data(data_flag='test', enter_flag='train', _try_model=self.try_model)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.checkpoints_file_path, patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        train_losses = []
        vali_losses = []
        test_losses = []

        stop_flag = False
        stop_epochs = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)  # [256, 96, 14]
                batch_y = batch_y.float().to(self.device)  # [256, 32, 14]
                batch_x_mark = batch_x_mark.float().to(self.device)  # [256, 96, 5]
                batch_y_mark = batch_y_mark.float().to(self.device)  # [256, 32, 5]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                if self.args.label_len != 0:
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()  # [256, 32, 14]

                # try model if needed
                if self.try_model:
                    # noinspection PyBroadException
                    try:
                        self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)
                        return True
                    except:
                        return False

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)

                if isinstance(outputs[1], bool):
                    stop_flag = outputs[1]
                    outputs = outputs[0]
                    if stop_flag:
                        break

                if isinstance(outputs, list):
                    loss = torch.zeros(1, device=self.device, requires_grad=True)  # [,]
                    for output in outputs:
                        if isinstance(output, tuple):
                            loss = loss + criterion(output)
                        else:
                            raise NotImplementedError('The output of the model should be list for the model with '
                                                      'custom loss function!')
                elif isinstance(outputs, tuple):
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = tuple([output[:, -self.args.pred_len:, f_dim:] for output in outputs])  # [256, 16, 1]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # [256, 16, 1]
                    loss = criterion(outputs, batch_y)
                else:
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
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
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

            # validate one epoch
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            _ = ("Epoch: {0}, Steps: {1} --- Train Loss: {2:.7f}; Vali Loss: {3:.7f}; Test Loss: {4:.7f};".
                 format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            self.print_content(_)

            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            test_losses.append(test_loss)

            if stop_flag:
                self.print_content("Raise error and stop")
                stop_epochs = epoch + 1
                break

            _ = early_stopping(vali_loss, self.model, path)
            if _ is not None:
                self.print_content(_)

            if early_stopping.early_stop:
                self.print_content("Early stopping")
                stop_epochs = epoch + 1
                break

            if adjust_lr:
                _ = adjust_learning_rate(model_optim, epoch + 1, self.args)
                if _ is not None:
                    self.print_content(_)
            else:
                lr = model_optim.param_groups[0]['lr']
                self.print_content(f'learning rate is: {lr}')

        # save train, vali, test loss to process path
        np.save(process_path + 'train_loss.npy', np.array(train_losses))
        np.save(process_path + 'vali_loss.npy', np.array(vali_losses))
        np.save(process_path + 'test_loss.npy', np.array(test_losses))

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
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                if self.args.label_len != 0:
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark)

                if isinstance(outputs[1], bool):
                    outputs = outputs[0]

                if isinstance(outputs, list):
                    loss = torch.zeros(1, device=self.device, requires_grad=False)  # [,]
                    for output in outputs:  # [32, 1], [32, 20], [32]
                        if isinstance(output, tuple):
                            loss = loss + criterion(output)
                        else:
                            raise NotImplementedError('The output of the model should be list for a model with custom '
                                                      'loss function!')

                    loss = loss.detach().cpu()
                elif isinstance(outputs, tuple):
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = tuple([output[:, -self.args.pred_len:, f_dim:] for output in outputs])  # [256, 16, 1]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # [256, 16, 1]

                    outputs = tuple([output.detach().cpu() for output in outputs])
                    batch_y = batch_y.detach().cpu()

                    loss = criterion(outputs, batch_y)
                else:
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
        test_data, test_loader = self._get_data(data_flag='test', enter_flag='test', _try_model=self.try_model)
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
            self._check_folders([self.root_test_results_path, self.root_results_path, self.root_prob_results_path])

        preds = []
        trues = []

        folder_path = self.root_test_results_path + f'/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        probability_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        probability_range_len = len(probability_range)

        pred_length = self.args.pred_len
        batch_size = self.args.batch_size
        data_length = len(test_data)
        loader_length = len(test_loader)
        pred_value = torch.zeros(pred_length, data_length).to(self.device)
        true_value = torch.zeros(pred_length, data_length).to(self.device)
        high_value = torch.zeros(pred_length, probability_range_len, data_length).to(self.device)
        low_value = torch.zeros(pred_length, probability_range_len, data_length).to(self.device)

        attention_maps = (torch.zeros(loader_length, pred_length, batch_size, self.args.n_heads, 1, self.args.seq_len)
                          .to(self.device))

        self.model.eval()
        with (torch.no_grad()):
            metrics = init_metrics(self.args.pred_len, self.device)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)  # [256, 96, 17]
                batch_y = batch_y.float().to(self.device)  # [256, 16, 17]

                batch_x_mark = batch_x_mark.float().to(self.device)  # [256, 96, 5]
                batch_y_mark = batch_y_mark.float().to(self.device)  # [256, 16, 5]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)  # [256, 16, 17]
                if self.args.label_len != 0:
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model.predict(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark,
                                                         probability_range=probability_range)[0]
                        else:
                            outputs = self.model.predict(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark,
                                                         probability_range=probability_range)
                else:
                    if self.args.output_attention:
                        outputs = self.model.predict(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark,
                                                     probability_range=probability_range)[0]
                    else:
                        outputs = self.model.predict(batch_x, batch_x_mark, dec_inp, batch_y, batch_y_mark,
                                                     probability_range=probability_range)

                samples, sample_mu, sample_std, samples_high, samples_low, attention_map = outputs
                # [99, 256, 12], [256, 12, 1], [256, 12, 1], [3, 256, 16], [3, 256, 16], [16, 256, 8, 1, 96]

                pred = sample_mu[:, :, -1].transpose(0, 1).squeeze()
                pred_value[:, i * batch_size: (i + 1) * batch_size] = pred
                high = samples_high.transpose(0, 2).transpose(1, 2)  # [16, 3, 256]
                high_value[:, :, i * batch_size: (i + 1) * batch_size] = high
                low = samples_low.transpose(0, 2).transpose(1, 2)  # [16, 3, 256]
                low_value[:, :, i * batch_size: (i + 1) * batch_size] = low

                if self.args.label_len == 0:
                    batch = torch.cat((batch_x, batch_y), dim=1).float()  # [256, 112, 17]
                else:
                    batch = torch.cat((batch_x, batch_y[:, :self.args.label_len, :]), dim=1)

                labels = batch[:, :, -1]  # [256, 112, 1]
                metrics = update_metrics(metrics, samples, labels, self.args.seq_len)
                labels = labels.unsqueeze(-1)  # [256, 112, 1]

                true = batch_y[:, :, -1].transpose(0, 1).squeeze()
                true_value[:, i * batch_size: (i + 1) * batch_size] = true

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = sample_mu[:, -self.args.pred_len:, :]  # [256, 16, 1]
                batch_y = labels[:, -self.args.pred_len:, :]  # [256, 16, 1]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]  # [256, 16, 1]
                batch_y = batch_y[:, :, f_dim:]  # [256, 16, 1]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                attention_maps[i] = attention_map

            summary = final_metrics(metrics)

        preds = np.array(preds)
        trues = np.array(trues)
        self.print_content(f'test shape: {preds.shape} {trues.shape}')  # (22, 256, 16, 1) (22, 256, 16, 1)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        self.print_content(f'test shape: {preds.shape} {trues.shape}')  # (5632, 16, 1) (5632, 16, 1)

        # result save
        folder_path = self.root_results_path + f'/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.print_content('mse:{}, mae:{}'.format(mse, mae))

        strings = '\nCRPS: ' + str(summary['CRPS']) + \
                  '\nmre:' + str(summary['mre'].abs().max(dim=1)[0].mean().item()) + \
                  '\nPINAW:' + str(summary['pinaw'].item())
        self.print_content('Full test metrics: ' + strings)

        ss_metric = {'CRPS_Mean': summary['CRPS'].mean(), 'mre': summary['mre'].abs().mean(), 'pinaw': summary['pinaw']}
        for i, crps in enumerate(summary['CRPS']):
            ss_metric[f'CRPS_{i}'] = crps
        for i, mre in enumerate(summary['mre'].mean(dim=0)):
            ss_metric[f'mre_{i}'] = mre

        # save results in txt
        # f = open("result_probability_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        self.print_content("", True)

        if test:
            folder_path = self.root_prob_results_path + f'/{setting}/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # pred value, true value & probability range
            # move to cpu and covert to numpy for plotting
            pred_value = pred_value.detach().cpu().numpy()  # [16, 15616]
            true_value = true_value.detach().cpu().numpy()  # [16, 15616]
            high_value = high_value.detach().cpu().numpy()  # [16, 3, 15616]
            low_value = low_value.detach().cpu().numpy()  # [16, 3, 15616]

            # save results in npy
            np.save(folder_path + 'pred_value.npy', pred_value)
            np.save(folder_path + 'true_value.npy', true_value)
            np.save(folder_path + 'high_value.npy', high_value)
            np.save(folder_path + 'low_value.npy', low_value)

            # integrate different probability range data
            pred_value = pred_value.reshape(-1)  # [16 * 15616]
            true_value = true_value.reshape(-1)  # [16 * 15616]
            high_value = high_value.reshape(-1)  # [16 * 46848]
            low_value = low_value.reshape(-1)  # [16 * 46848]

            # convert to shape: (sample, feature) for inverse transform
            new_shape = (pred_length * data_length, self.args.enc_in)
            _ = np.zeros(new_shape)
            _[:, -1] = pred_value
            pred_value = _
            _ = np.zeros(new_shape)
            _[:, -1] = true_value
            true_value = _
            new_shape = (pred_length * probability_range_len * data_length, self.args.enc_in)
            _ = np.zeros(new_shape)
            _[:, -1] = high_value
            high_value = _
            _ = np.zeros(new_shape)
            _[:, -1] = low_value
            low_value = _

            # perform inverse transform
            dataset = test_data
            pred_value = dataset.inverse_transform(pred_value)
            true_value = dataset.inverse_transform(true_value)
            high_value = dataset.inverse_transform(high_value)
            low_value = dataset.inverse_transform(low_value)

            # get the original data
            pred_value = pred_value[:, -1].squeeze()  # [16 * 15616]
            true_value = true_value[:, -1].squeeze()  # [16 * 15616]
            high_value = high_value[:, -1].squeeze()  # [16 * 46848]
            low_value = low_value[:, -1].squeeze()  # [16 * 46848]

            # restore different probability range data
            pred_value = pred_value.reshape(pred_length, data_length)  # [16, 15616]
            true_value = true_value.reshape(pred_length, data_length)  # [16, 15616]
            high_value = high_value.reshape(pred_length, probability_range_len, data_length)  # [16, 3, 15616]
            low_value = low_value.reshape(pred_length, probability_range_len, data_length)  # [16, 3, 15616]

            # draw figures
            print('drawing probabilistic figure')
            for i in tqdm(range(pred_length)):
                _path = os.path.join(folder_path, f'probabilistic_figure', f'step {i}')
                if not os.path.exists(_path):
                    os.makedirs(_path)

                interval = 128
                num = math.floor(data_length / interval)
                for j in range(num):
                    if j * interval >= data_length:
                        continue
                    draw_figure(range(interval),
                                pred_value[i, j * interval: (j + 1) * interval],
                                true_value[i, j * interval: (j + 1) * interval],
                                high_value[i, :, j * interval: (j + 1) * interval],
                                low_value[i, :, j * interval: (j + 1) * interval],
                                probability_range,
                                os.path.join(_path, f'prediction {j}.png'))

            # attention map
            # move to cpu and covert to numpy for plotting
            attention_maps = attention_maps.detach().cpu().numpy()  # [61, 16, 256, 8, 1, 96]

            # save results in npy
            np.save(folder_path + 'attention_maps.npy', attention_maps)

            # draw attention map
            print('drawing attention map')
            # for i in tqdm(range(loader_length)):
            #     _path = os.path.join(folder_path, f'attention_map', f'loader {i}')
            #     if not os.path.exists(_path):
            #         os.makedirs(_path)
            #
            #     attention_map = attention_maps[i]
            #     attention_map = attention_map.reshape(batch_size, self.args.n_heads, 1 * pred_length,
            #                                           self.args.seq_len)
            #     for j in range(batch_size):
            #         _ = attention_map[j]
            #         draw_attention_map(attention_map[j], os.path.join(_path, f'attention map {j}.png'))

            for i in tqdm(range(pred_length)):
                _path = os.path.join(folder_path, f'attention_map', f'step {i}')
                if not os.path.exists(_path):
                    os.makedirs(_path)

                attention_map = attention_maps[:, i, :, :, :, :]  # [61, 256, 8, 1, 96]
                attention_map = attention_map.reshape(loader_length * batch_size, self.args.n_heads, 1,
                                                      self.args.seq_len)
                # [15616, 8, 1, 96]

                interval = 96
                num = math.floor(data_length / interval)
                for j in range(num):
                    if j * interval >= data_length:
                        continue

                    _attention_map = attention_map[j * interval: (j + 1) * interval]  # [96, 8, 1, 96]
                    _attention_map = _attention_map.reshape(self.args.n_heads, 1 * interval, self.args.seq_len)
                    # [8, 96, 96]
                    draw_attention_map(_attention_map, os.path.join(_path, f'attention map {j}.png'))

        # draw demo data for overall structure
        # draw_demo(0, 19, pred_value, true_value, high_value, low_value, folder_path, probability_range)

        # convert to float
        crps = float(ss_metric['CRPS_Mean'].item())
        mre = float(ss_metric['mre'].item())
        pinaw = float(ss_metric['pinaw'].item())

        return {
            'mse': mse,
            'mae': mae,
            'crps': crps,
            'mre': mre,
            'pinaw': pinaw
        }


def draw_demo(i, j, pred, true, high, low, folder_path, probability_range):
    from matplotlib import pyplot as plt

    interval = 128
    pred_data_prop = 1 / 3
    demo_pred_len = int(interval * pred_data_prop)
    demo_true_data = true[i, j * interval: (j + 1) * interval - demo_pred_len]
    demo_true_data_1 = true[i, j * interval + interval - demo_pred_len: (j + 1) * interval]
    demo_pred_data = pred[i, j * interval + interval - demo_pred_len: (j + 1) * interval]
    demo_high_data = high[i, :, j * interval + interval - demo_pred_len: (j + 1) * interval]
    demo_low_data = low[i, :, j * interval + interval - demo_pred_len: (j + 1) * interval]
    plt.clf()
    plt.plot(demo_true_data.squeeze(), color='black')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f'overall structure 0.png'))
    plt.clf()
    plt.plot(demo_true_data_1.squeeze(), color='gray')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f'overall structure 1.png'))
    plt.clf()
    plt.plot(demo_pred_data.squeeze(), color='black')
    for j in range(len(probability_range)):
        plt.fill_between(range(demo_pred_len), demo_high_data[j, :].squeeze(), demo_low_data[j, :].squeeze(),
                         color='gray',
                         alpha=1 - probability_range[j])
    plt.legend()
    plt.savefig(os.path.join(folder_path, f'overall structure 2.png'))
