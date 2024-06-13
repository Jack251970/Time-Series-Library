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
from utils.tools import EarlyStopping, adjust_learning_rate, draw_figure, draw_attention_map, draw_density_figure

warnings.filterwarnings('ignore')


# noinspection DuplicatedCode
class Exp_Probability_Forecast(Exp_Basic):
    def __init__(self, root_path, args, try_model=False, save_process=True):
        super(Exp_Probability_Forecast, self).__init__(root_path, args, try_model, save_process)

    def train(self, setting, check_folder=False, only_init=False, adjust_lr=False):
        if check_folder:
            self._check_folders([self.root_checkpoints_path, self.root_process_path])

        checkpoints_path = os.path.join(self.root_checkpoints_path, setting)

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
        early_stopping = EarlyStopping(self.checkpoints_file, patience=self.args.patience, verbose=True)

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

                    # train don't need to select the length with pred_len
                    outputs = tuple([output[:, :, f_dim:] for output in outputs])  # [256, 16, 1]
                    batch_y = batch_y[:, :, f_dim:].to(self.device)  # [256, 16, 1]

                    loss = criterion(outputs, batch_y)
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:].to(self.device)
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

            _ = early_stopping(vali_loss, self.model, checkpoints_path)
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

        best_model_path = checkpoints_path + '/' + self.checkpoints_file
        if os.path.exists(best_model_path):
            if self.device == torch.device('cpu'):
                self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
            else:
                self.model.load_state_dict(torch.load(best_model_path))

        if stop_epochs == 0:
            stop_epochs = self.args.train_epochs
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

                    # validation need to select the length with pred_len
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

    def test(self, setting, test=False, check_folder=False,
             probabilistic_flag=True, sample_flag=True, attention_flag=True, parameter_flag=True):
        test_data, test_loader = self._get_data(data_flag='test', enter_flag='test', _try_model=self.try_model)
        if test:
            self.print_content('loading model')
            path = os.path.join(self.root_checkpoints_path, setting)
            best_model_path = path + '/' + self.checkpoints_file
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

        seq_length = self.args.seq_len
        pred_length = self.args.pred_len
        batch_size = self.args.batch_size
        data_length = len(test_data)
        loader_length = len(test_loader)

        if not test:
            probabilistic_flag = False
            sample_flag = False
            attention_flag = False
            parameter_flag = False

        # probabilistic forecast
        if probabilistic_flag:
            pred_value = torch.zeros(pred_length, data_length).to(self.device)
            true_value = torch.zeros(pred_length, data_length).to(self.device)
            high_value = torch.zeros(pred_length, probability_range_len, data_length).to(self.device)
            low_value = torch.zeros(pred_length, probability_range_len, data_length).to(self.device)
        else:
            pred_value = None
            true_value = None
            high_value = None
            low_value = None

        # sample value on certain time steps
        if sample_flag:
            samples_index = [15, 31, 63, 95]
            samples_number = len(samples_index)
            samples_value = torch.zeros(self.args.sample_times, samples_number, data_length).to(self.device)
        else:
            samples_index = None
            samples_number = None
            samples_value = None

        # attention map
        if attention_flag:
            attention_maps = (torch.zeros(loader_length, pred_length, batch_size, self.args.n_heads, 1, seq_length)
                              .to(self.device))
        else:
            attention_maps = None

        # parameters
        if parameter_flag:
            if self.args.model == 'QSQF-C':
                parameter_flag = True
                samples_lambda = torch.zeros(pred_length, data_length, 1).to(self.device)
                samples_gamma = torch.zeros(pred_length, data_length, 1).to(self.device)
                samples_eta_k = torch.zeros(pred_length, data_length, self.args.num_spline).to(self.device)
            elif self.args.model == 'LSTM-AQ':
                parameter_flag = True
                samples_lambda = torch.zeros(pred_length, data_length, 1).to(self.device)
                samples_gamma = torch.zeros(pred_length, data_length, self.args.num_spline).to(self.device)
                samples_eta_k = torch.zeros(pred_length, data_length, self.args.num_spline).to(self.device)
            else:
                parameter_flag = False
                samples_lambda = None
                samples_gamma = None
                samples_eta_k = None
        else:
            samples_lambda = None
            samples_gamma = None
            samples_eta_k = None

        self.model.eval()
        with (torch.no_grad()):
            metrics = init_metrics(pred_length, self.device)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)  # [256, 96, 17]
                batch_y = batch_y.float().to(self.device)  # [256, 16, 17]

                batch_x_mark = batch_x_mark.float().to(self.device)  # [256, 96, 5]
                batch_y_mark = batch_y_mark.float().to(self.device)  # [256, 16, 5]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_length:, :]).float().to(self.device)  # [256, 16, 17]
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

                samples, sample_mu, sample_std, samples_high, samples_low, attention_map, parameters = outputs
                # [99, 256, 12], [256, 12, 1], [256, 12, 1], [3, 256, 16], [3, 256, 16], [16, 256, 8, 1, 96], ()

                # test need to select the length with pred_len
                samples = samples[:, :, -pred_length:]
                sample_mu = sample_mu[:, -pred_length:, :]
                # sample_std = sample_std[:, -pred_length:, :]
                samples_high = samples_high[:, :, -pred_length:]
                samples_low = samples_low[:, :, -pred_length:]

                if attention_map is not None and attention_flag:
                    attention_map = attention_map[-pred_length:, :, :, :, :]
                else:
                    attention_flag = False
                if parameters is not None and parameter_flag:
                    if isinstance(parameters, tuple):
                        parameters = tuple([parameter[-pred_length:, :, :] for parameter in parameters])
                else:
                    parameter_flag = False

                if sample_flag:
                    pred_samples = samples.transpose(1, 2)  # [99, 16, 256]
                    pred_samples = pred_samples[:, samples_index, :]  # [99, 4, 256]
                    samples_value[:, :, i * batch_size: (i + 1) * batch_size] = pred_samples
                if probabilistic_flag:
                    pred = sample_mu[:, :, -1].transpose(0, 1)
                    pred_value[:, i * batch_size: (i + 1) * batch_size] = pred
                    high = samples_high.transpose(0, 2).transpose(1, 2)  # [16, 3, 256]
                    high_value[:, :, i * batch_size: (i + 1) * batch_size] = high
                    low = samples_low.transpose(0, 2).transpose(1, 2)  # [16, 3, 256]
                    low_value[:, :, i * batch_size: (i + 1) * batch_size] = low
                    true = batch_y[:, -pred_length:, -1].transpose(0, 1)
                    true_value[:, i * batch_size: (i + 1) * batch_size] = true

                if self.args.label_len == 0:
                    batch = torch.cat((batch_x, batch_y), dim=1).float()  # [256, 112, 17]
                else:
                    batch = torch.cat((batch_x, batch_y[:, -pred_length:, :]), dim=1)

                labels = batch[:, -pred_length:, -1]  # [256, 96, 1]
                metrics = update_metrics(metrics, samples, labels, pred_length)
                labels = labels.unsqueeze(-1)  # [256, 96, 1]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = sample_mu  # [256, 16, 1]
                batch_y = labels  # [256, 16, 1]

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

                if attention_flag:
                    attention_maps[i] = attention_map

                if parameter_flag:
                    samples_lambda[:, i * batch_size: (i + 1) * batch_size, :] = parameters[0]
                    samples_gamma[:, i * batch_size: (i + 1) * batch_size, :] = parameters[1]
                    samples_eta_k[:, i * batch_size: (i + 1) * batch_size, :] = parameters[2]

            summary = final_metrics(metrics, pred_length)

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

        strings = '\nCRPS: ' + str(summary['CRPS'])
        for i in range(pred_length):
            strings += '\nCRPS_' + str(i) + ': ' + str(summary[f'CRPS_{i}'])
        strings += '\nmre:' + str(summary['mre'].abs().max(dim=1)[0].mean().item())
        strings += '\nPINAW:' + str(summary['pinaw'].item())
        for i in range(pred_length):
            strings += '\nPINAW_' + str(i) + ': ' + str(summary[f'pinaw_{i}'].item())
        self.print_content('Full test metrics: ' + strings)

        ss_metric = {
            'CRPS': summary['CRPS'].mean().detach().cpu(),
            'mre': summary['mre'].abs().mean().detach().cpu(),
            'pinaw': summary['pinaw'].detach().cpu()
        }
        for i in range(pred_length):
            ss_metric[f'CRPS_{i}'] = summary[f'CRPS_{i}'].mean().detach().cpu()
            ss_metric[f'pinaw_{i}'] = summary[f'pinaw_{i}'].detach().cpu()

        prob_metrics = np.array([ss_metric['CRPS']])
        for i in range(pred_length):
            prob_metrics = np.append(prob_metrics, ss_metric[f'CRPS_{i}'])
        prob_metrics = np.append(prob_metrics, ss_metric['mre'])
        prob_metrics = np.append(prob_metrics, ss_metric['pinaw'])
        for i in range(pred_length):
            prob_metrics = np.append(prob_metrics, ss_metric[f'pinaw_{i}'])
        np.save(folder_path + 'prob_metrics.npy', prob_metrics)

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
            if probabilistic_flag:
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

                # save results in npy
                np.save(folder_path + 'pred_value_inverse.npy', pred_value)
                np.save(folder_path + 'true_value_inverse.npy', true_value)
                np.save(folder_path + 'high_value_inverse.npy', high_value)
                np.save(folder_path + 'low_value_inverse.npy', low_value)

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

            if sample_flag:
                # move to cpu and covert to numpy for plotting
                samples_value = samples_value.detach().cpu().numpy()  # [99, 5, 15616]

                # integrate different probability range data
                samples_value = samples_value.reshape(-1)  # [99 * 5 * 15616]

                # convert to shape: (sample, feature) for inverse transform
                new_shape = (self.args.sample_times * samples_number * data_length, self.args.enc_in)
                _ = np.zeros(new_shape)
                _[:, -1] = samples_value
                samples_value = _

                # perform inverse transform
                dataset = test_data
                samples_value = dataset.inverse_transform(samples_value)

                # get the original data
                samples_value = samples_value[:, -1].squeeze()  # [99 * 5 * 15616]

                # restore different probability range data
                samples_value = samples_value.reshape(self.args.sample_times, samples_number,
                                                      data_length)  # [99, 5, 15616]

                # draw figures
                print('drawing probabilistic density figure')
                for i in range(samples_number):
                    _path = os.path.join(folder_path, f'probability_density', f'step {samples_index[i]}')
                    if not os.path.exists(_path):
                        os.makedirs(_path)

                    for j in tqdm(range(data_length), desc=f'step {samples_index[i]}'):
                        draw_density_figure(samples_value[:, i, j], true_value[i, j],
                                            os.path.join(_path, f'prediction {j}.png'))

            # attention map
            if attention_flag:
                # move to cpu and covert to numpy for plotting
                attention_maps = attention_maps.detach().cpu().numpy()  # [61, 16, 256, 8, 1, 96]

                # save results in npy
                np.save(folder_path + 'attention_maps.npy', attention_maps)

                # draw figure
                print('drawing attention map')
                for i in tqdm(range(loader_length)):
                    _path = os.path.join(folder_path, f'attention_map', f'loader {i}')
                    if not os.path.exists(_path):
                        os.makedirs(_path)

                    attention_map = attention_maps[i]
                    attention_map = attention_map.reshape(batch_size, self.args.n_heads, 1 * pred_length, seq_length)
                    for j in range(batch_size):
                        _ = attention_map[j]
                        draw_attention_map(attention_map[j], os.path.join(_path, f'attention map {j}.png'))

                for i in tqdm(range(pred_length)):
                    _path = os.path.join(folder_path, f'attention_map', f'step {i}')
                    if not os.path.exists(_path):
                        os.makedirs(_path)

                    attention_map = attention_maps[:, i, :, :, :, :]  # [61, 256, 8, 1, 96]
                    attention_map = attention_map.reshape(loader_length * batch_size, self.args.n_heads, 1, seq_length)
                    # [15616, 8, 1, 96]

                    interval = 96
                    num = math.floor(loader_length * batch_size / interval)
                    for j in range(num):
                        if j * interval >= data_length:
                            continue

                        _attention_map = attention_map[j * interval: (j + 1) * interval]  # [96, 8, 1, 96]
                        _attention_map = _attention_map.reshape(self.args.n_heads, 1 * interval, seq_length)
                        # [8, 96, 96]
                        draw_attention_map(_attention_map, os.path.join(_path, f'attention map {j}.png'))

            if parameter_flag:
                # move to cpu and covert to numpy for plotting
                samples_lambda = samples_lambda.detach().cpu().numpy()
                samples_gamma = samples_gamma.detach().cpu().numpy()
                samples_eta_k = samples_eta_k.detach().cpu().numpy()

                # save results in npy
                np.save(folder_path + 'samples_lambda.npy', samples_lambda)
                np.save(folder_path + 'samples_gamma.npy', samples_gamma)
                np.save(folder_path + 'samples_eta_k.npy', samples_eta_k)

        # draw demo data for overall structure
        # draw_demo(0, 19, pred_value, true_value, high_value, low_value, folder_path, probability_range)

        # convert to float
        crps = float(ss_metric['CRPS'].item())
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
