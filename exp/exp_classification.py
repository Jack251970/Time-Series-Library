import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy

warnings.filterwarnings('ignore')


# noinspection DuplicatedCode
class Exp_Classification(Exp_Basic):
    def __init__(self, args, try_model=False, save_process=True):
        super(Exp_Classification, self).__init__(args, try_model, save_process)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN', _try_model=self.try_model)
        test_data, test_loader = self._get_data(flag='TEST', _try_model=self.try_model)
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        return super()._build_model()

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

        if only_init:
            return

        train_data, train_loader = self._get_data(flag='TRAIN', _try_model=self.try_model)
        vali_data, vali_loader = self._get_data(flag='TEST', _try_model=self.try_model)
        test_data, test_loader = self._get_data(flag='TEST', _try_model=self.try_model)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.checkpoints_file_path, patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        stop_epochs = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # try model if needed
                if self.try_model:
                    # noinspection PyBroadException
                    try:
                        self.model(batch_x, padding_mask, None, None)
                        return True
                    except:
                        return False

                outputs = self.model(batch_x, padding_mask, None, None)

                loss = criterion(outputs, label.long().squeeze(-1))
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

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
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
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            _ = (("Epoch: {0}, Steps: {1} --- Train Loss: {2:.3f}; Vali Loss: {3:.3f}; Vali Acc: {4:.3f}; "
                 "Test Loss: {5:.3f}; Test Acc: {6:.3f};")
                 .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            self.print_content(_)

            _ = early_stopping(-val_accuracy, self.model, path)
            if _ is not None:
                self.print_content(_)

            if early_stopping.early_stop:
                self.print_content("Early stopping")
                stop_epochs = epoch + 1
                break

            if (epoch + 1) % 5 == 0:
                _ = adjust_learning_rate(model_optim, epoch + 1, self.args)
                if _ is not None:
                    self.print_content(_)

        best_model_path = path + '/' + self.checkpoints_file_path
        if os.path.exists(best_model_path):
            if self.device == torch.device('cpu'):
                self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
            else:
                self.model.load_state_dict(torch.load(best_model_path))

        self.print_content("", True)

        return stop_epochs

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss_values = [tensor_item.item() for tensor_item in total_loss]
        total_loss = np.average(total_loss_values)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def test(self, setting, test=False, check_folder=False):
        test_data, test_loader = self._get_data(flag='TEST', _try_model=self.try_model)
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

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        self.print_content(f'test shape: {preds.shape}, {trues.shape}')

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = self.root_results_path + f'/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.print_content('accuracy:{}'.format(accuracy))

        # save results in txt
        # file_name='result_classification.txt'
        # f = open(os.path.join(folder_path,file_name), 'a')
        # f.write(setting + "  \n")
        # f.write('accuracy:{}'.format(accuracy))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        self.print_content("", True)

        return {
            'accuracy': accuracy,
        }
