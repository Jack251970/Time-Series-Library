import os
import torch
from torch import optim, nn

from data_provider.data_factory import data_provider
from models import (Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, Informer, LightTS,
                    Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, Koopa, TiDE,
                    FreTS)
from models.quantile_function import (rnn_sf, lstm_cq, lstm_aq, lstm_aq1, lstm_aq2, lstm_aq3, lstm_aq4, lstm_yjqr,
                                      lstm_ed_yjqr, qsqf_c, qsqf_c1)
from utils.losses import mape_loss, mase_loss, smape_loss


class Exp_Basic(object):
    def __init__(self, root_path, args, try_model, save_process, initialize_later=False):
        self.args = args
        self.try_model = try_model
        self.save_process = save_process
        self.process_content = ""
        self.process_file_path = None
        if not initialize_later:
            self.device = self._acquire_device(try_model)
            self.model = self._build_model().to(self.device)
        self.new_index = None

        # folder paths
        self.root_checkpoints_path = os.path.join(root_path, 'checkpoints')
        self.root_process_path = os.path.join(root_path, 'process')
        self.root_results_path = os.path.join(root_path, 'results')
        self.root_test_results_path = os.path.join(root_path, 'test_results')
        self.root_m4_results_path = os.path.join(root_path, 'm4_results')
        self.root_prob_results_path = os.path.join(root_path, 'prob_results')

        # file paths
        self.checkpoints_file = 'checkpoint.pth'

    def _build_model(self):
        # get model from model dictionary
        model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'QSQF-C': qsqf_c,
            'RNN-SF': rnn_sf,
            'LSTM-CQ': lstm_cq,
            'LSTM-AQ': lstm_aq,
            'LSTM-YJQR': lstm_yjqr,
            'LSTM-ED-YJQR': lstm_ed_yjqr,
            'LSTM-AQ1': lstm_aq1,
            'LSTM-AQ2': lstm_aq2,
            'LSTM-AQ3': lstm_aq3,
            'LSTM-AQ4': lstm_aq4,
            'QSQF-C1': qsqf_c1
        }
        model = model_dict[self.args.model].Model(self.args).float()
        # use multi gpus if enabled
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self, try_model):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
            if not try_model:
                if self.save_process:
                    self.print_content('Use GPU: cuda:{}'.format(self.args.gpu))
                    if torch.backends.cudnn.benchmark:
                        self.print_content('Use cudnn.benchmark')
        else:
            device = torch.device('cpu')
            if not try_model:
                if self.save_process:
                    self.print_content('Use CPU')
        return device

    def _get_data(self, data_flag, enter_flag, _try_model):
        data_set, data_loader, info, new_index = data_provider(self.args, data_flag, enter_flag,
                                                               new_indexes=self.new_index, cache_data=True)
        if new_index is not None and self.new_index is None:
            self.new_index = new_index
            try:
                self.model.new_index = new_index
                if not _try_model:
                    self.print_content('New index has been set for the model: {}'.format(new_index))
            except:
                pass
        if not self.try_model:
            self.print_content(info)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion_dict = {
            'QSQF-C': lstm_cq.loss_fn_crps,
            'RNN-SF': lstm_cq.loss_fn_crps,
            'LSTM-CQ': lstm_cq.loss_fn_crps,
            'LSTM-AQ': lstm_cq.loss_fn_crps,
            'LSTM-YJQR': lstm_yjqr.loss_fn,
            'LSTM-ED-YJQR': lstm_yjqr.loss_fn,
            'LSTM-AQ1': lstm_cq.loss_fn_crps,
            'LSTM-AQ2': lstm_cq.loss_fn_mse,
            'LSTM-AQ3': lstm_cq.loss_fn_mae,
            'LSTM-AQ4': lstm_cq.loss_fn_quantiles,
            'QSQF-C1': lstm_cq.loss_fn_crps
        }

        loss = self.args.loss
        if loss == 'auto':
            if self.args.task_name == 'classification':
                return nn.CrossEntropyLoss()
            elif self.args.task_name != "probability_forecast":
                return nn.MSELoss()
            else:
                if self.args.model in criterion_dict:
                    return criterion_dict[self.args.model]
                else:
                    raise NotImplementedError(f'Loss function for model {self.args.model} not implemented')
        else:
            if loss == 'MSE':
                return nn.MSELoss()
            elif loss == 'MAPE':
                return mape_loss()
            elif loss == 'MASE':
                return mase_loss()
            elif loss == 'SMAPE':
                return smape_loss()
            elif loss == 'CrossEntropy':
                return nn.CrossEntropyLoss()
            else:
                raise NotImplementedError(f'Loss function {loss} not implemented')

    def train(self, setting):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def test(self, setting, test=False):
        pass

    def predict(self, setting, load=False):
        pass

    @staticmethod
    def _check_folders(folders):
        if not isinstance(folders, list):
            folders = [folders]

        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def print_content(self, content, write=False):
        print(content)
        if self.save_process:
            self.process_content = self.process_content + content + "\n"
            if write:
                f = open(self.process_file_path, 'a')
                f.write(self.process_content)
                f.write('\n')
                f.write('\n')
                f.close()
                self.process_content = ""
