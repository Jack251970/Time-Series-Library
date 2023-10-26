import os
import torch
from torch import optim, nn

from data_provider.data_factory import data_provider
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer
from utils.losses import mape_loss, mase_loss, smape_loss


class Exp_Basic(object):
    def __init__(self, args, try_model=False):
        self.args = args
        self.device = self._acquire_device(try_model)
        self.model = self._build_model().to(self.device)
        self.try_model = try_model

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
            if not try_model:
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            if not try_model:
                print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.try_model)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss='MSE'):
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
            raise NotImplementedError

    def train(self, setting):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def test(self, setting, test=False):
        pass

    def predict(self, setting, load=False):
        pass

    # noinspection PyMethodMayBeStatic
    def _check_folders(self, folders):
        if not isinstance(folders, list):
            folders = [folders]

        for folder in folders:
            # Delete blank folders under the folder
            if os.path.exists(folder):
                for path in os.listdir(folder):
                    sub_folder = os.path.join(folder, path)
                    if os.path.isdir(sub_folder) and not os.listdir(sub_folder):
                        os.rmdir(sub_folder)
