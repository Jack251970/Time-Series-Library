import argparse
import datetime
import os
import time

import torch

from hyper_optimizer.optimizer import HyperOptimizer


# noinspection DuplicatedCode
def parse_launch_parameters(_script_mode):
    parser = argparse.ArgumentParser(description='Time Series Library')

    # basic config
    parser.add_argument('--task_name', type=str, required=_script_mode, default='long_term_forecast',
                        help="task name, options:['long_term_forecast', 'short_term_forecast', 'imputation', "
                             "'classification', 'anomaly_detection', 'probability_forecast']")
    parser.add_argument('--is_training', type=int, required=_script_mode, default=1,
                        help='1: train and test, 0: only test')
    parser.add_argument('--model_id', type=str, required=_script_mode, default='unknown',
                        help='model id for interface')
    parser.add_argument('--model', type=str, required=_script_mode, default='Autoformer',
                        help="model name, options: ['TimesNet', 'Autoformer', 'Transformer', "
                             "'Nonstationary_Transformer', 'DLinear', 'FEDformer', 'Informer', 'LightTS', 'Reformer', "
                             "'ETSformer', 'PatchTST', 'Pyraformer', 'MICN', 'Crossformer', 'FiLM', 'iTransformer', "
                             "'Koopa', 'QSQF-AB', 'QSQF-C', 'Transformer-QSQF', 'Autoformer-QSQF']")

    # data loader
    parser.add_argument('--data', type=str, required=_script_mode, default='ETTm1',
                        help="dataset type, options: ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom', 'm4', 'PSM', "
                             "'MSL', 'SMAP', 'SMD', 'SWAT', 'UEA']")
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:uni-variate '
                             'predict uni-variate, MS:multivariate predict uni-variate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min '
                             'or 3h')
    parser.add_argument('--lag', type=int, default=0, help='lag of time series, only for RNN & LSTM related model')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scaler', type=str, default='StandardScaler', help='feature scaling method')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average in encoders')
    parser.add_argument('--series_decomp_mode', type=str, default='avg',
                        help="series decomposition mod, options: ['avg', 'exp', 'stl', 'adp_avg', 'moe']")
    parser.add_argument('--factor', type=int, default=1, help='attn factor or router factor in Crossformer')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoders')
    parser.add_argument('--channel_independence', type=int, default=0, help='1: channel dependence 0: channel '
                                                                            'independence for FreTS model')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='deprecated')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='auto', help='loss function, detect automatically if not set')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # QSQF
    parser.add_argument('--num_spline', type=int, default=20, help='number of spline')
    parser.add_argument('--sample_times', type=int, default=99, help='sample times')

    return parser.parse_args()


# noinspection DuplicatedCode
def build_config_dict(_args):
    return {
        # basic config
        'task_name': _args.task_name,
        'is_training': _args.is_training,
        'model_id': _args.model_id,
        'model': _args.model,

        # data loader
        'data': _args.data,
        'root_path': _args.root_path,
        'data_path': _args.data_path,
        'features': _args.features,
        'target': _args.target,
        'freq': _args.freq,
        'lag': _args.lag,
        'checkpoints': _args.checkpoints,
        'scaler': _args.scaler,

        # forecasting task
        'seq_len': _args.seq_len,
        'label_len': _args.label_len,
        'pred_len': _args.pred_len,
        'seasonal_patterns': _args.seasonal_patterns,
        'inverse': _args.inverse,

        # imputation task
        'mask_rate': _args.mask_rate,

        # anomaly detection task
        'anomaly_ratio': _args.anomaly_ratio,

        # model define
        'top_k': _args.top_k,
        'num_kernels': _args.num_kernels,
        'enc_in': _args.enc_in,
        'dec_in': _args.dec_in,
        'c_out': _args.c_out,
        'd_model': _args.d_model,
        'n_heads': _args.n_heads,
        'e_layers': _args.e_layers,
        'd_layers': _args.d_layers,
        'd_ff': _args.d_ff,
        'moving_avg': _args.moving_avg,
        'series_decomp_mode': _args.series_decomp_mode,
        'factor': _args.factor,
        'distil': _args.distil,
        'dropout': _args.dropout,
        'embed': _args.embed,
        'activation': _args.activation,
        'output_attention': _args.output_attention,
        'channel_independence': _args.channel_independence,

        # optimization
        'num_workers': _args.num_workers,
        'train_epochs': _args.train_epochs,
        'batch_size': _args.batch_size,
        'patience': _args.patience,
        'learning_rate': _args.learning_rate,
        'des': _args.des,
        'loss': _args.loss,
        'lradj': _args.lradj,
        'use_amp': _args.use_amp,

        # GPU
        'use_gpu': _args.use_gpu,
        'gpu': _args.gpu,
        'use_multi_gpu': _args.use_multi_gpu,
        'devices': _args.devices,

        # de-stationary projector params
        'p_hidden_dims': _args.p_hidden_dims,
        'p_hidden_layers': _args.p_hidden_layers,

        # QSQF
        'num_spline': _args.num_spline,
        'sample_times': _args.sample_times
    }


# noinspection DuplicatedCode
def set_args(_args, _config):
    # basic config
    _args.task_name = _config['task_name']
    _args.is_training = _config['is_training']
    _args.model_id = _config['model_id']
    _args.model = _config['model']

    # data loader
    _args.data = _config['data']
    _args.root_path = _config['root_path']
    _args.data_path = _config['data_path']
    _args.features = _config['features']
    _args.target = _config['target']
    _args.freq = _config['freq']
    _args.lag = _config['lag']
    _args.checkpoints = _config['checkpoints']
    _args.scaler = _config['scaler']

    # forecasting task
    _args.seq_len = _config['seq_len']
    _args.label_len = _config['label_len']
    _args.pred_len = _config['pred_len']
    _args.seasonal_patterns = _config['seasonal_patterns']
    _args.inverse = _config['inverse']

    # imputation task
    _args.mask_rate = _config['mask_rate']

    # anomaly detection task
    _args.anomaly_ratio = _config['anomaly_ratio']

    # model define
    _args.top_k = _config['top_k']
    _args.num_kernels = _config['num_kernels']
    _args.enc_in = _config['enc_in']
    _args.dec_in = _config['dec_in']
    _args.c_out = _config['c_out']
    _args.d_model = _config['d_model']
    _args.n_heads = _config['n_heads']
    _args.e_layers = _config['e_layers']
    _args.d_layers = _config['d_layers']
    _args.d_ff = _config['d_ff']
    _args.moving_avg = _config['moving_avg']
    _args.series_decomp_mode = _config['series_decomp_mode']
    _args.factor = _config['factor']
    _args.distil = _config['distil']
    _args.dropout = _config['dropout']
    _args.embed = _config['embed']
    _args.activation = _config['activation']
    _args.output_attention = _config['output_attention']
    _args.channel_independence = _config['channel_independence']

    # optimization
    _args.num_workers = _config['num_workers']
    _args.train_epochs = _config['train_epochs']
    _args.batch_size = _config['batch_size']
    _args.patience = _config['patience']
    _args.learning_rate = _config['learning_rate']
    _args.des = _config['des']
    _args.loss = _config['loss']
    _args.lradj = _config['lradj']
    _args.use_amp = _config['use_amp']

    # GPU
    _args.use_gpu = _config['use_gpu']
    _args.gpu = _config['gpu']
    _args.use_multi_gpu = _config['use_multi_gpu']
    _args.devices = _config['devices']

    # de-stationary projector params
    _args.p_hidden_dims = _config['p_hidden_dims']
    _args.p_hidden_layers = _config['p_hidden_layers']

    # QSQF
    _args.num_spline = _config['num_spline']
    _args.sample_times = _config['sample_times']

    return _args


# noinspection DuplicatedCode
def prepare_config(_params, _script_mode=False):
    # parse launch parameters
    _args = parse_launch_parameters(_script_mode)

    # load device config
    _args.use_gpu = True if torch.cuda.is_available() and _args.use_gpu else False
    if _args.use_gpu and _args.use_multi_gpu:
        _args.devices = _args.devices.replace(' ', '')
        device_ids = _args.devices.split(',')
        _args.device_ids = [int(id_) for id_ in device_ids]
        _args.gpu = _args.device_ids[0]

    # build model_id for interface
    _args.model_id = f'{_args.target}_{_args.seq_len}_{_args.pred_len}'

    if _script_mode is True:
        return _args
    else:
        # load optimized parameters from _params
        # basic config
        if 'task_name' in _params:
            _args.task_name = _params['task_name']
        if 'is_training' in _params:
            _args.is_training = _params['is_training']
        if 'model_id' in _params:
            _args.model_id = _params['model_id']
        if 'model' in _params:
            _args.model = _params['model']

        # data loader
        if 'data' in _params:
            _args.data = _params['data']
        if 'root_path' in _params:
            _args.root_path = _params['root_path']
        if 'data_path' in _params:
            _args.data_path = _params['data_path']
        if 'features' in _params:
            _args.features = _params['features']
        if 'target' in _params:
            _args.target = _params['target']
        if 'freq' in _params:
            _args.freq = _params['freq']
        if 'lag' in _params:
            _args.lag = _params['lag']
        if 'checkpoints' in _params:
            _args.checkpoints = _params['checkpoints']
        if 'scaler' in _params:
            _args.scaler = _params['scaler']

        # forecasting task
        if 'seq_len' in _params:
            _args.seq_len = _params['seq_len']
        if 'label_len' in _params:
            _args.label_len = _params['label_len']
        if 'pred_len' in _params:
            _args.pred_len = _params['pred_len']
        if 'seasonal_patterns' in _params:
            _args.seasonal_patterns = _params['seasonal_patterns']
        if 'inverse' in _params:
            _args.inverse = _params['inverse']

        # inputation task
        if 'mask_rate' in _params:
            _args.mask_rate = _params['mask_rate']

        # anomaly detection task
        if 'anomaly_ratio' in _params:
            _args.anomaly_ratio = _params['anomaly_ratio']

        # model define
        if 'top_k' in _params:
            _args.top_k = _params['top_k']
        if 'num_kernels' in _params:
            _args.num_kernels = _params['num_kernels']
        if 'enc_in' in _params:
            _args.enc_in = _params['enc_in']
        if 'dec_in' in _params:
            _args.dec_in = _params['dec_in']
        if 'c_out' in _params:
            _args.c_out = _params['c_out']
        if 'd_model' in _params:
            _args.d_model = _params['d_model']
        if 'n_heads' in _params:
            _args.n_heads = _params['n_heads']
        if 'e_layers' in _params:
            _args.e_layers = _params['e_layers']
        if 'd_layers' in _params:
            _args.d_layers = _params['d_layers']
        if 'd_ff' in _params:
            _args.d_ff = _params['d_ff']
        if 'moving_avg' in _params:
            _args.moving_avg = _params['moving_avg']
        if 'series_decomp_mode' in _params:
            _args.series_decomp_mode = _params['series_decomp_mode']
        if 'factor' in _params:
            _args.factor = _params['factor']
        if 'distil' in _params:
            _args.distil = _params['distil']
        if 'dropout' in _params:
            _args.dropout = _params['dropout']
        if 'embed' in _params:
            _args.embed = _params['embed']
        if 'activation' in _params:
            _args.activation = _params['activation']
        if 'output_attention' in _params:
            _args.output_attention = _params['output_attention']
        if 'channel_independence' in _params:
            _args.channel_independence = _params['channel_independence']

        # optimization
        if 'num_workers' in _params:
            _args.num_workers = _params['num_workers']
        if 'train_epochs' in _params:
            _args.train_epochs = _params['train_epochs']
        if 'batch_size' in _params:
            _args.batch_size = _params['batch_size']
        if 'patience' in _params:
            _args.patience = _params['patience']
        if 'learning_rate' in _params:
            _args.learning_rate = _params['learning_rate']
        if 'des' in _params:
            _args.des = _params['des']
        if 'loss' in _params:
            _args.loss = _params['loss']
        if 'lradj' in _params:
            _args.lradj = _params['lradj']
        if 'use_amp' in _params:
            _args.use_amp = _params['use_amp']

        # GPU
        if 'use_gpu' in _params:
            _args.use_gpu = _params['use_gpu']
        if 'gpu' in _params:
            _args.gpu = _params['gpu']
        if 'use_multi_gpu' in _params:
            _args.use_multi_gpu = _params['use_multi_gpu']
        if 'devices' in _params:
            _args.devices = _params['devices']

        # de-stationary projector params
        if 'p_hidden_dims' in _params:
            _args.p_hidden_dims = _params['p_hidden_dims']
        if 'p_hidden_layers' in _params:
            _args.p_hidden_layers = _params['p_hidden_layers']

        # QSQF
        if 'num_spline' in _params:
            _args.num_spline = _params['num_spline']
        if 'sample_times' in _params:
            _args.sample_times = _params['sample_times']

        # build new model_id for interface
        _args.model_id = f'{_args.target}_{_args.seq_len}_{_args.pred_len}'

        # load new device config
        _args.use_gpu = True if torch.cuda.is_available() and _args.use_gpu else False
        if _args.use_gpu and _args.use_multi_gpu:
            _args.devices = _args.devices.replace(' ', '')
            device_ids = _args.devices.split(',')
            _args.device_ids = [int(id_) for id_ in device_ids]
            _args.gpu = _args.device_ids[0]

    return _args


# noinspection DuplicatedCode
def build_setting(_args, _time, _format):
    prefix = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dm{}_ma{}_df{}_fc{}_eb{}_dt{}_de{}'.format(
        _args.task_name,
        _args.model_id,
        _args.model,
        _args.data,
        _args.features,
        _args.seq_len,
        _args.label_len,
        _args.pred_len,
        _args.d_model,
        _args.n_heads,
        _args.e_layers,
        _args.d_layers,
        _args.series_decomp_mode,
        _args.moving_avg,
        _args.d_ff,
        _args.factor,
        _args.embed,
        _args.distil,
        _args.des)
    checkpoints_folder = _args.checkpoints

    if not _args.is_training:
        checkpoints = os.listdir(checkpoints_folder)
        latest_time = None
        for checkpoint in checkpoints:
            if not os.listdir(os.path.join(checkpoints_folder, checkpoint)):
                continue
            if checkpoint.startswith(prefix):
                checkpoint_time = checkpoint.split('_')[-1]
                checkpoint_time = datetime.datetime.strptime(checkpoint_time, _format)
                if latest_time is None or checkpoint_time > latest_time:
                    latest_time = checkpoint_time

        if latest_time is not None:
            return '{}_{}'.format(prefix, latest_time.strftime(_format))

    return '{}_{}'.format(prefix, time.strftime(_format, _time))


# noinspection DuplicatedCode
def check_jump_experiment(_parameter):
    return False


# noinspection DuplicatedCode
def get_fieldnames(mode='all'):
    # init the all fieldnames
    all_fieldnames = ['mse', 'mae', 'acc', 'smape', 'f_score', 'crps', 'mre', 'pinaw', 'setting', 'seed', 'task_name',
                      'is_training', 'model_id', 'model', 'data', 'data_path', 'features', 'target', 'freq', 'lag',
                      'checkpoints', 'scaler', 'seq_len', 'label_len', 'pred_len', 'seasonal_patterns', 'inverse',
                      'mask_rate', 'anomaly_ratio', 'top_k', 'num_kernels', 'enc_in', 'dec_in', 'c_out', 'd_model',
                      'n_heads', 'e_layers', 'd_layers', 'd_ff', 'moving_avg', 'series_decomp_mode', 'factor', 'distil',
                      'dropout', 'embed', 'activation', 'output_attention', 'channel_independence', 'num_workers',
                      'train_epochs', 'batch_size', 'patience', 'learning_rate', 'des', 'loss', 'lradj', 'use_amp',
                      'use_gpu', 'gpu', 'use_multi_gpu', 'devices', 'run_time', 'p_hidden_dims', 'p_hidden_layers',
                      'num_spline', 'sample_times']

    # init the fieldnames need to be checked
    _removed_fieldnames = ['mse', 'mae', 'acc', 'smape', 'f_score', 'crps', 'mre', 'pinaw', 'setting', 'is_training',
                           'model_id', 'root_path', 'checkpoints', 'output_attention', 'num_workers', 'use_gpu', 'gpu',
                           'use_multi_gpu', 'devices', 'run_time']
    checked_fieldnames = [field for field in all_fieldnames if field not in _removed_fieldnames]

    # init the required fieldnames
    required_fieldnames = ['task_name', 'is_training', 'model', 'data']

    if mode == 'all':
        return all_fieldnames
    elif mode == 'checked':
        return checked_fieldnames
    elif mode == 'required':
        return required_fieldnames
    else:
        raise ValueError("The input 'mode' of get_fieldnames() should be 'all', 'checked' or 'csv_data'!")


# noinspection DuplicatedCode
def get_model_id_tags(_args, _add_tags):
    tags = []
    if _args.learning_rate == 0.001:
        tags.append('extra_large_lr')
    elif _args.learning_rate == 0.0001:
        tags.append('large_lr')
    elif _args.learning_rate == 0.00005:
        tags.append('medium_lr')
    elif _args.learning_rate == 0.00001:
        tags.append('small_lr')

    for add_tag in _add_tags:
        tags.append(add_tag)

    if len(tags) == 0:
        return ''
    else:
        tags_text = ''
        for label in tags:
            tags_text = tags_text + label + ', '
        tags_text = tags_text[:-2]
        return f'({tags_text})'


# noinspection DuplicatedCode
def get_search_space(_model):
    default_config = {
        'task_name': {'_type': 'single', '_value': 'probability_forecast'},
        'is_training': {'_type': 'single', '_value': 1},
        'des': {'_type': 'single', '_value': 'Exp'},
        'use_gpu': {'_type': 'single', '_value': True},
        'embed': {'_type': 'single', '_value': 'timeF'},
        'freq': {'_type': 'single', '_value': 't'},
        'batch_size': {'_type': 'single', '_value': 256},
    }

    dataset_config = {
        # solar dataset
        # 'root_path': {'_type': 'single', '_value': './dataset/power/pvod/'},
        # 'data_path': {'_type': 'single', '_value': 'station00.csv'},
        # 'target': {'_type': 'single', '_value': 'power'},
        # 'data': {'_type': 'single', '_value': 'custom'},
        # 'features': {'_type': 'single', '_value': 'MS'},
        # 'enc_in': {'_type': 'single', '_value': 14},  # make sure it's same as the feature size
        # 'dec_in': {'_type': 'single', '_value': 14},  # make sure it's same as the feature size
        # 'c_out': {'_type': 'single', '_value': 14},

        # wind dataset
        'root_path': {'_type': 'single', '_value': './dataset/wind/Zone1/'},
        'data_path': {'_type': 'single', '_value': 'Zone1.csv'},
        'target': {'_type': 'single', '_value': 'wind'},
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'MS'},
        'enc_in': {'_type': 'single', '_value': 5},
        'dec_in': {'_type': 'single', '_value': 5},
        'c_out': {'_type': 'single', '_value': 5},
    }

    learning_config = {
        # learning mode 1: extra large lr
        # 'learning_rate': {'_type': 'single', '_value': 0.001},
        # 'train_epochs': {'_type': 'single', '_value': 3},

        # learning mode 2: large lr
        'learning_rate': {'_type': 'single', '_value': 0.0001},
        'train_epochs': {'_type': 'single', '_value': 3},

        # learning mode 3: medium lr
        # 'learning_rate': {'_type': 'single', '_value': 0.00005},
        # 'train_epochs': {'_type': 'single', '_value': 6},

        # learning mode 4: small lr
        # 'learning_rate': {'_type': 'single', '_value': 0.00001},
        # 'train_epochs': {'_type': 'single', '_value': 10},
    }

    period_config = {
        # mode 1: short period 1
        # 'seq_len': {'_type': 'single', '_value': 16},
        # 'label_len': {'_type': 'single', '_value': 16},
        # 'pred_len': {'_type': 'single', '_value': 16},
        # 'e_layers': {'_type': 'single', '_value': 1},
        # 'd_layers': {'_type': 'single', '_value': 1},

        # mode 2: short period 2
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 16},
        'pred_len': {'_type': 'single', '_value': 16},
        'e_layers': {'_type': 'single', '_value': 1},
        'd_layers': {'_type': 'single', '_value': 1},

        # mode 3: medium period
        # 'seq_len': {'_type': 'single', '_value': 96},
        # 'label_len': {'_type': 'single', '_value': 96},
        # 'pred_len': {'_type': 'single', '_value': 96},
        # 'e_layers': {'_type': 'single', '_value': 1},
        # 'd_layers': {'_type': 'single', '_value': 1},
    }

    autoformer_config = {
        'factor': {'_type': 'single', '_value': 2},

        # avg
        # 'series_decomp_mode': {'_type': 'single', '_value': 'avg'},
        'moving_avg': {'_type': 'single', '_value': 25},
        # adp_avg
        # 'series_decomp_mode': {'_type': 'single', '_value': 'adp_avg'},
        # 'moving_avg': {'_type': 'single', '_value': 41},
    }

    fedformer_config = {
        'factor': {'_type': 'single', '_value': 2},

        # avg
        # 'series_decomp_mode': {'_type': 'single', '_value': 'avg'},
        # 'moving_avg': {'_type': 'single', '_value': 25},
        # adp_avg
        'series_decomp_mode': {'_type': 'single', '_value': 'adp_avg'},
        'moving_avg': {'_type': 'single', '_value': 31},
    }

    crossformer_config = {
        'factor': {'_type': 'single', '_value': 7},
    }

    timesnet_config = {
        'factor': {'_type': 'single', '_value': 3},

        'top_k': {'_type': 'single', '_value': 5},
        'd_model': {'_type': 'single', '_value': 32},

        'd_ff': {'_type': 'single', '_value': 32},
    }

    transformer_config = {
    }

    qsqf_config = {
        # model
        'label_len': {'_type': 'single', '_value': 0},
        'lag': {'_type': 'single', '_value': 3},
        'dropout': {'_type': 'single', '_value': 0},

        'learning_rate': {'_type': 'single', '_value': 0.001},
        'train_epochs': {'_type': 'single', '_value': 20},

        'num_spline': {'_type': 'single', '_value': 20},
        'sample_times': {'_type': 'single', '_value': 99},

        'scaler': {'_type': 'single', '_value': 'MinMaxScaler'},
    }

    transformer_qsqf_config = {
        'dropout': {'_type': 'single', '_value': 0.1},

        'e_layers': {'_type': 'single', '_value': 2},
        'd_layers': {'_type': 'single', '_value': 1},
        'train_epochs': {'_type': 'single', '_value': 10},

        'sample_times': {'_type': 'single', '_value': 99},
    }

    autoformer_qsqf_config = {
        'd_model': {'_type': 'single', '_value': 512},
        'train_epochs': {'_type': 'single', '_value': 10},
        'c_out': {'_type': 'single', '_value': 5},

        'num_spline': {'_type': 'single', '_value': 5},

        'scaler': {'_type': 'single', '_value': 'MinMaxScaler'},
    }

    model_configs = {
        'Autoformer': autoformer_config,
        'FEDformer': fedformer_config,
        'Crossformer': crossformer_config,
        'TimesNet': timesnet_config,
        'Transformer': transformer_config,
        'QSQF-AB': qsqf_config,
        'QSQF-C': qsqf_config,
        'Transformer-QSQF': transformer_qsqf_config,
        'Autoformer-QSQF': autoformer_qsqf_config
    }

    # get config for specific model
    model_config = model_configs[_model] if model_configs.get(_model) else {}
    model_config['model'] = {'_type': 'single', '_value': _model}

    # get config
    _config = {**default_config, **dataset_config, **learning_config, **period_config}

    # integrate model config
    for key, value in model_config.items():
        _config[key] = value

    return _config


h = HyperOptimizer(False, ['QSQF-C'],
                   prepare_config, build_setting, build_config_dict, set_args, get_fieldnames, get_search_space,
                   get_model_id_tags=get_model_id_tags, check_jump_experiment=check_jump_experiment)
# h.output_script('Power')
h.config_optimizer_settings(scan_all_csv=True, add_tags=[], try_model=False, force_exp=True)

if __name__ == "__main__":
    h.start_search(0)
