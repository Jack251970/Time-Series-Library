import argparse

import torch

from hyper_optimizer.optimizer import HyperOptimizer


# noinspection DuplicatedCode
def parse_launch_parameters():
    parser = argparse.ArgumentParser(description='Time Series Library')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help="task name, options:['long_term_forecast', 'short_term_forecast', 'imputation', "
                             "'classification', 'anomaly_detection']")
    parser.add_argument('--is_training', type=int, default=1, help='1: train and test, 0: only test')
    parser.add_argument('--model_id', type=str, default='unknown', help='model id for interface')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help="model name, options: ['TimesNet', 'Autoformer', 'Transformer', "
                             "'Nonstationary_Transformer', 'DLinear', 'FEDformer', 'Informer', 'LightTS', 'Reformer', "
                             "'ETSformer', 'PatchTST', 'Pyraformer', 'MICN', 'Crossformer', 'FiLM', 'iTransformer']")

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1',
                        help="dataset type, options: ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom', 'm4', 'PSM', "
                             "'MSL', 'SMAP', 'SMD', 'SWAT', 'UEA']")
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate '
                             'predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min '
                             'or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

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
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoders')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
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
        'checkpoints': _args.checkpoints,

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

        # optimization
        'num_workers': _args.num_workers,
        'itr': _args.itr,
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
    }


# noinspection DuplicatedCode
def prepare_config(_params):
    # parse launch parameters
    _args = parse_launch_parameters()

    # load device config
    _args.use_gpu = True if torch.cuda.is_available() and _args.use_gpu else False
    if _args.use_gpu and _args.use_multi_gpu:
        _args.devices = _args.devices.replace(' ', '')
        device_ids = _args.devices.split(',')
        _args.device_ids = [int(id_) for id_ in device_ids]
        _args.gpu = _args.device_ids[0]

    # build model_id for interface
    _args.model_id = f'{_args.target}_{_args.seq_len}_{_args.pred_len}'

    if _params is None:
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
        if 'checkpoints' in _params:
            _args.checkpoints = _params['checkpoints']

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

        # optimization
        if 'num_workers' in _params:
            _args.num_workers = _params['num_workers']
        if 'itr' in _params:
            _args.itr = _params['itr']
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
def build_setting(_args, ii):
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_dm{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
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
        _args.d_ff,
        _args.factor,
        _args.embed,
        _args.distil,
        _args.des, ii)


# noinspection DuplicatedCode
def check_jump_experiment(_parameter):
    return False


# noinspection DuplicatedCode
def get_fieldnames(mode='all'):
    # init the all fieldnames
    all_fieldnames = ['mse', 'mae', 'acc', 'setting', 'seed', 'task_name', 'is_training', 'model_id', 'model', 'data',
                      'data_path', 'features', 'target', 'freq', 'checkpoints', 'seq_len', 'label_len', 'pred_len',
                      'seasonal_patterns', 'inverse', 'mask_rate', 'anomaly_ratio', 'top_k', 'num_kernels', 'enc_in',
                      'dec_in', 'c_out', 'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'moving_avg',
                      'series_decomp_mode', 'factor', 'distil', 'dropout', 'embed', 'activation', 'output_attention',
                      'num_workers', 'itr', 'train_epochs', 'batch_size', 'patience', 'learning_rate', 'des', 'loss',
                      'lradj', 'use_amp', 'use_gpu', 'gpu', 'use_multi_gpu', 'devices', 'p_hidden_dims',
                      'p_hidden_layers']

    # init the fieldnames need to be checked
    _removed_fieldnames = ['mse', 'mae', 'acc', 'model_id', 'root_path', 'checkpoints', 'output_attention',
                           'num_workers', 'use_gpu', 'gpu', 'use_multi_gpu', 'devices']
    checked_fieldnames = [field for field in all_fieldnames if field not in _removed_fieldnames]

    # init the fieldnames need to be showed in csv data file name
    csv_data_fieldnames = ['task_name']

    # init the required fieldnames
    required_fieldnames = ['task_name', 'is_training', 'model_id', 'model', 'data']

    if mode == 'all':
        return all_fieldnames
    elif mode == 'checked':
        return checked_fieldnames
    elif mode == 'csv_data':
        return csv_data_fieldnames
    elif mode == 'required':
        return required_fieldnames
    else:
        raise ValueError("The input 'mode' of get_fieldnames() should be 'all', 'checked' or 'csv_data'!")


# noinspection DuplicatedCode
def get_tags(_args):
    tags = []
    if _args.learning_rate == 0.0001:
        tags.append('large_lr')
    elif _args.learning_rate == 0.00005:
        tags.append('medium_lr')
    elif _args.learning_rate == 0.00001:
        tags.append('small_lr')

    if len(tags) == 0:
        return ''
    else:
        tags_text = ''
        for label in tags:
            tags_text = label + ', '
        tags_text = tags_text[:-2]
        return f'({tags_text})'


# noinspection DuplicatedCode
def get_search_space():
    default_config = {
        'task_name': {'_type': 'single', '_value': 'long_term_forecast'},
        'is_training': {'_type': 'single', '_value': 1},
        'root_path': {'_type': 'single', '_value': './dataset/power/pvod/'},
        'data_path': {'_type': 'single', '_value': 'station00.csv'},
        'target': {'_type': 'single', '_value': 'power'},
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'M'},
        'enc_in': {'_type': 'single', '_value': 14},  # make sure it's same as the feature size
        'dec_in': {'_type': 'single', '_value': 14},  # make sure it's same as the feature size
        'c_out': {'_type': 'single', '_value': 14},
        'des': {'_type': 'single', '_value': 'Exp'},
        'itr': {'_type': 'single', '_value': 1},
        'use_gpu': {'_type': 'single', '_value': True},
        'embed': {'_type': 'single', '_value': 'timeF'},
        'freq': {'_type': 'single', '_value': 't'},
    }

    model_config = {
        # model mode 1: Autoformer
        # 'model': {'_type': 'single', '_value': 'Autoformer'},

        # model mode 2: FEDformer
        # 'model': {'_type': 'single', '_value': 'FEDformer'},

        # model mode 2: Autoformer, FEDformer
        'model': {'_type': 'choice', '_value': ['Autoformer', 'FEDformer']},
    }

    learning_config = {
        # learning mode 1: large lr
        'learning_rate': {'_type': 'single', '_value': 0.0001},
        'train_epochs': {'_type': 'single', '_value': 3},

        # learning mode 2: medium lr
        # 'learning_rate': {'_type': 'single', '_value': 0.00005},
        # 'train_epochs': {'_type': 'single', '_value': 6},

        # learning mode 3: small lr
        # 'learning_rate': {'_type': 'single', '_value': 0.00001},
        # 'train_epochs': {'_type': 'single', '_value': 20},
    }

    heads_config = {
        'n_heads': {'_type': 'single', '_value': 8},
    }

    period_config = {
        # mode 1: short period
        # 'seq_len': {'_type': 'single', '_value': 16},
        # 'label_len': {'_type': 'single', '_value': 16},
        # 'pred_len': {'_type': 'single', '_value': 16},
        # 'e_layers': {'_type': 'single', '_value': 1},
        # 'd_layers': {'_type': 'single', '_value': 1},
        # 'factor': {'_type': 'single', '_value': 2},

        # mode 2: medium period
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 96},
        'pred_len': {'_type': 'single', '_value': 96},
        'e_layers': {'_type': 'single', '_value': 1},
        'd_layers': {'_type': 'single', '_value': 1},
        'factor': {'_type': 'single', '_value': 2},
    }

    decomp_config = {
        # avg
        'series_decomp_mode': {'_type': 'single', '_value': 'avg'},
        # adp_avg
        # 'series_decomp_mode': {'_type': 'single', '_value': 'adp_avg'},
        # mode 2: Transformer, Informer, Reformer
        # 'series_decomp_mode': {'_type': 'single', '_value': 'avg'},
    }

    return {**default_config, **model_config, **learning_config, **heads_config, **period_config, **decomp_config}


h = HyperOptimizer(get_fieldnames, get_search_space, prepare_config, build_setting, build_config_dict,
                   get_tags=get_tags, check_jump_experiment=check_jump_experiment)

if __name__ == "__main__":
    h.start_search(0, False, False)
