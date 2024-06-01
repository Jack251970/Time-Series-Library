from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer


# noinspection DuplicatedCode
def link_fieldnames_data(_config):
    _data_path = _config['data_path']
    if _data_path == 'electricity/electricity.csv':
        # electricity dataset
        _config['reindex_tolerance'] = 0.80
        _config['enc_in'] = 321
        _config['dec_in'] = 321
        _config['c_out'] = 321
    elif (_data_path == 'ETT-small/ETTh1.csv' or _data_path == 'ETT-small/ETTh2.csv' or
          _data_path == 'ETT-small/ETTm1.csv' or _data_path == 'ETT-small/ETTm2.csv'):
        # ETT dataset
        _config['enc_in'] = 7
        _config['dec_in'] = 7
        _config['c_out'] = 7
    elif _data_path == 'exchange_rate/exchange_rate.csv':
        # exchange rate dataset
        _config['enc_in'] = 8
        _config['dec_in'] = 8
        _config['c_out'] = 8
    elif _data_path == 'illness/national_illness.csv':
        # illness dataset
        _config['enc_in'] = 7
        _config['dec_in'] = 7
        _config['c_out'] = 7
    elif _data_path == 'traffic/traffic.csv':
        # traffic dataset
        _config['enc_in'] = 862
        _config['dec_in'] = 862
        _config['c_out'] = 862
    elif _data_path == 'weather/weather.csv':
        # weather dataset
        _config['enc_in'] = 21
        _config['dec_in'] = 21
        _config['c_out'] = 21
    elif _data_path == 'pvod/station00.csv':
        # solar dataset
        _config['target'] = 'power'
        _config['enc_in'] = 14
        _config['dec_in'] = 14
        _config['c_out'] = 14
    elif _data_path == 'wind/Zone1/Zone1.csv':
        # wind power dataset
        _config['target'] = 'wind'
        _config['enc_in'] = 5
        _config['dec_in'] = 5
        _config['c_out'] = 5
    _pred_len = _config['pred_len']
    if _pred_len == 16 or _pred_len == 32:
        _config['label_len'] = 16
    elif _pred_len == 96 or _pred_len == 192:
        _config['label_len'] = 48
    return _config


# noinspection DuplicatedCode
def get_search_space():
    default_config = {
        'task_name': {'_type': 'single', '_value': 'long_term_forecast'},
        'is_training': {'_type': 'single', '_value': 1},
        'des': {'_type': 'single', '_value': 'Exp'},
        'use_gpu': {'_type': 'single', '_value': True},
        'embed': {'_type': 'single', '_value': 'timeF'},
        'freq': {'_type': 'single', '_value': 't'},
        'batch_size': {'_type': 'single', '_value': 256},
    }

    dataset_config = {
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'MS'},
        'root_path': {'_type': 'single', '_value': './dataset/'},
        # 'data_path': {'_type': 'choice', '_value': ['electricity/electricity.csv', 'exchange_rate/exchange_rate.csv']},
        'data_path': {'_type': 'choice', '_value': ['traffic/traffic.csv']},
        'pin_memory': {'_type': 'single', '_value': False},
    }

    learning_config = {
        'learning_rate': {'_type': 'single', '_value': 0.00005},
        'train_epochs': {'_type': 'single', '_value': 10},
    }

    period_config = {
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 16},
        'pred_len': {'_type': 'choice', '_value': [16, 32, 64, 96]},
        'e_layers': {'_type': 'single', '_value': 1},
        'd_layers': {'_type': 'single', '_value': 1},
    }

    transformer_config = {
        'scaler': {'_type': 'single', '_value': 'MinMaxScaler'},
    }

    model_configs = {
        'Transformer': transformer_config,
        'Autoformer': transformer_config,
        'Informer': transformer_config,
        'Reformer': transformer_config,
    }

    return [default_config, dataset_config, learning_config, period_config], model_configs


h = HyperParameterOptimizer(script_mode=False, models=['Transformer', 'Informer', 'Reformer'],
                            get_search_space=get_search_space, link_fieldnames_data=link_fieldnames_data)
h.config_optimizer_settings(root_path='.', scan_all_csv=True, try_model=False, force_exp=False)
