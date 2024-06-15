from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer


# noinspection DuplicatedCode
def link_fieldnames_data(_config):
    data_path = _config['data_path']
    model = _config['model']
    pred_len = _config['pred_len']
    if data_path == 'electricity/electricity.csv':
        # electricity dataset
        _config['reindex_tolerance'] = 0.80
        _config['enc_in'] = 321
        _config['dec_in'] = 321
        _config['c_out'] = 321

        if model == 'LSTM-AQ':
            if pred_len == 16:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 3
                _config['n_heads'] = 1
                _config['d_model'] = 24
            elif pred_len == 32:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 3
                _config['n_heads'] = 2
                _config['d_model'] = 24
            elif pred_len == 64:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 1
                _config['n_heads'] = 2
                _config['d_model'] = 40
            elif pred_len == 96:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 1
                _config['n_heads'] = 4
                _config['d_model'] = 24
    elif (data_path == 'ETT-small/ETTh1.csv' or data_path == 'ETT-small/ETTh2.csv' or
          data_path == 'ETT-small/ETTm1.csv' or data_path == 'ETT-small/ETTm2.csv'):
        # ETT dataset
        _config['enc_in'] = 7
        _config['dec_in'] = 7
        _config['c_out'] = 7
    elif data_path == 'exchange_rate/exchange_rate.csv':
        # exchange rate dataset
        _config['enc_in'] = 8
        _config['dec_in'] = 8
        _config['c_out'] = 8

        if model == 'LSTM-AQ':
            if pred_len == 16:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 1
                _config['n_heads'] = 2
                _config['d_model'] = 64
            elif pred_len == 32:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 2
                _config['n_heads'] = 1
                _config['d_model'] = 40
            elif pred_len == 64:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 1
                _config['n_heads'] = 2
                _config['d_model'] = 64
            elif pred_len == 96:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 1
                _config['n_heads'] = 2
                _config['d_model'] = 64
    elif data_path == 'illness/national_illness.csv':
        # illness dataset
        _config['enc_in'] = 7
        _config['dec_in'] = 7
        _config['c_out'] = 7
    elif data_path == 'traffic/traffic.csv':
        # traffic dataset
        _config['enc_in'] = 862
        _config['dec_in'] = 862
        _config['c_out'] = 862

        if model == 'LSTM-AQ':
            if pred_len == 16:
                _config['lstm_hidden_size'] = 64
                _config['lstm_layers'] = 2
                _config['n_heads'] = 2
                _config['d_model'] = 40
            elif pred_len == 32:
                _config['lstm_hidden_size'] = 64
                _config['lstm_layers'] = 2
                _config['n_heads'] = 2
                _config['d_model'] = 40
            elif pred_len == 64:
                _config['lstm_hidden_size'] = 40
                _config['lstm_layers'] = 3
                _config['n_heads'] = 8
                _config['d_model'] = 64
            elif pred_len == 96:
                _config['lstm_hidden_size'] = 64
                _config['lstm_layers'] = 2
                _config['n_heads'] = 2
                _config['d_model'] = 40
    elif data_path == 'weather/weather.csv':
        # weather dataset
        _config['enc_in'] = 21
        _config['dec_in'] = 21
        _config['c_out'] = 21
    elif data_path == 'pvod/station00.csv':
        # solar dataset
        _config['target'] = 'power'
        _config['enc_in'] = 14
        _config['dec_in'] = 14
        _config['c_out'] = 14
    elif data_path == 'wind/Zone1/Zone1.csv':
        # wind power dataset
        _config['target'] = 'wind'
        _config['enc_in'] = 5
        _config['dec_in'] = 5
        _config['c_out'] = 5
    return _config


# noinspection DuplicatedCode
def get_search_space():
    default_config = {
        'task_name': {'_type': 'single', '_value': 'probability_forecast'},
        'is_training': {'_type': 'single', '_value': 1},
        'des': {'_type': 'single', '_value': 'Exp'},
        'use_gpu': {'_type': 'single', '_value': True},
        'embed': {'_type': 'single', '_value': 'timeF'},
        'freq': {'_type': 'single', '_value': 't'},
        'batch_size': {'_type': 'single', '_value': 256},
        'pin_memory': {'_type': 'single', '_value': False},
    }

    dataset_config = {
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'MS'},
        'root_path': {'_type': 'single', '_value': './dataset/'},
        'data_path': {'_type': 'choice', '_value': ['electricity/electricity.csv', 'exchange_rate/exchange_rate.csv',
                                                    'traffic/traffic.csv']},
    }

    learning_config = {
        'learning_rate': {'_type': 'single', '_value': 0.0001},
        'train_epochs': {'_type': 'single', '_value': 3},
    }

    period_config = {
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 1},
        'pred_len': {'_type': 'choice', '_value': [16, 32, 64, 96]},
        'e_layers': {'_type': 'single', '_value': 1},
        'd_layers': {'_type': 'single', '_value': 1},
    }

    lstm_aq_config = {
        'label_len': {'_type': 'single', '_value': 0},
        'lag': {'_type': 'single', '_value': 3},
        'dropout': {'_type': 'single', '_value': 0},

        'scaler': {'_type': 'single', '_value': 'MinMaxScaler'},
        'reindex': {'_type': 'single', '_value': 0},

        'learning_rate': {'_type': 'single', '_value': 0.001},
        'train_epochs': {'_type': 'single', '_value': 50},

        'num_spline': {'_type': 'single', '_value': 20},
        'sample_times': {'_type': 'single', '_value': 99},

        'custom_params': {'_type': 'single', '_value': 'AA_attn_dhz_ap1_norm'},
    }

    model_configs = {
        'QSQF-C1': lstm_aq_config,
    }

    return [default_config, dataset_config, learning_config, period_config], model_configs


h = HyperParameterOptimizer(script_mode=False, models=['QSQF-C1'],
                            get_search_space=get_search_space, link_fieldnames_data=link_fieldnames_data)
h.config_optimizer_settings(root_path='.', data_csv_file='data_qsqf_comp.csv',
                            scan_all_csv=False, try_model=False, force_exp=False, save_process=False)
