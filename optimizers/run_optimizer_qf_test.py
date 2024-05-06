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
    return _config


# noinspection DuplicatedCode
def get_search_space():
    default_config = {
        'task_name': {'_type': 'single', '_value': 'probability_forecast'},
        'is_training': {'_type': 'single', '_value': 0},
        'des': {'_type': 'single', '_value': 'Exp'},
        'use_gpu': {'_type': 'single', '_value': True},
        'embed': {'_type': 'single', '_value': 'timeF'},
        'freq': {'_type': 'single', '_value': 't'},
        'batch_size': {'_type': 'single', '_value': 256},
    }

    dataset_config = {
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'MS'},
        'root_path': {'_type': 'single', '_value': '../dataset/'},
        'data_path': {'_type': 'single', '_value': 'electricity/electricity.csv'},
    }

    learning_config = {
        'learning_rate': {'_type': 'single', '_value': 0.0001},
        'train_epochs': {'_type': 'single', '_value': 3},
    }

    period_config = {
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 16},
        'pred_len': {'_type': 'single', '_value': 96},
        'e_layers': {'_type': 'single', '_value': 1},
        'd_layers': {'_type': 'single', '_value': 1},
    }

    qsqf_config = {
        # model
        'label_len': {'_type': 'single', '_value': 0},
        'lag': {'_type': 'single', '_value': 3},
        'dropout': {'_type': 'single', '_value': 0},

        'learning_rate': {'_type': 'single', '_value': 0.001},
        'train_epochs': {'_type': 'single', '_value': 50},

        'num_spline': {'_type': 'single', '_value': 20},
        'sample_times': {'_type': 'single', '_value': 99},

        'scaler': {'_type': 'single', '_value': 'MinMaxScaler'},

        'lstm_hidden_size': {'_type': 'single', '_value': 40},
        'lstm_layers': {'_type': 'single', '_value': 2},
    }

    lstm_ed_cq_config = {
        # model
        'label_len': {'_type': 'single', '_value': 0},
        'lag': {'_type': 'single', '_value': 3},
        'dropout': {'_type': 'single', '_value': 0},

        'scaler': {'_type': 'single', '_value': 'MinMaxScaler'},
        'reindex': {'_type': 'single', '_value': 0},

        'learning_rate': {'_type': 'single', '_value': 0.001},
        'train_epochs': {'_type': 'single', '_value': 50},

        'num_spline': {'_type': 'single', '_value': 20},
        'sample_times': {'_type': 'single', '_value': 99},

        # Config the target parameter here!
        # 2024-04-23 10-33-28: Electricity, 96_0_16
        # 'data_path': {'_type': 'single', '_value': 'electricity/electricity.csv'},
        # 'pred_len': {'_type': 'single', '_value': 16},
        # 'lstm_hidden_size': {'_type': 'single', '_value': 40},
        # 'lstm_layers': {'_type': 'single', '_value': 3},
        # 'n_heads': {'_type': 'single', '_value': 1},
        # 'd_model': {'_type': 'single', '_value': 24},

        # 2024-04-23 17-32-45: Electricity, 96_0_96
        # 'data_path': {'_type': 'single', '_value': 'electricity/electricity.csv'},
        # 'pred_len': {'_type': 'single', '_value': 96},
        # 'lstm_hidden_size': {'_type': 'single', '_value': 40},
        # 'lstm_layers': {'_type': 'single', '_value': 1},
        # 'n_heads': {'_type': 'single', '_value': 4},
        # 'd_model': {'_type': 'single', '_value': 24},

        # 2024-05-02 23-51-22: Electricity, 96_0_192
        # 'data_path': {'_type': 'single', '_value': 'electricity/electricity.csv'},
        # 'pred_len': {'_type': 'single', '_value': 192},
        # 'lstm_hidden_size': {'_type': 'single', '_value': 40},
        # 'lstm_layers': {'_type': 'single', '_value': 3},
        # 'n_heads': {'_type': 'single', '_value': 4},
        # 'd_model': {'_type': 'single', '_value': 40},

        # 2024-04-24 17-16-19: Exchange, 96_0_96
        'data_path': {'_type': 'single', '_value': 'exchange_rate/exchange_rate.csv'},
        'pred_len': {'_type': 'single', '_value': 96},
        'lstm_hidden_size': {'_type': 'single', '_value': 40},
        'lstm_layers': {'_type': 'single', '_value': 1},
        'n_heads': {'_type': 'single', '_value': 2},
        'd_model': {'_type': 'single', '_value': 64},

        # 2024-05-02 01-37-55: Exchange, 96_0_192
        # 'data_path': {'_type': 'single', '_value': 'exchange_rate/exchange_rate.csv'},
        # 'pred_len': {'_type': 'single', '_value': 192},
        # 'lstm_hidden_size': {'_type': 'single', '_value': 40},
        # 'lstm_layers': {'_type': 'single', '_value': 1},
        # 'n_heads': {'_type': 'single', '_value': 4},
        # 'd_model': {'_type': 'single', '_value': 64},

        'custom_params': {'_type': 'single', '_value': 'AA_attn_dhz_ap1_norm'},
    }

    model_configs = {
        'LSTM-ED-CQ': lstm_ed_cq_config,
    }

    return [default_config, dataset_config, learning_config, period_config], model_configs


h = HyperParameterOptimizer(script_mode=False, models=['LSTM-ED-CQ'],
                            get_search_space=get_search_space, link_fieldnames_data=link_fieldnames_data)
h.config_optimizer_settings(root_path='..', scan_all_csv=True, try_model=False, force_exp=True,
                            custom_test_time='2024-04-24 17-16-19')

if __name__ == "__main__":
    h.start_search(process_index=0)
