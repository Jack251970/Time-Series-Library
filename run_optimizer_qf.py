from hyper_optimizer.basic_settings import prepare_config, build_setting, build_config_dict, set_args, get_fieldnames
from hyper_optimizer.optimizer import HyperOptimizer


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
        'root_path': {'_type': 'single', '_value': './dataset/wind/'},
        # 'data_path': {'_type': 'single', '_value': 'Zone1/Zone1.csv'},
        'data_path': {'_type': 'choice', '_value': ['Zone2/Zone2.csv', 'Zone3/Zone3.csv', 'Zone4/Zone4.csv',
                                                    'Zone5/Zone5.csv', 'Zone6/Zone6.csv', 'Zone7/Zone7.csv',
                                                    'Zone8/Zone8.csv', 'Zone9/Zone9.csv', 'Zone10/Zone10.csv']},
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
    }

    lstm_cq_config = {
        # model
        'label_len': {'_type': 'single', '_value': 0},
        'lag': {'_type': 'single', '_value': 3},
        'dropout': {'_type': 'single', '_value': 0},

        'learning_rate': {'_type': 'single', '_value': 0.001},
        # 'train_epochs': {'_type': 'single', '_value': 20},
        'train_epochs': {'_type': 'single', '_value': 50},
        # 'train_epochs': {'_type': 'choice', '_value': [20, 50]},

        'num_spline': {'_type': 'single', '_value': 20},
        'sample_times': {'_type': 'single', '_value': 99},

        'scaler': {'_type': 'single', '_value': 'MinMaxScaler'},

        'reindex': {'_type': 'single', '_value': 1},
        # 'reindex': {'_type': 'choice', '_value': [0, 1]},
    }

    model_configs = {
        'QSQF-C': qsqf_config,
        'LSTM-CQ': lstm_cq_config,
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


h = HyperOptimizer(False, ['LSTM-CQ', 'QSQF-C', 'RNN-SF'],
                   prepare_config, build_setting, build_config_dict, set_args, get_fieldnames, get_search_space,
                   get_model_id_tags=get_model_id_tags, check_jump_experiment=None)
# 2024-04-02 12-15-46: standard and best
# 2024-04-08 16-49-40, 2024-04-08 21-29-59, 2024-04-09 11-29-40: remove strange lines
# 2024-04-09 20-20-21: 1+2 algorithm, 1e-4
# 2024-04-09 21-05-56: 1+2 algorithm, 1e-6
# 2024-04-09 21-59-59: 1 algorithm, 1e-6
h.config_optimizer_settings(custom_test_time="2024-04-09 21-05-56", scan_all_csv=True, try_model=False, force_exp=False,
                            add_tags=[])

if __name__ == "__main__":
    h.start_search(0)
