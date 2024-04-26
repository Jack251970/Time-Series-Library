from hyper_optimizer.optimizer import HyperOptimizer


# noinspection DuplicatedCode
def get_model_id_tags(_args):
    tags = []
    if _args.learning_rate == 0.001:
        tags.append('extra_large_lr')
    elif _args.learning_rate == 0.0001:
        tags.append('large_lr')
    elif _args.learning_rate == 0.00005:
        tags.append('medium_lr')
    elif _args.learning_rate == 0.00001:
        tags.append('small_lr')
    return tags


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
        # solar dataset
        'root_path': {'_type': 'single', '_value': './dataset/pvod/'},
        'data_path': {'_type': 'single', '_value': 'station00.csv'},
        'target': {'_type': 'single', '_value': 'power'},
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'MS'},
        'enc_in': {'_type': 'single', '_value': 14},  # make sure it's same as the feature size
        'dec_in': {'_type': 'single', '_value': 14},  # make sure it's same as the feature size
        'c_out': {'_type': 'single', '_value': 14},
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
        # 'seq_len': {'_type': 'single', '_value': 96},
        # 'label_len': {'_type': 'single', '_value': 16},
        # 'pred_len': {'_type': 'single', '_value': 16},
        # 'e_layers': {'_type': 'single', '_value': 1},
        # 'd_layers': {'_type': 'single', '_value': 1},

        # mode 3: medium period
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 96},
        'pred_len': {'_type': 'single', '_value': 96},
        'e_layers': {'_type': 'single', '_value': 1},
        'd_layers': {'_type': 'single', '_value': 1},
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

    model_configs = {
        'Autoformer': autoformer_config,
        'FEDformer': fedformer_config,
        'Crossformer': crossformer_config,
        'TimesNet': timesnet_config,
        'Transformer': transformer_config
    }

    return [default_config, dataset_config, learning_config, period_config], model_configs


h = HyperOptimizer(script_mode=False, models=['Autoformer'],
                   get_search_space=get_search_space, get_model_id_tags=get_model_id_tags)
# h.output_script('Power')
h.config_optimizer_settings(custom_test_time="", scan_all_csv=True, try_model=False, force_exp=False, add_tags=[])

if __name__ == "__main__":
    h.start_search(0)
