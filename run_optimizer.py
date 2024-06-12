from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer


def check_jump_experiment(_parameter):
    return False


def link_fieldnames_data(_config):
    data_path = _config['data_path']
    if data_path == 'electricity/electricity.csv':
        # electricity dataset
        _config['enc_in'] = 321
        _config['dec_in'] = 321
        _config['c_out'] = 321
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
    elif data_path == 'weather/weather.csv':
        # weather dataset
        _config['enc_in'] = 21
        _config['dec_in'] = 21
        _config['c_out'] = 21
    return _config


def get_custom_test_time(_args):
    return None


def get_model_id_tags(_args):
    tags = []
    return tags


def get_search_space():
    default_config = {
        'task_name': {'_type': 'single', '_value': 'long_term_forecast'},
        'is_training': {'_type': 'single', '_value': 1},
        'des': {'_type': 'single', '_value': 'Exp'},
        'batch_size': {'_type': 'single', '_value': 256},
    }

    dataset_config = {
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'MS'},
        'root_path': {'_type': 'single', '_value': './dataset/'},
        'data_path': {'_type': 'choice', '_value': ['electricity/electricity.csv', 'ETT-small/ETTh1.csv',
                                                    'ETT-small/ETTh2.csv', 'ETT-small/ETTm1.csv', 'ETT-small/ETTm2.csv',
                                                    'exchange_rate/exchange_rate.csv', 'illness/national_illness.csv',
                                                    'traffic/traffic.csv', 'weather/weather.csv']},
    }

    learning_config = {
        'learning_rate': {'_type': 'single', '_value': 0.0001},
        'train_epochs': {'_type': 'single', '_value': 10},
    }

    period_config = {
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 96},
        'pred_len': {'_type': 'single', '_value': 96},
    }

    transformer_config = {
        'e_layers': {'_type': 'single', '_value': 1},
        'd_layers': {'_type': 'single', '_value': 1},
    }

    model_configs = {
        'Transformer': transformer_config
    }

    return [default_config, dataset_config, learning_config, period_config], model_configs


h = HyperParameterOptimizer(script_mode=False,
                            models=['Transformer'],
                            get_search_space=get_search_space,
                            prepare_config=None,
                            build_setting=None,
                            build_config_dict=None,
                            set_args=None,
                            get_fieldnames=None,
                            get_model_id_tags=get_model_id_tags,
                            check_jump_experiment=check_jump_experiment,
                            link_fieldnames_data=link_fieldnames_data,
                            get_custom_test_time=get_custom_test_time)
# Uncomment the following line to output the script
# h.output_script('Electricity')
h.config_optimizer_settings(root_path='.',
                            data_dir='data',
                            jump_csv_file='jump_data.csv',
                            data_csv_file='data.csv',
                            data_csv_file_format='data_{}.csv',
                            scan_all_csv=True,
                            process_number=1,
                            save_process=True,
                            try_model=False,
                            force_exp=False,
                            add_tags=[])

if __name__ == "__main__":
    h.start_search(process_index=0, inverse_exp=False, shutdown_after_done=False)
