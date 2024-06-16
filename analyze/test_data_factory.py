import csv
import os

import numpy as np

from exp.exp_basic import Exp_Basic
from hyper_parameter_optimizer import basic_settings

root_path = '..'
data_dir = 'data'

# build basic experiment
exp_basic = Exp_Basic(root_path=root_path, args=None, try_model=True, save_process=True, initialize_later=True)

# get all root folders
checkpoints_folder = exp_basic.root_checkpoints_path
process_folder = exp_basic.root_process_path
results_folder = exp_basic.root_results_path
test_results_folder = exp_basic.root_test_results_path
m4_results_folder = exp_basic.root_m4_results_path
prob_results_folder = exp_basic.root_prob_results_path

# get data folder
data_folder = os.path.join(root_path, data_dir)

# get fieldnames
fieldnames = basic_settings.get_fieldnames('all')

# config test experiments
_exp_time_dict = {
    # LSTM-AQ
    'LSTM-AQ_Electricity_16': '2024-04-23 10-33-28',
    'LSTM-AQ_Electricity_32': '2024-05-06 23-47-33',
    'LSTM-AQ_Electricity_64': '2024-05-07 02-02-08',
    'LSTM-AQ_Electricity_96': '2024-04-23 17-32-45',
    'LSTM-AQ_Exchange_16': '2024-04-28 05-04-11',
    'LSTM-AQ_Exchange_32': '2024-04-24 11-34-20',
    'LSTM-AQ_Exchange_64': '2024-05-07 05-34-41',
    'LSTM-AQ_Exchange_96': '2024-04-24 17-16-19',
    'LSTM-AQ_Traffic_16': '2024-06-02 16-12-42',
    'LSTM-AQ_Traffic_32': '2024-06-02 22-01-15',
    'LSTM-AQ_Traffic_64': '2024-06-02 11-49-53',
    'LSTM-AQ_Traffic_96': '2024-06-02 01-57-16',
    # QSQF-C
    'QSQF-C_Electricity_16': '2024-06-15 21-18-26',
    'QSQF-C_Electricity_32': '2024-06-16 03-13-00',
    'QSQF-C_Electricity_64': '2024-06-16 07-10-04',
    'QSQF-C_Electricity_96': '2024-06-16 10-15-41',
    'QSQF-C_Exchange_16': '2024-06-15 21-12-30',
    'QSQF-C_Exchange_32': '2024-06-15 21-56-56',
    'QSQF-C_Exchange_64': '2024-06-15 22-54-38',
    'QSQF-C_Exchange_96': '2024-06-16 00-57-08',
    'QSQF-C_Traffic_16': '2024-06-15 14-02-11',
    'QSQF-C_Traffic_32': '2024-06-15 17-11-50',
    'QSQF-C_Traffic_64': '2024-06-15 21-24-55',
    'QSQF-C_Traffic_96': '2024-06-16 01-14-15',
    # LSTM-AQ2
    'LSTM-AQ2_Electricity_96': '2024-06-14 15-17-50',
    # LSTM-AQ3
    'LSTM-AQ3_Electricity_96': '2024-06-14 15-17-50',
    # LSTM-AQ4
    'LSTM-AQ4_Electricity_96': '2024-06-15 02-53-21',
}
_exp_dict = {}


def get_exp_time(key):
    return _exp_time_dict[key]


def build_exp_dict():
    global _exp_time_dict
    exp_names = os.listdir(process_folder)
    for exp_name, exp_time in _exp_time_dict.items():
        for _exp_name in exp_names:
            if _exp_name[-len(exp_time):] == exp_time:
                _exp_dict[exp_name] = _exp_name


def get_exp_settings(exp_name):
    global _exp_dict
    if _exp_dict == {}:
        build_exp_dict()
    return _exp_dict[exp_name]


def get_config_row(exp_name):
    _exp_settings = get_exp_settings(exp_name)

    # scan all csv files under data folder
    file_paths = []
    for root, dirs, files in os.walk(str(data_folder)):
        for _file in files:
            if _file.endswith('.csv') and _file not in file_paths:
                _append_path = os.path.join(root, _file)
                file_paths.append(_append_path)

    # find target item
    target_row = None
    for file_path in file_paths:
        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            next(reader)  # skip the header
            for row in reader:
                setting = row['setting']
                if setting == _exp_settings:
                    target_row = row

    return target_row


def get_attention_map(exp_name):
    _exp_path = get_exp_settings(exp_name)
    _path = os.path.join(prob_results_folder, _exp_path, 'attention_maps.npy')
    return np.load(_path)


def get_all_value(exp_name):
    _exp_path = get_exp_settings(exp_name)
    pred_value_path = os.path.join(prob_results_folder, _exp_path, 'pred_value.npy')
    true_value_path = os.path.join(prob_results_folder, _exp_path, 'true_value.npy')
    high_value_path = os.path.join(prob_results_folder, _exp_path, 'high_value.npy')
    low_value_path = os.path.join(prob_results_folder, _exp_path, 'low_value.npy')
    return np.load(pred_value_path), np.load(true_value_path), np.load(high_value_path), np.load(low_value_path)


def get_all_value_inverse(exp_name):
    _exp_path = get_exp_settings(exp_name)
    pred_value_path = os.path.join(prob_results_folder, _exp_path, 'pred_value_inverse.npy')
    true_value_path = os.path.join(prob_results_folder, _exp_path, 'true_value_inverse.npy')
    high_value_path = os.path.join(prob_results_folder, _exp_path, 'high_value_inverse.npy')
    low_value_path = os.path.join(prob_results_folder, _exp_path, 'low_value_inverse.npy')
    return np.load(pred_value_path), np.load(true_value_path), np.load(high_value_path), np.load(low_value_path)


def get_loss(exp_name):
    _exp_path = get_exp_settings(exp_name)
    files = ['train_loss.npy', 'vali_loss.npy', 'test_loss.npy']
    _paths = [os.path.join(process_folder, _exp_path, file) for file in files]

    _train_loss = None
    _vali_loss = None
    _test_loss = None

    for _path in _paths:
        data = np.load(_path)
        if 'train' in _path:
            _train_loss = data
        elif 'vali' in _path:
            _vali_loss = data
        elif 'test' in _path:
            _test_loss = data

    return _train_loss, _vali_loss, _test_loss


def get_prob_metrics(exp_name):
    _exp_path = get_exp_settings(exp_name)
    pred_len = int(get_config_row(exp_name)['pred_len'])
    _path = os.path.join(results_folder, _exp_path, 'prob_metrics.npy')
    metrics_data = np.load(_path)

    crps = metrics_data[0]
    crps_steps = metrics_data[1:pred_len + 1]
    pinaw = metrics_data[pred_len + 1]
    mre = metrics_data[pred_len + 2]
    pinaw_steps = metrics_data[pred_len + 3:]

    return crps, crps_steps, mre, pinaw, pinaw_steps


def get_parameter(exp_name):
    _exp_path = get_exp_settings(exp_name)
    lambda_path = os.path.join(prob_results_folder, _exp_path, 'samples_lambda.npy')
    gamma_path = os.path.join(prob_results_folder, _exp_path, 'samples_gamma.npy')
    eta_k_path = os.path.join(prob_results_folder, _exp_path, 'samples_eta_k.npy')
    return np.load(lambda_path), np.load(gamma_path), np.load(eta_k_path)
