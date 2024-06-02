import os

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
    # QSQF-C
    'QSQF-C_Electricity_16': '2024-04-22 11-26-26',
    'QSQF-C_Electricity_32': '2024-04-22 22-17-10',
    'QSQF-C_Electricity_64': '2024-05-07 14-32-20',
    'QSQF-C_Electricity_96': '2024-04-22 23-30-41',
    'QSQF-C_Exchange_16': '2024-05-07 21-31-39',
    'QSQF-C_Exchange_32': '2024-05-07 21-45-02',
    'QSQF-C_Exchange_64': '2024-05-07 21-52-48',
    'QSQF-C_Exchange_96': '2024-05-07 23-42-11',
}
_exp_dict = {}


def build_time_list():
    _time_list = []
    for exp_name, exp_time in _exp_time_dict.items():
        _time_list.append(exp_time)
    return _time_list


def build_exp_dict():
    exp_names = os.listdir(process_folder)
    for exp_name, exp_time in _exp_time_dict.items():
        for exp_name_ in exp_names:
            if exp_name_[-len(exp_time):] == exp_time:
                _exp_time_dict[exp_name] = exp_name_


def get_exp_settings(exp_name):
    if _exp_dict == {}:
        build_exp_dict()
    return _exp_dict[exp_name]


def get_attention_map_path(exp_name):
    _exp_path = get_exp_settings(exp_name)
    return os.path.join(prob_results_folder, _exp_path, 'attention_maps.npy')


def get_all_value_inverse_path(exp_name):
    _exp_path = get_exp_settings(exp_name)
    return os.path.join(prob_results_folder, _exp_path, 'pred_value_inverse.npy'), \
        os.path.join(prob_results_folder, _exp_path, 'true_value_inverse.npy'), \
        os.path.join(prob_results_folder, _exp_path, 'high_value_inverse.npy'), \
        os.path.join(prob_results_folder, _exp_path, 'low_value_inverse.npy')


def get_loss_path(exp_name):
    _exp_path = get_exp_settings(exp_name)
    files = ['train_loss.npy', 'vali_loss.npy', 'test_loss.npy']
    return [os.path.join(process_folder, _exp_path, file) for file in files]
