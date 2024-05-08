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

# config experiment
_exp_dict = {
    'LSTM-AQ_Electricity_96': 'probability_forecast_OT_96_96_LSTM-ED'
                              '-CQ_custom_ftMS_sl96_ll0_pl96_dm24_nh4_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-23 17-32-45',
    'LSTM-AQ_Exchange_96': 'probability_forecast_OT_96_96_LSTM-ED'
                           '-CQ_custom_ftMS_sl96_ll0_pl96_dm64_nh2_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-24 17-16-19',
}


def get_exp_settings(exp_name):
    return _exp_dict[exp_name]


def get_attention_map_path(exp_name):
    _exp_path = get_exp_settings(exp_name)
    return os.path.join(prob_results_folder, _exp_path, 'attention_maps.npy')
