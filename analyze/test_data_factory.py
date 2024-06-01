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
    # LSTM-ED-CQ
    'LSTM-AQ_Electricity_16': 'probability_forecast_OT_96_16_LSTM-ED'
                              '-CQ_custom_ftMS_sl96_ll0_pl16_dm24_nh1_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-23 10-33-28',
    'LSTM-AQ_Electricity_32': 'probability_forecast_OT_96_32_LSTM-ED'
                              '-CQ_custom_ftMS_sl96_ll0_pl32_dm24_nh2_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-05-06 23-47-33',
    'LSTM-AQ_Electricity_64': 'probability_forecast_OT_96_64_LSTM-ED'
                              '-CQ_custom_ftMS_sl96_ll0_pl64_dm40_nh2_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-05-07 02-02-08',
    'LSTM-AQ_Electricity_96': 'probability_forecast_OT_96_96_LSTM-ED'
                              '-CQ_custom_ftMS_sl96_ll0_pl96_dm24_nh4_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-23 17-32-45',
    'LSTM-AQ_Exchange_16': 'probability_forecast_OT_96_16_LSTM-ED'
                           '-CQ_custom_ftMS_sl96_ll0_pl16_dm64_nh2_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-28 05-04-11',
    'LSTM-AQ_Exchange_32': 'probability_forecast_OT_96_32_LSTM-ED'
                           '-CQ_custom_ftMS_sl96_ll0_pl32_dm40_nh1_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-24 11-34-20',
    'LSTM-AQ_Exchange_64': 'probability_forecast_OT_96_64_LSTM-ED'
                           '-CQ_custom_ftMS_sl96_ll0_pl64_dm64_nh2_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-05-07 05-34-41',
    'LSTM-AQ_Exchange_96': 'probability_forecast_OT_96_96_LSTM-ED'
                           '-CQ_custom_ftMS_sl96_ll0_pl96_dm64_nh2_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-24 17-16-19',
    # QSQF-C
    'QSQF-C_Electricity_96': 'probability_forecast_OT_96_96_QSQF'
                             '-C_custom_ftMS_sl96_ll0_pl96_dm512_nh8_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-22 23-30-41',
    'QSQF-C_Exchange_96': 'probability_forecast_OT_96_96_QSQF'
                          '-C_custom_ftMS_sl96_ll0_pl96_dm512_nh8_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-05-07 23-42-11',
}


def get_exp_settings(exp_name):
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
