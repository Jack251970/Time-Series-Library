import os

import matplotlib.pyplot as plt
import numpy as np

pred_len = 96

root_path = '../results/'

# Electricity 96
lstm_aq_ele_96 = 'probability_forecast_OT_96_96_LSTM-ED-CQ_custom_ftMS_sl96_ll0_pl96_dm24_nh4_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-23 17-32-45'
qsqf_c_ele_96 = 'probability_forecast_OT_96_96_QSQF-C_custom_ftMS_sl96_ll0_pl96_dm512_nh8_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-22 23-30-41'


def read_data(path):
    _path = root_path + path + '/prob_metrics.npy'
    if not os.path.exists(_path):
        return None
    data = np.load(_path, allow_pickle=True)
    return data


x_data = range(1, pred_len + 1, 1)
y_data1 = read_data(lstm_aq_ele_96)
y_data2 = read_data(qsqf_c_ele_96)


def phase_data(data):
    if data is None:
        return None, None
    global pred_len
    return data[1:pred_len + 1], data[-pred_len-1:-1]


y_crps1, y_pinaw1 = phase_data(y_data1)
y_crps2, y_pinaw2 = phase_data(y_data2)

plt.clf()
if y_crps1 is not None:
    plt.plot(x_data, y_crps1, 'b', alpha=0.5, linewidth=1, label='LSTM-AQ')
if y_crps2 is not None:
    plt.plot(x_data, y_crps2, 'r', alpha=0.5, linewidth=1, label='QSQF-C')

plt.legend()
plt.xlabel('Prediction Step')
plt.ylabel('CRPS')
plt.show()
# plt.savefig('CRPS.png')

plt.clf()
if y_pinaw1 is not None:
    plt.plot(x_data, y_pinaw1, 'b', alpha=0.5, linewidth=1, label='LSTM-AQ')
if y_pinaw2 is not None:
    plt.plot(x_data, y_pinaw2, 'r', alpha=0.5, linewidth=1, label='QSQF-C')

plt.legend()
plt.xlabel('Prediction Step')
plt.ylabel('PINAW')
plt.show()
# plt.savefig('PINAW.png')
