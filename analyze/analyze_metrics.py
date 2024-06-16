import os

import matplotlib.pyplot as plt

from analyze.test_data_factory import get_prob_metrics, get_config_row

from utils.tools import set_times_new_roman_font

set_times_new_roman_font()

plt.rcParams['figure.figsize'] = (12.8, 7.2)

output_dir = 'metrics'


def draw_steps_figure(config_list, fig_name):
    if len(config_list) == 2:
        lstm_aq_exp_name = config_list[0][0]
        crps_lstm_aq, crps_steps_lstm_aq, mre_lstm_aq, pinaw_lstm_aq, pinaw_steps_lstm_aq = get_prob_metrics(
            lstm_aq_exp_name)

        qsqf_c_exp_name = config_list[1][0]
        crps_qsqf_c, crps_steps_qsqf_c, mre_qsqf_c, pinaw_qsqf_c, pinaw_steps_qsqf_c = get_prob_metrics(qsqf_c_exp_name)

        config_row_lstm_aq = get_config_row(lstm_aq_exp_name)
        pred_len = config_row_lstm_aq['pred_len']

        x_data = range(1, pred_len, 1)

        plt.clf()
        plt.plot(x_data, crps_steps_lstm_aq[1:], 'bo-', alpha=0.5, linewidth=1, label=config_list[0][1])
        plt.plot(x_data, crps_steps_qsqf_c[1:], 'ro-', alpha=0.5, linewidth=1, label=config_list[1][1])

        plt.legend()
        plt.xlabel('Prediction Step')
        plt.ylabel('CRPS')
        plt.savefig(os.path.join(output_dir, f'CRPS {fig_name}.png'))

        plt.clf()
        plt.plot(x_data, pinaw_steps_lstm_aq[1:], 'bo-', alpha=0.5, linewidth=1, label=config_list[0][1])
        plt.plot(x_data, pinaw_steps_qsqf_c[1:], 'ro-', alpha=0.5, linewidth=1, label=config_list[1][1])

        plt.legend()
        plt.xlabel('Prediction Step')
        plt.ylabel('PINAW')
        plt.savefig(os.path.join(output_dir, f'PINAW {fig_name}.png'))
    elif len(config_list) == 1:
        exp_name = config_list[0][0]
        crps, crps_steps, mre, pinaw, pinaw_steps = get_prob_metrics(exp_name)

        config_row = get_config_row(exp_name)
        pred_len = int(config_row['pred_len'])

        x_data = range(1, pred_len, 1)

        plt.clf()
        plt.plot(x_data, crps_steps[1:], 'bo-', alpha=0.5, linewidth=1, label=config_list[0][1])

        plt.legend()
        plt.xlabel('Prediction Step')
        plt.ylabel('CRPS')
        plt.savefig(os.path.join(output_dir, f'CRPS {fig_name}.png'))

        plt.clf()
        plt.plot(x_data, pinaw_steps[1:], 'bo-', alpha=0.5, linewidth=1, label=config_list[0][1])

        plt.legend()
        plt.xlabel('Prediction Step')
        plt.ylabel('PINAW')
        plt.savefig(os.path.join(output_dir, f'PINAW {fig_name}.png'))
    else:
        raise ValueError('config list length should be 1 or 2')


draw_steps_figure(config_list=[('LSTM-AQ_Electricity_96', 'AL-QSQF with attention'),
                               ('LSTM-AQ1_Electricity_96', 'AL-QSQF without attention ')],
                  fig_name='AL-QSQF Attention Electricity Step 96')

draw_steps_figure(config_list=[('QSQF-C1_Electricity_96', 'QSQF-C with attention'),
                               ('QSQF-C_Electricity_96', 'QSQF-C without attention ')],
                  fig_name='QSQF-C Attention Electricity Step 96')
