import matplotlib.pyplot as plt

from analyze.test_data_factory import get_prob_metrics, get_config_row

plt.rcParams['figure.figsize'] = (12.8, 7.2)

lstm_aq_exp_name = 'LSTM-AQ_Electricity_96'
crps_lstm_aq, crps_steps_lstm_aq, mre_lstm_aq, pinaw_lstm_aq, pinaw_steps_lstm_aq = get_prob_metrics(lstm_aq_exp_name)

qsqf_c_exp_name = 'QSQF-C_Electricity_96'
crps_qsqf_c, crps_steps_qsqf_c, mre_qsqf_c, pinaw_qsqf_c, pinaw_steps_qsqf_c = get_prob_metrics(qsqf_c_exp_name)

config_row = get_config_row(qsqf_c_exp_name)
pred_len = int(config_row['pred_len'])

x_data = range(1, pred_len, 1)

plt.clf()
if crps_steps_lstm_aq is not None:
    plt.plot(x_data, crps_steps_lstm_aq[1:], 'bo-', alpha=0.5, linewidth=1, label='LSTM-AQ')
if crps_steps_qsqf_c is not None:
    plt.plot(x_data, crps_steps_qsqf_c[1:], 'ro-', alpha=0.5, linewidth=1, label='QSQF-C')

plt.legend()
plt.xlabel('Prediction Step')
plt.ylabel('CRPS')
plt.savefig('CRPS_steps.png')

plt.clf()
if pinaw_steps_lstm_aq is not None:
    plt.plot(x_data, pinaw_steps_lstm_aq[1:], 'bo-', alpha=0.5, linewidth=1, label='LSTM-AQ')
if pinaw_steps_qsqf_c is not None:
    plt.plot(x_data, pinaw_steps_qsqf_c[1:], 'ro-', alpha=0.5, linewidth=1, label='QSQF-C')

plt.legend()
plt.xlabel('Prediction Step')
plt.ylabel('PINAW')
plt.savefig('PINAW_steps.png')
