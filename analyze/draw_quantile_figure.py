import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from analyze.test_data_factory import get_parameter, get_config_row, get_all_value
from models.quantile_function.lstm_cq import sample_pred

from utils.tools import set_times_new_roman_font

set_times_new_roman_font()

samples_index = [15, 31, 63, 95]
folder_path = 'quantile_figure'

select_step = 63
select_data_length = 59

lstm_aq_exp_name = 'LSTM-AQ_Electricity_96'
lambda_lstm_aq, gamma_lstm_aq, eta_k_lstm_aq = get_parameter(lstm_aq_exp_name)  # [96, 5165, 1]

qsqf_c_exp_name = 'QSQF-C_Electricity_96'
lambda_qsqf_c, gamma_qsqf_c, eta_k_qsqf_c = get_parameter(qsqf_c_exp_name)

_, true_value, _, _ = get_all_value(lstm_aq_exp_name)  # [96, 5165]

config_row_lstm_aq = get_config_row(lstm_aq_exp_name)
num_spline = int(config_row_lstm_aq['num_spline'])
pred_len = int(config_row_lstm_aq['pred_len'])
data_length = lambda_lstm_aq.shape[1]

x_data = [i / 100 for i in range(101)]
x_data[0] = 0.0001
x_data[-1] = 0.9999


def get_q_alpha_data(lambda_param, gamma_param, eta_k_param, algorithm_type, _samples_index):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_number = len(_samples_index)
    batch_size = sample_number * data_length

    y = torch.ones(num_spline) / num_spline
    alpha_prime_k = y.repeat(batch_size, 1).to(device)  # [4 * 5165, 1]
    q_alpha_data = torch.Tensor(batch_size, len(x_data)).to(device)  # [4 * 5165, 101]

    lambda_tensor = torch.Tensor(lambda_param).to(device)  # [96, 5165, 1]
    gamma_tensor = torch.Tensor(gamma_param).to(device)  # [96, 5165, 20]
    eta_k_tensor = torch.Tensor(eta_k_param).to(device)  # [96, 5165, 20]

    lambda_tensor = lambda_tensor[_samples_index, :, :]  # [4, 5165, 1]
    gamma_tensor = gamma_tensor[_samples_index, :, :]  # [4, 5165, 20]
    eta_k_tensor = eta_k_tensor[_samples_index, :, :]  # [4, 5165, 20]

    lambda_tensor = lambda_tensor.reshape(batch_size)  # [4 * 5165,]
    gamma_tensor = gamma_tensor.reshape(batch_size, gamma_param.shape[-1])  # [4 * 5165, 20]
    eta_k_tensor = eta_k_tensor.reshape(batch_size, eta_k_param.shape[-1])  # [4 * 5165, 20]

    for i in range(len(x_data)):
        alpha = x_data[i]
        q_alpha_data1 = sample_pred(alpha_prime_k, alpha, lambda_tensor, gamma_tensor, eta_k_tensor, algorithm_type)
        q_alpha_data[:, i] = q_alpha_data1

    return q_alpha_data.detach().cpu().numpy()


# [4 * 5165, 101]
y_data_lstm_aq = get_q_alpha_data(lambda_lstm_aq, gamma_lstm_aq, eta_k_lstm_aq, '1+2', samples_index)
y_data_qsqf_c = get_q_alpha_data(lambda_qsqf_c, gamma_qsqf_c, eta_k_qsqf_c, '2', samples_index)

# [4, 5165, 101]
y_data_lstm_aq = y_data_lstm_aq.reshape(len(samples_index), data_length, len(x_data))
y_data_qsqf_c = y_data_qsqf_c.reshape(len(samples_index), data_length, len(x_data))

# [4, 5165]
y_data_true = true_value[samples_index, :]

# [4, 5165, 101]
y_data_true = y_data_true.reshape(len(samples_index), data_length, 1).repeat(len(x_data), 2)

# draw selected figures
print('drawing selected quantile figure')
i = samples_index.index(select_step)
j = select_data_length
plt.clf()
plt.plot(x_data, y_data_lstm_aq[i, j, :], label='AL-QSQF ', color='blue')  # add blank to create interval
plt.plot(x_data, y_data_qsqf_c[i, j, :], label='QSQF-C', color='red')
plt.plot(x_data, y_data_true[i, j, :], label='True', color='green')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('Q(alpha)')
plt.savefig(os.path.join(folder_path, f'QF Electricity Pred 96 Step {select_step+1} Data {j+1}.png'))

# draw all figures
print('drawing all quantile figures')
for i in range(len(samples_index)):
    step = samples_index[i]
    _path = os.path.join(folder_path, f'step {step + 1}')
    if not os.path.exists(_path):
        os.makedirs(_path)

    for j in tqdm(range(data_length), desc=f'step {step + 1}'):
        plt.clf()
        plt.plot(x_data, y_data_lstm_aq[i, j, :], label='AL-QSQF ', color='blue')
        plt.plot(x_data, y_data_qsqf_c[i, j, :], label='QSQF-C', color='red')
        plt.plot(x_data, y_data_true[i, j, :], label='True', color='green')
        plt.legend()
        plt.xlabel('alpha')
        plt.ylabel('Q(alpha)')
        plt.savefig(os.path.join(_path, f'QF Electricity Pred 96 Step {step+1} Data {j+1}.png'))
