import os

import numpy as np
import torch
from tqdm import tqdm

from analyze.test_data_factory import get_parameter, get_config_row, get_all_value_inverse, get_args
from data_provider.data_factory import data_provider
from models.quantile_function.lstm_cq import sample_pred

from utils.tools import set_times_new_roman_font, draw_density_figure

set_times_new_roman_font()

out_dir = 'probabilistic_density_figure'


def draw_probabilistic_density_figure(exp_name, samples_index, sample_times, _lambda, algorithm_type, select_data=None,
                                      draw_all=True, folder=None, replace_regex=None):
    if replace_regex is None:
        replace_regex = []

    # get data and config
    _, true_value_inverse, _, _ = get_all_value_inverse(exp_name)  # [96, 5165], [96, 5165], [96, 5165], [96, 5165]
    lambda_param, gamma_param, eta_k_param = get_parameter(exp_name)  # [96, 5165, 1], [96, 5165, 20], [96, 5165, 20]

    config_row = get_config_row(exp_name)
    enc_in = config_row['enc_in']
    pred_length = config_row['pred_len']
    num_spline = config_row['num_spline']
    samples_number = len(samples_index)
    data_length = true_value_inverse.shape[1]

    # start tensor operation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init output tensor
    samples_value_candidate = torch.zeros(sample_times, samples_number * data_length).to(device)

    # get input tensor
    gamma_tensor = torch.Tensor(gamma_param).to(device)  # [96, 5165, *]
    eta_k_tensor = torch.Tensor(eta_k_param).to(device)  # [96, 5165, *]

    # filter sample index
    gamma_tensor = gamma_tensor[samples_index, :, :]  # [4, 5165, *]
    eta_k_tensor = eta_k_tensor[samples_index, :, :]  # [4, 5165, *]

    # combine sample number and data length
    gamma_tensor = gamma_tensor.reshape(samples_number * data_length, -1)  # [4 * 5165, *]
    eta_k_tensor = eta_k_tensor.reshape(samples_number * data_length, -1)  # [4 * 5165, *]

    # init alpha prime k
    y = torch.ones(num_spline) / num_spline
    alpha_prime_k = y.repeat(samples_number * data_length, 1).to(device)  # [4 * 5165, 20]

    # get samples
    for i in range(sample_times):
        # get pred alpha
        uniform = torch.distributions.uniform.Uniform(
            torch.tensor([0.0], device=device),
            torch.tensor([1.0], device=device))
        pred_alpha = uniform.sample(torch.Size([samples_number * data_length]))  # [4 * 5165, 1]

        # [256] = [256, 20], [256, 1], -0.001, [256, 20], [256, 20], [256, 20], '1+2'
        pred = sample_pred(alpha_prime_k, pred_alpha, _lambda, gamma_tensor, eta_k_tensor, algorithm_type)
        samples_value_candidate[i, :] = pred
    samples_value = samples_value_candidate.reshape(sample_times, samples_number, data_length)  # [99, 4, 5165]

    # move to cpu and covert to numpy for plotting
    samples_value = samples_value.detach().cpu().numpy()  # [99, 4, 15616]

    # integrate different probability range data
    samples_value = samples_value.reshape(-1)  # [99 * 4 * 15616]

    # convert to shape: (sample, feature) for inverse transform
    new_shape = (sample_times * samples_number * data_length, enc_in)
    _ = np.zeros(new_shape)
    _[:, -1] = samples_value
    samples_value = _

    # perform inverse transform
    data_set, _, _, _ = data_provider(get_args(exp_name), data_flag='test', enter_flag='test', new_indexes=None,
                                      cache_data=False)
    samples_value = data_set.inverse_transform(samples_value)

    # get the original data
    samples_value = samples_value[:, -1].squeeze()  # [99 * 4 * 15616]

    # restore different probability range data
    samples_value = samples_value.reshape(sample_times, samples_number, data_length)  # [99, 4, 15616]

    # draw selected figures
    print('drawing selected probabilistic density figures')
    for k in select_data:
        i = k[0]
        j = k[1] - 1

        _path = os.path.join(out_dir, f'step {samples_index[i] + 1}')
        if not os.path.exists(_path):
            os.makedirs(_path)

        file_name = f'PDF {exp_name} Pred {pred_length} Step {samples_index[i] + 1} Data {j + 1}.png'
        for regex in replace_regex:
            file_name = file_name.replace(regex[0], regex[1])

        if folder is not None:
            if not os.path.exists(os.path.join(_path, folder)):
                os.makedirs(os.path.join(_path, folder))
            file_path = os.path.join(_path, folder, file_name)
        else:
            file_path = os.path.join(_path, file_name)

        draw_density_figure(samples=samples_value[:, i, j],
                            true=true_value_inverse[i, j],
                            path=file_path)

    # draw figures
    if draw_all:
        print('drawing all probabilistic density figures')
        for i in range(samples_number):
            _path = os.path.join(out_dir, f'step {samples_index[i] + 1}')
            if not os.path.exists(_path):
                os.makedirs(_path)

            for j in tqdm(range(data_length), desc=f'step {samples_index[i] + 1}'):
                file_name = f'PDF {exp_name} Pred {pred_length} Step {samples_index[i] + 1} Data {j + 1}.png'
                for regex in replace_regex:
                    file_name = file_name.replace(regex[0], regex[1])

                if folder is not None:
                    if not os.path.exists(os.path.join(_path, folder)):
                        os.makedirs(os.path.join(_path, folder))
                    file_path = os.path.join(_path, folder, file_name)
                else:
                    file_path = os.path.join(_path, file_name)

                draw_density_figure(samples=samples_value[:, i, j],
                                    true=true_value_inverse[i, j],
                                    path=file_path)


draw_probabilistic_density_figure(exp_name='LSTM-AQ_Electricity_96',
                                  samples_index=[15, 31, 63, 95],
                                  sample_times=99,
                                  _lambda=-0.001,
                                  algorithm_type='1+2',
                                  select_data=[[0, 97],
                                               [1, 91],
                                               [2, 181],
                                               [3, 235]],
                                  draw_all=False,
                                  folder='AL-QSQF',
                                  replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])

draw_probabilistic_density_figure(exp_name='QSQF-C_Electricity_96',
                                  samples_index=[15, 31, 63, 95],
                                  sample_times=99,
                                  _lambda=-0.001,
                                  algorithm_type='2',
                                  select_data=[[0, 97],
                                               [1, 91],
                                               [2, 181],
                                               [3, 235]],
                                  draw_all=False,
                                  folder='QSQF-C',
                                  replace_regex=[['QSQF-C_Electricity_96', 'QSQF-C Electricity']])
