import os

import torch
from tqdm import tqdm

from analyze.test_data_factory import get_parameter, get_config_row, get_all_value_inverse, get_args
from data_provider.data_factory import data_provider
from models.quantile_function.lstm_cq import sample_pred

from utils.tools import set_times_new_roman_font, draw_density_figure

set_times_new_roman_font()

out_dir = 'probabilistic_density_figure'


def draw_probabilistic_density_figure(exp_name, samples_index, sample_times, _lambda, algorithm_type, select_data=None,
                                      draw_all=True, folder=None, replace_regex=None, use_cupy=False):
    # check cuda
    if use_cupy:
        import cupy as cp
        use_cupy = cp.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if replace_regex is None:
        replace_regex = []

    # get data
    _, true_value_inverse, _, _ = get_all_value_inverse(exp_name, use_cupy=use_cupy)  # [96, 5165]
    lambda_param, gamma_param, eta_k_param = get_parameter(exp_name, use_cupy=use_cupy)  # [96, 5165, *]

    # get config
    config_row = get_config_row(exp_name)
    enc_in = config_row['enc_in']
    pred_length = config_row['pred_len']
    num_spline = config_row['num_spline']
    samples_number = len(samples_index)
    data_length = true_value_inverse.shape[1]

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
    for i in tqdm(range(sample_times), desc='sampling'):
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
    if not use_cupy:
        samples_value = samples_value.detach().cpu().numpy()  # [99, 4, 5165]
    else:
        samples_value = cp.from_dlpack(samples_value)  # [99, 4, 5165]

    # integrate different probability range data
    samples_value = samples_value.reshape(-1)  # [99 * 4 * 5165]

    # perform inverse transform
    data_set, _, _, _ = data_provider(get_args(exp_name), data_flag='test', enter_flag='test', new_indexes=None,
                                      cache_data=False)
    min_max_scaler = data_set.get_scaler()

    features_range = min_max_scaler.feature_range
    copy = min_max_scaler.copy
    clip = min_max_scaler.clip
    scale_ = min_max_scaler.scale_[-1]
    min_ = min_max_scaler.min_[-1]
    data_min_ = min_max_scaler.data_min_
    data_max_ = min_max_scaler.data_max_
    data_range_ = min_max_scaler.data_range_
    n_features_in_ = min_max_scaler.n_features_in_
    n_samples_seen_ = min_max_scaler.n_samples_seen_

    # convert to cupy if using cuda
    if use_cupy:
        scale_ = cp.array(scale_)
        min_ = cp.array(min_)

    samples_value_inverse = (samples_value - min_) / scale_

    # restore different probability range data
    samples_value_inverse = samples_value_inverse.reshape(sample_times, samples_number, data_length)  # [99, 4, 5165]

    # convert to numpy for plotting
    if use_cupy:
        samples_value_inverse = cp.asnumpy(samples_value_inverse)
        true_value_inverse = cp.asnumpy(true_value_inverse)

    # draw selected figures
    if select_data is not None:
        for k in select_data:
            i = k[0]
            j = k[1] - 1
            xlim = k[2] if len(k) >= 3 else None
            ylim = k[3] if len(k) >= 4 else None

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

            draw_density_figure(samples=samples_value_inverse[:, i, j],
                                true=true_value_inverse[i, j],
                                path=file_path,
                                xlim=xlim,
                                ylim=ylim)

    # draw figures
    if draw_all:
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

                draw_density_figure(samples=samples_value_inverse[:, i, j],
                                    true=true_value_inverse[i, j],
                                    path=file_path)


draw_probabilistic_density_figure(exp_name='LSTM-AQ_Electricity_96',
                                  samples_index=[15, 31, 63, 95],
                                  sample_times=500,
                                  _lambda=-0.001,
                                  algorithm_type='1+2',
                                  select_data=[[0, 97, [-500, 4000], [0, 0.00250]],
                                               [1, 91, [-500, 4500], [0, 0.00225]],
                                               [2, 181, [-500, 5000], [0, 0.00200]],
                                               [3, 235, [-500, 5000], [0, 0.00225]]],
                                  draw_all=False,
                                  folder='AL-QSQF',
                                  replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']],
                                  use_cupy=True)

draw_probabilistic_density_figure(exp_name='QSQF-C_Electricity_96',
                                  samples_index=[15, 31, 63, 95],
                                  sample_times=500,
                                  _lambda=-0.001,
                                  algorithm_type='2',
                                  select_data=[[0, 97, [-500, 4000], [0, 0.00250]],
                                               [1, 91, [-500, 4500], [0, 0.00225]],
                                               [2, 181, [-500, 5000], [0, 0.00200]],
                                               [3, 235, [-500, 5000], [0, 0.00225]]],
                                  draw_all=False,
                                  folder='QSQF-C',
                                  replace_regex=[['QSQF-C_Electricity_96', 'QSQF-C Electricity']],
                                  use_cupy=True)
