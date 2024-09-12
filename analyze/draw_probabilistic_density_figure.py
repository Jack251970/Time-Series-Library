import os

import torch
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from analyze.test_data_factory import get_parameter, get_config_row, get_all_value_inverse, get_args
from data_provider.data_factory import data_provider
from models.quantile_function.lstm_cq import sample_pred

from utils.tools import set_times_new_roman_font, draw_density_figure

set_times_new_roman_font()

out_dir = 'probabilistic_density_figure'


def draw_comp_density_figure(model_names, samples1, true1, pred1, samples2, true2, pred2, path,
                             xlabel=None, ylabel=None, font_size=19, draw_pred=False):
    model_name1, model_name2 = model_names
    true_color = 'green'
    pred1_color = 'blue'
    pred2_color = 'red'
    plt.clf()
    plt.figure(figsize=(7.8, 5.25))  # Adjust the figure size to increase resolution
    plt.axvline(true1.squeeze(), color=true_color, linestyle='--', label='True Value')
    if draw_pred:
        plt.axvline(pred1.squeeze(), color=pred1_color, linestyle='--', label=f'{model_name1} Predicted Value')
    sns.kdeplot(samples1.squeeze(), fill=True, label=f'{model_name1} Probability Density', alpha=0.5)
    if draw_pred:
        plt.axvline(pred2.squeeze(), color=pred2_color, linestyle='--', label=f'{model_name2} Predicted Value')
    sns.kdeplot(samples2.squeeze(), fill=True, label=f'{model_name2} Probability Density', alpha=0.5)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()  # Adjust the layout to prevent ylabel from being cut off
    plt.savefig(path.replace('.png', '.svg'), format='svg')


def _sample(exp_name, samples_index, sample_times, _lambda, algorithm_type, use_cupy=False):
    # check cuda
    if use_cupy:
        import cupy as cp
        use_cupy = cp.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    return pred_length, samples_number, data_length, samples_value_inverse, true_value_inverse


def draw_probabilistic_density_figure(exp_name, samples_index, sample_times, _lambda, algorithm_type, select_data=None,
                                      draw_all=True, max_data_length=0, folder=None, replace_regex=None, use_cupy=False,
                                      xlabel=None):
    if replace_regex is None:
        replace_regex = []

    pred_length, samples_number, data_length, samples_value_inverse, true_value_inverse = (
        _sample(exp_name, samples_index, sample_times, _lambda, algorithm_type, use_cupy))

    # draw selected figures
    if select_data is not None:
        for k in select_data:
            i = k[0]
            j = k[1] - 1
            xlim = k[2] if len(k) >= 3 else None
            ylim = k[3] if len(k) >= 4 else None

            _path = out_dir
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
                                xlabel=xlabel,
                                ylabel='Probability',
                                xlim=xlim,
                                ylim=ylim)

    # draw all figures
    if draw_all:
        for i in range(samples_number):
            _path = os.path.join(out_dir, f'step {samples_index[i] + 1}')
            if not os.path.exists(_path):
                os.makedirs(_path)

            for j in tqdm(range(0, max(data_length, max_data_length) if max_data_length > 0 else data_length),
                          desc=f'step {samples_index[i] + 1}'):
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
                                    xlabel=xlabel,
                                    ylabel='Probability')


def draw_comp_probabilistic_density_figure(exp_name1, exp_name2, comp_tag, samples_index, sample_times, _lambda1,
                                           algorithm_type1, _lambda2, algorithm_type2, select_data=None, max_data_length=0,
                                           folder=None, xlabel=None, replace_regex=None, use_cupy=False):
    if replace_regex is None:
        replace_regex = []

    pred_length1, samples_number1, data_length1, samples_value_inverse1, true_value_inverse1 = (
        _sample(exp_name1, samples_index, sample_times, _lambda1, algorithm_type1, use_cupy))

    pred_value_inverse1, _, _, _ = get_all_value_inverse(exp_name1)

    pred_length2, samples_number2, data_length2, samples_value_inverse2, true_value_inverse2 = (
        _sample(exp_name2, samples_index, sample_times, _lambda2, algorithm_type2, use_cupy))

    pred_value_inverse2, _, _, _ = get_all_value_inverse(exp_name2)

    # draw selected figures
    if select_data is not None:
        for k in select_data:
            i = k[0]
            j = k[1] - 1 if len(k) >= 2 else -1

            _path = out_dir
            if not os.path.exists(_path):
                os.makedirs(_path)

            file_name = f'PDF Comp {comp_tag} Pred {pred_length1} Step {samples_index[i] + 1} Data {j + 1}.png'
            for regex in replace_regex:
                file_name = file_name.replace(regex[0], regex[1])

            if folder is not None:
                if not os.path.exists(os.path.join(_path, folder)):
                    os.makedirs(os.path.join(_path, folder))
                file_path = os.path.join(_path, folder, file_name)
            else:
                file_path = os.path.join(_path, file_name)

            if j == -1:
                for t in tqdm(range(0, max_data_length if max_data_length > 0 else 1),
                              desc=f'step {samples_index[i] + 1}'):
                    draw_comp_density_figure(model_names=['AL-QSQF', 'QSQF-C'],
                                             samples1=samples_value_inverse1[:, i, t],
                                             true1=true_value_inverse1[i, t],
                                             pred1=pred_value_inverse1[i, t],
                                             samples2=samples_value_inverse2[:, i, t],
                                             true2=true_value_inverse2[i, t],
                                             pred2=pred_value_inverse2[i, t],
                                             path=file_path.replace('Data 0.png', f'Data {t + 1}.png'),
                                             xlabel=xlabel,
                                             ylabel='Probability',
                                             draw_pred=False)
            else:
                draw_comp_density_figure(model_names=['AL-QSQF', 'QSQF-C'],
                                         samples1=samples_value_inverse1[:, i, j],
                                         true1=true_value_inverse1[i, j],
                                         pred1=pred_value_inverse1[i, j],
                                         samples2=samples_value_inverse2[:, i, j],
                                         true2=true_value_inverse2[i, j],
                                         pred2=pred_value_inverse2[i, j],
                                         path=file_path,
                                         xlabel=xlabel,
                                         ylabel='Probability',
                                         draw_pred=False)


# # Electricity
# # AL-QSQF
# draw_probabilistic_density_figure(exp_name='LSTM-AQ_Electricity_96',
#                                   samples_index=[15, 31, 63, 95],
#                                   sample_times=500,
#                                   _lambda=-0.001,
#                                   algorithm_type='1+2',
#                                   select_data=[[0, 97, [-500, 4000], [0, 0.00250]],
#                                                [1, 91, [-500, 4500], [0, 0.00225]],
#                                                [2, 181, [-500, 5000], [0, 0.00200]],
#                                                [3, 235, [-500, 5000], [0, 0.00225]]],
#                                   draw_all=False,
#                                   xlabel='Power/KW',
#                                   folder=None,
#                                   replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']],
#                                   use_cupy=False)
#
# # QSQF-C
# draw_probabilistic_density_figure(exp_name='QSQF-C_Electricity_96',
#                                   samples_index=[15, 31, 63, 95],
#                                   sample_times=500,
#                                   _lambda=-0.001,
#                                   algorithm_type='2',
#                                   select_data=[[0, 97, [-500, 4000], [0, 0.00250]],
#                                                [1, 91, [-500, 4500], [0, 0.00225]],
#                                                [2, 181, [-500, 5000], [0, 0.00200]],
#                                                [3, 235, [-500, 5000], [0, 0.00225]]],
#                                   draw_all=False,
#                                   xlabel='Power/KW',
#                                   folder=None,
#                                   replace_regex=[['QSQF-C_Electricity_96', 'QSQF-C Electricity']],
#                                   use_cupy=False)
#
# # AL-QSQF & QSQF-C
draw_comp_probabilistic_density_figure(exp_name1='LSTM-AQ(HLF)_Electricity_96',
                                       exp_name2='QSQF-C_Electricity_96',
                                       comp_tag='Electricity',
                                       samples_index=[15, 31, 63, 95],
                                       sample_times=500,
                                       _lambda1=-0.001,
                                       algorithm_type1='1+2',
                                       _lambda2=-0.001,
                                       algorithm_type2='2',
                                       select_data=[[0, 97],
                                                    [1, 91],
                                                    [2, 181],
                                                    [3, 235]],
                                       xlabel='Power/KW',
                                       folder=None,
                                       replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity'],
                                                      ['QSQF-C_Electricity_96', 'QSQF-C Electricity']],
                                       use_cupy=False)

# # Traffic
# AL-QSQF
# draw_probabilistic_density_figure(exp_name='LSTM-AQ_Traffic_96',
#                                   samples_index=[15, 31, 63, 95],
#                                   sample_times=500,
#                                   _lambda=-0.001,
#                                   algorithm_type='1+2',
#                                   draw_all=True,
#                                   max_data_length=200,
#                                   folder=None,
#                                   xlabel='Road occupancy',
#                                   replace_regex=[['LSTM-AQ_Traffic_96', 'AL-QSQF Traffic']],
#                                   use_cupy=False)
#
# # QSQF-C
# draw_probabilistic_density_figure(exp_name='QSQF-C_Traffic_96',
#                                   samples_index=[15, 31, 63, 95],
#                                   sample_times=500,
#                                   _lambda=-0.001,
#                                   algorithm_type='2',
#                                   draw_all=True,
#                                   max_data_length=200,
#                                   folder=None,
#                                   xlabel='Road occupancy',
#                                   replace_regex=[['QSQF-C_Traffic_96', 'QSQF-C Traffic']],
#                                   use_cupy=False)
#
# # AL-QSQF & QSQF-C
# draw_comp_probabilistic_density_figure(exp_name1='LSTM-AQ(HLF)_Traffic_96',
#                                        exp_name2='QSQF-C_Traffic_96',
#                                        comp_tag='Traffic',
#                                        samples_index=[15, 31, 63, 95],
#                                        sample_times=500,
#                                        _lambda1=-0.001,
#                                        algorithm_type1='1+2',
#                                        _lambda2=-0.001,
#                                        algorithm_type2='2',
#                                        select_data=[[0],
#                                                     [1],
#                                                     [2],
#                                                     [3]],
#                                        max_data_length=200,
#                                        folder=None,
#                                        xlabel='Road occupancy',
#                                        replace_regex=[['LSTM-AQ_Traffic_96', 'AL-QSQF Traffic'],
#                                                       ['QSQF-C_Traffic_96', 'QSQF-C Traffic']],
#                                        use_cupy=False)
#
draw_comp_probabilistic_density_figure(exp_name1='LSTM-AQ(HLF)_Traffic_96',
                                       exp_name2='QSQF-C_Traffic_96',
                                       comp_tag='Traffic',
                                       samples_index=[15, 31, 63, 95],
                                       sample_times=500,
                                       _lambda1=-0.001,
                                       algorithm_type1='1+2',
                                       _lambda2=-0.001,
                                       algorithm_type2='2',
                                       select_data=[[0, 10],
                                                    [1, 133],
                                                    [2, 46],
                                                    [3, 157]],
                                       xlabel='Road occupancy',
                                       folder=None,
                                       replace_regex=[['LSTM-AQ_Traffic_96', 'AL-QSQF Traffic'],
                                                      ['QSQF-C_Traffic_96', 'QSQF-C Traffic']],
                                       use_cupy=False)
