import os

from matplotlib import pyplot as plt

from analyze.test_data_factory import get_all_value_inverse, get_config_row
from utils.tools import draw_figure, set_times_new_roman_font

set_times_new_roman_font()

out_dir = 'probabilistic_figure'


def draw_comp_figure(model_names, x, selected_x, pred1, true1, high1, low1, pred2, true2, high2, low2, pred_range,
                     selected_pred_range, path, xlabel=None, ylabel=None, use_window=False):
    model_name1, model_name2 = model_names
    plt.clf()
    if use_window:
        selected_x = list(selected_x)
    else:
        selected_x = list(x)
    true_color = 'green'
    pred1_color = 'blue'
    pred2_color = 'red'
    plt.plot(selected_x, true1[selected_x].squeeze(), label=f'True Value', color=true_color)
    plt.plot(selected_x, pred1[selected_x].squeeze(), label=f'{model_name1} Predicted Value ', color=pred1_color)
    plt.plot(selected_x, pred2[selected_x].squeeze(), label=f'{model_name2} Predicted Value ', color=pred2_color)

    draw_label = False
    if pred_range is not None:
        for j in range(len(pred_range)):
            confidence_level = pred_range[j]
            if confidence_level in selected_pred_range:
                plt.fill_between(selected_x, high1[j, selected_x].squeeze(), low1[j, selected_x].squeeze(),
                                 alpha=pred_range[j])

                plt.fill_between(selected_x, high2[j, selected_x].squeeze(), low2[j, selected_x].squeeze(),
                                 alpha=pred_range[j])

    # # draw intervals with lines
    # colors = ['orange', 'purple', 'yellow', 'gray', 'pink', 'brown']
    # color_index = 0
    # if pred_range is not None:
    #     for j in range(len(pred_range)):
    #         confidence_level = pred_range[j]
    #         if confidence_level == selected_pred_range:
    #             plt.plot(selected_x, high1[j, selected_x].squeeze(),
    #                      label=f'{model_name1} Confidence Interval ({confidence_level}) ',
    #                      color=colors[color_index], linestyle='--')
    #             plt.plot(selected_x, low1[j, selected_x].squeeze(), color=colors[color_index], linestyle='--')
    #             color_index += 1
    #             plt.plot(selected_x, high2[j, selected_x].squeeze(),
    #                      label=f'{model_name2} Confidence Interval ({confidence_level}) ',
    #                      color=colors[color_index], linestyle='--')
    #             plt.plot(selected_x, low2[j, selected_x].squeeze(), color=colors[color_index], linestyle='--')
    #             color_index += 1

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend(fontsize='small')
    plt.savefig(path)


def draw_probabilistic_figure(exp_name, interval=128, folder=None, selected_data=None, replace_regex=None):
    if replace_regex is None:
        replace_regex = []

    pred_value, true_value, high_value, low_value = get_all_value_inverse(exp_name)

    config_row = get_config_row(exp_name)
    pred_length = config_row['pred_len']
    data_length = pred_value.shape[1]
    probability_range = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    # draw selected figures
    if selected_data is not None:
        for k in selected_data:
            i = k[0] - 1
            j = k[1] - 1
            xlim = k[2] if len(k) >= 3 else None
            ylim = k[3] if len(k) >= 4 else None

            _path = out_dir
            if not os.path.exists(_path):
                os.makedirs(_path)

            # skip the last part of the data
            if j * interval >= data_length:
                continue

            file_name = f'PF {exp_name} Pred {pred_length} Step {i + 1} Data {j + 1}.png'
            for regex in replace_regex:
                file_name = file_name.replace(regex[0], regex[1])

            if folder is not None:
                if not os.path.exists(os.path.join(_path, folder)):
                    os.makedirs(os.path.join(_path, folder))
                file_path = os.path.join(_path, folder, file_name)
            else:
                file_path = os.path.join(_path, file_name)

            draw_figure(range(interval),
                        pred_value[i, j * interval: (j + 1) * interval],
                        true_value[i, j * interval: (j + 1) * interval],
                        high_value[i, :, j * interval: (j + 1) * interval],
                        low_value[i, :, j * interval: (j + 1) * interval],
                        probability_range,
                        file_path,
                        xlabel='Timestamp/Step',
                        ylabel='Power/KW',
                        xlim=xlim,
                        ylim=ylim)


def draw_comp_probabilistic_figure(exp_name1, exp_name2, model_names, interval=128, folder=None, selected_data=None,
                                   replace_regex=None):
    if replace_regex is None:
        replace_regex = []

    pred_value1, true_value1, high_value1, low_value1 = get_all_value_inverse(exp_name1)
    pred_value2, true_value2, high_value2, low_value2 = get_all_value_inverse(exp_name2)

    config_row = get_config_row(exp_name1)
    pred_length = config_row['pred_len']
    data_length = pred_value1.shape[1]
    probability_range = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    selected_probability_range = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    def contact_list_to_str(contact_list):
        _str = ''
        for t in range(len(contact_list)):
            if t == 0:
                _str += f'{contact_list[t]}'
            else:
                _str += f'-{contact_list[t]}'
        return _str

    # draw selected figures
    if selected_data is not None:
        for k in selected_data:
            i = k[0] - 1
            j = k[1] - 1
            x1 = k[2] if len(k) >= 3 else None
            x2 = k[3] if len(k) >= 4 else None

            _path = out_dir
            if not os.path.exists(_path):
                os.makedirs(_path)

            # skip the last part of the data
            if j * interval >= data_length:
                continue

            if x1 is not None and x2 is not None:
                file_name = (f'PF Comp Pred {pred_length} CL {contact_list_to_str(selected_probability_range)} '
                             f'Step {i + 1} Data {j + 1}'
                             f' ({x1}-{x2}).png')
            else:
                file_name = (f'PF Comp Pred {pred_length} CL {contact_list_to_str(selected_probability_range)} '
                             f'Step {i + 1} Data {j + 1}.png')
            for regex in replace_regex:
                file_name = file_name.replace(regex[0], regex[1])

            if folder is not None:
                if not os.path.exists(os.path.join(_path, folder)):
                    os.makedirs(os.path.join(_path, folder))
                file_path = os.path.join(_path, folder, file_name)
            else:
                file_path = os.path.join(_path, file_name)

            draw_comp_figure(model_names,
                             range(interval),
                             range(x1, x2) if x1 is not None and x2 is not None else None,
                             pred_value1[i, j * interval: (j + 1) * interval],
                             true_value1[i, j * interval: (j + 1) * interval],
                             high_value1[i, :, j * interval: (j + 1) * interval],
                             low_value1[i, :, j * interval: (j + 1) * interval],
                             pred_value2[i, j * interval: (j + 1) * interval],
                             true_value2[i, j * interval: (j + 1) * interval],
                             high_value2[i, :, j * interval: (j + 1) * interval],
                             low_value2[i, :, j * interval: (j + 1) * interval],
                             probability_range,
                             selected_probability_range,
                             file_path,
                             xlabel='Timestamp/Step',
                             ylabel='Power/KW',
                             use_window=x1 is not None and x2 is not None)


# AL-QSQF
draw_probabilistic_figure(exp_name='LSTM-AQ_Electricity_96',
                          interval=128,
                          folder=None,
                          selected_data=[[16, 11, None, [1500, 5500]],
                                         [32, 19, None, [1500, 5500]],
                                         [64, 17, None, [1500, 5000]],
                                         [96, 20, None, [1500, 5000]]],
                          replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])

# QSQF-C
draw_probabilistic_figure(exp_name='QSQF-C_Electricity_96',
                          interval=128,
                          folder=None,
                          selected_data=[[16, 11, None, [1500, 5500]],
                                         [32, 19, None, [1500, 5500]],
                                         [64, 17, None, [1500, 5000]],
                                         [96, 20, None, [1500, 5000]]],
                          replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])

# AL-QSQF & QSQF-C
draw_comp_probabilistic_figure(exp_name1='LSTM-AQ_Electricity_96',
                               exp_name2='QSQF-C_Electricity_96',
                               model_names=['AL-QSQF', 'QSQF-C'],
                               interval=128,
                               folder=None,
                               selected_data=[[16, 11],
                                              [32, 19],
                                              [64, 17],
                                              [96, 20]],
                               replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])

draw_comp_probabilistic_figure(exp_name1='LSTM-AQ_Electricity_96',
                               exp_name2='QSQF-C_Electricity_96',
                               model_names=['AL-QSQF', 'QSQF-C'],
                               interval=128,
                               folder=None,
                               selected_data=[[16, 11, 84, 97],
                                              [32, 19, 86, 99],
                                              [64, 17, 40, 53],
                                              [96, 20, 84, 97]],
                               replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])
