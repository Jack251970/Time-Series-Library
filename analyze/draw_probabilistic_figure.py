import os

from analyze.test_data_factory import get_all_value_inverse, get_config_row
from utils.tools import draw_figure, set_times_new_roman_font

set_times_new_roman_font()

out_dir = 'probabilistic_figure'


def draw_probabilistic_figure(exp_name, interval=128, folder=None, selected_data=None, replace_regex=None):
    if replace_regex is None:
        replace_regex = []

    pred_value, true_value, high_value, low_value = get_all_value_inverse(exp_name)

    config_row = get_config_row(exp_name)
    pred_length = config_row['pred_len']
    data_length = pred_value.shape[1]
    probability_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # draw selected figures
    if selected_data is not None:
        for k in selected_data:
            i = k[0] - 1
            j = k[1] - 1
            xlim = k[2] if len(k) >= 3 else None
            ylim = k[3] if len(k) >= 4 else None

            _path = os.path.join(out_dir, f'step {i + 1}')
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
                        xlim=xlim,
                        ylim=ylim)


# AL-QSQF
draw_probabilistic_figure(exp_name='LSTM-AQ_Electricity_96',
                          interval=128,
                          folder='AL-QSQF',
                          selected_data=[[16, 11, None, [1500, 5500]],
                                         [32, 19, None, [1500, 5500]],
                                         [64, 17, None, [1500, 5000]],
                                         [96, 20, None, [1500, 5000]]],
                          replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])

# QSQF-C
draw_probabilistic_figure(exp_name='QSQF-C_Electricity_96',
                          interval=128,
                          folder='QSQF-C',
                          selected_data=[[16, 11, None, [1500, 5500]],
                                         [32, 19, None, [1500, 5500]],
                                         [64, 17, None, [1500, 5000]],
                                         [96, 20, None, [1500, 5000]]],
                          replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])
