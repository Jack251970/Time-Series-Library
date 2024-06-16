import math
import os

from tqdm import tqdm

from analyze.test_data_factory import get_all_value_inverse, get_config_row
from utils.tools import draw_figure, set_times_new_roman_font

set_times_new_roman_font()

out_dir = 'probabilistic_figure'


def draw_probabilistic_figure(exp_name, samples_index, folder=None, max_data_length=None, replace_regex=None):
    if replace_regex is None:
        replace_regex = []

    pred_value, true_value, high_value, low_value = get_all_value_inverse(exp_name)

    config_row = get_config_row(exp_name)
    pred_length = int(config_row['pred_len'])
    data_length = pred_value.shape[1]
    probability_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # draw figures
    print('drawing probabilistic figure')
    for i in tqdm(range(pred_length)):
        if i in samples_index:
            _path = os.path.join(out_dir, f'step {i}')
            if not os.path.exists(_path):
                os.makedirs(_path)

            interval = 128
            num = math.floor(data_length / interval)
            for j in range(num):
                if j * interval >= data_length:
                    continue
                if max_data_length is not None and j < max_data_length:
                    if folder is not None:
                        if not os.path.exists(os.path.join(_path, folder)):
                            os.makedirs(os.path.join(_path, folder))
                        file_name = os.path.join(_path, folder, f'PF {exp_name} Pred {pred_length} Step {i + 1} '
                                                                     f'Data {j + 1}.png')
                    else:
                        file_name = os.path.join(_path, f'PF {exp_name} Pred {pred_length} Step {i + 1} '
                                                        f'Data {j + 1}.png')

                    # 执行替换规则
                    for regex in replace_regex:
                        file_name = file_name.replace(regex[0], regex[1])

                    draw_figure(range(interval),
                                pred_value[i, j * interval: (j + 1) * interval],
                                true_value[i, j * interval: (j + 1) * interval],
                                high_value[i, :, j * interval: (j + 1) * interval],
                                low_value[i, :, j * interval: (j + 1) * interval],
                                probability_range,
                                file_name,
                                ylim=[1500, 4500])


# AL-QSQF
draw_probabilistic_figure(exp_name='LSTM-AQ_Electricity_96',
                          samples_index=[15, 31, 63, 95],
                          max_data_length=100,
                          folder='AL-QSQF',
                          replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])

# # QSQF-C
draw_probabilistic_figure(exp_name='QSQF-C_Electricity_96',
                          samples_index=[15, 31, 63, 95],
                          max_data_length=100,
                          folder='QSQF-C',
                          replace_regex=[['QSQF-C_Electricity_96', 'QSQF-C Electricity']])
