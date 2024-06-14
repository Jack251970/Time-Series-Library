import math
import os

from tqdm import tqdm

from analyze.test_data_factory import get_all_value_inverse, get_config_row
from utils.tools import draw_figure, set_times_new_roman_font

set_times_new_roman_font()

samples_index = [15, 31, 63, 95]

folder_path = 'probabilistic_figure'
exp_name = 'QSQF-C_Electricity_96'
pred_value, true_value, high_value, low_value = get_all_value_inverse(exp_name)

config_row = get_config_row(exp_name)
pred_length = int(config_row['pred_len'])
data_length = pred_value.shape[1]
probability_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# draw figures
print('drawing probabilistic figure')
for i in tqdm(range(pred_length)):
    if i in samples_index:
        _path = os.path.join(folder_path, f'step {i}')
        if not os.path.exists(_path):
            os.makedirs(_path)

        interval = 128
        num = math.floor(data_length / interval)
        for j in range(num):
            if j * interval >= data_length:
                continue
            if j == 0:
                draw_figure(range(interval),
                            pred_value[i, j * interval: (j + 1) * interval],
                            true_value[i, j * interval: (j + 1) * interval],
                            high_value[i, :, j * interval: (j + 1) * interval],
                            low_value[i, :, j * interval: (j + 1) * interval],
                            probability_range,
                            os.path.join(_path, f'PF QSQF-C Electricity Pred 96 Step {i+1} Data {j+1}.png'),
                            ylim=[1500, 4500])
