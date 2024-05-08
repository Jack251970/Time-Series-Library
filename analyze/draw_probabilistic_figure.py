import csv
import math
import os

import numpy as np
from tqdm import tqdm

from analyze.test_data_factory import get_all_value_inverse_path, data_folder, fieldnames, get_exp_settings
from utils.tools import draw_figure

folder_path = 'probabilistic_figure'
exp_name = 'QSQF-C_Electricity_96'
exp_settings = get_exp_settings(exp_name)

pred_value_path, true_value_path, high_value_path, low_value_path = get_all_value_inverse_path(exp_name)

pred_value = np.load(pred_value_path)
true_value = np.load(true_value_path)
high_value = np.load(high_value_path)
low_value = np.load(low_value_path)


def get_exp_config(_exp_settings):
    # scan all csv files under data folder
    file_paths = []
    for root, dirs, files in os.walk(str(data_folder)):
        for _file in files:
            if _file.endswith('.csv') and _file not in file_paths:
                _append_path = os.path.join(root, _file)
                file_paths.append(_append_path)

    # find target item
    target_row = None
    for file_path in file_paths:
        with open(file_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            next(reader)  # skip the header
            for row in reader:
                setting = row['setting']
                if setting == _exp_settings:
                    target_row = row

    # phase the target item
    return int(target_row['pred_len'])


pred_length = get_exp_config(exp_settings)
data_length = pred_value.shape[1]

probability_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# draw figures
print('drawing probabilistic figure')
for i in tqdm(range(pred_length)):
    _path = os.path.join(folder_path, f'probabilistic_figure', f'step {i}')
    if not os.path.exists(_path):
        os.makedirs(_path)

    interval = 128
    num = math.floor(data_length / interval)
    for j in range(num):
        if j * interval >= data_length:
            continue
        draw_figure(range(interval),
                    pred_value[i, j * interval: (j + 1) * interval],
                    true_value[i, j * interval: (j + 1) * interval],
                    high_value[i, :, j * interval: (j + 1) * interval],
                    low_value[i, :, j * interval: (j + 1) * interval],
                    probability_range,
                    os.path.join(_path, f'prediction {j}.png'))
