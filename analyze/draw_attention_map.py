import csv
import math
import os

import numpy as np
from tqdm import tqdm

from analyze.test_data_factory import get_attention_map_path, data_folder, fieldnames, get_exp_settings
from utils.tools import draw_attention_map

folder_path = 'attention_map'
exp_name = 'LSTM-AQ_Electricity_96'
exp_settings = get_exp_settings(exp_name)
attention_maps = np.load(get_attention_map_path(exp_name))


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
    return int(target_row['batch_size']), int(target_row['seq_len']), int(target_row['pred_len']), \
             int(target_row['n_heads'])


batch_size, seq_length, pred_length, n_heads = get_exp_config(exp_settings)
loader_length = attention_maps.shape[0]

# draw attention map
print('drawing attention map')
for i in tqdm(range(loader_length)):
    _path = os.path.join(folder_path, f'attention_map', f'loader {i}')
    if not os.path.exists(_path):
        os.makedirs(_path)

    attention_map = attention_maps[i]
    attention_map = attention_map.reshape(batch_size, n_heads, 1 * pred_length, seq_length)
    for j in range(batch_size):
        _ = attention_map[j]
        draw_attention_map(attention_map[j], os.path.join(_path, f'attention map {j}.png'), cols=3)

for i in tqdm(range(pred_length)):
    _path = os.path.join(folder_path, f'attention_map', f'step {i}')
    if not os.path.exists(_path):
        os.makedirs(_path)

    attention_map = attention_maps[:, i, :, :, :, :]  # [61, 256, 8, 1, 96]
    attention_map = attention_map.reshape(loader_length * batch_size, n_heads, 1, seq_length)
    # [15616, 8, 1, 96]

    interval = 96
    num = math.floor(loader_length * batch_size / interval)
    for j in range(num):
        # if j * interval >= data_length:
        #     continue

        _attention_map = attention_map[j * interval: (j + 1) * interval]  # [96, 8, 1, 96]
        _attention_map = _attention_map.reshape(n_heads, 1 * interval, seq_length)
        # [8, 96, 96]
        draw_attention_map(_attention_map, os.path.join(_path, f'attention map {j}.png'), cols=3)
