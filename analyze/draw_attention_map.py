import math
import os

from tqdm import tqdm

from analyze.test_data_factory import get_attention_map, get_config_row
from utils.tools import draw_attention_map, set_times_new_roman_font

set_times_new_roman_font()

folder_path = 'attention_map'
exp_name = 'LSTM-AQ_Electricity_96'
attention_maps = get_attention_map(exp_name)

config_row = get_config_row(exp_name)
batch_size = int(config_row['batch_size'])
seq_length = int(config_row['seq_len'])
pred_length = int(config_row['pred_len'])
n_heads = int(config_row['n_heads'])
loader_length = attention_maps.shape[0]

# draw attention map for every loader
print('drawing attention map for every loader')
for i in tqdm(range(loader_length)):
    _path = os.path.join(folder_path, f'loader {i}')
    if not os.path.exists(_path):
        os.makedirs(_path)

    attention_map = attention_maps[i]
    attention_map = attention_map.reshape(batch_size, n_heads, 1 * pred_length, seq_length)
    for j in range(batch_size):
        _ = attention_map[j]
        draw_attention_map(attention_map[j], os.path.join(_path, f'attention map {j}.png'), cols=3)

# draw attention map for every prediction step
print('drawing attention map for every prediction step')
for i in tqdm(range(pred_length)):
    _path = os.path.join(folder_path, f'step {i}')
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
