import math
import os

from tqdm import tqdm

from analyze.test_data_factory import get_attention_map, get_config_row
from utils.tools import draw_attention_map, set_times_new_roman_font

set_times_new_roman_font()

output_dir = 'attention_map'


def draw_attention_map_figure(exp_name, folder_path, max_loader=3, selected_pred_step_indexes=None):
    if selected_pred_step_indexes is None:
        selected_pred_step_indexes = []

    attention_maps = get_attention_map(exp_name)

    config_row = get_config_row(exp_name)
    batch_size = config_row['batch_size']
    seq_length = config_row['seq_len']
    pred_length = config_row['pred_len']
    n_heads = config_row['n_heads']
    loader_length = attention_maps.shape[0]

    # draw attention map for every loader
    for i in range(loader_length if loader_length < max_loader else max_loader):
        _path = os.path.join(output_dir, f'loader {i + 1}', folder_path)
        if not os.path.exists(_path):
            os.makedirs(_path)

        attention_map = attention_maps[i]
        attention_map = attention_map.reshape(batch_size, n_heads, 1 * pred_length, seq_length)  # [512, 8, 32, 96]

        for j in tqdm(range(batch_size), desc=f'loader {i + 1}'):
            _ = attention_map[j]
            draw_attention_map(attention_map[j], os.path.join(_path, f'attention map {j + 1}.png'), cols=3)

    # draw attention map for every prediction step
    for i in selected_pred_step_indexes:
        _path = os.path.join(output_dir, f'step {i + 1}', folder_path)
        if not os.path.exists(_path):
            os.makedirs(_path)

        attention_map = attention_maps[:, i, :, :, :, :]  # [61, 256, 8, 1, 96]
        attention_map = attention_map.reshape(loader_length * batch_size, n_heads, 1, seq_length)  # [16384, 8, 1, 96]

        interval = 96
        num = math.floor(loader_length * batch_size / interval)
        for j in tqdm(range(num), desc=f'pred {i}'):
            _attention_map = attention_map[j * interval: (j + 1) * interval]  # [96, 8, 1, 96]
            _attention_map = _attention_map.reshape(n_heads, 1 * interval, seq_length)  # [8, 96, 96]
            draw_attention_map(_attention_map, os.path.join(_path, f'attention map {j + 1}.png'), cols=3)


draw_attention_map_figure(exp_name='LSTM-AQ_Electricity_96',
                          folder_path='LSTM-AQ',
                          max_loader=3,
                          selected_pred_step_indexes=[0, 31, 63, 95])
