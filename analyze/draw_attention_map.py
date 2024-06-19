import math
import os

from tqdm import tqdm

from analyze.test_data_factory import get_attention_map, get_config_row
from utils.tools import draw_attention_map, set_times_new_roman_font

set_times_new_roman_font()

output_dir = 'attention_map'


def draw_attention_map_figure(exp_name, selected_loader_data=None, selected_step_data=None, folder=None,
                              replace_regex=None):
    attention_maps = get_attention_map(exp_name)

    config_row = get_config_row(exp_name)
    batch_size = config_row['batch_size']
    seq_length = config_row['seq_len']
    pred_length = config_row['pred_len']
    n_heads = config_row['n_heads']
    loader_length = attention_maps.shape[0]

    # draw attention map for every loader
    if selected_loader_data is not None:
        for k in selected_loader_data:
            i = k[0] - 1
            j = k[1] - 1
            only_average = k[2] if len(k) >= 3 else False

            _path = os.path.join(output_dir, folder, f'loader {i + 1}')
            if not os.path.exists(_path):
                os.makedirs(_path)

            attention_map = attention_maps[i]
            attention_map = attention_map.reshape(batch_size, n_heads, 1 * pred_length, seq_length)  # [*, 8, 32, 96]

            file_name = f'AM {exp_name} Pred {pred_length} Loader {i + 1} Data {j + 1}.png'
            for regex in replace_regex:
                file_name = file_name.replace(regex[0], regex[1])

            draw_attention_map(attention_map[j], os.path.join(_path, file_name), only_average=only_average, cols=3)
    else:
        for i in range(loader_length):
            _path = os.path.join(output_dir, folder, f'loader {i + 1}')
            if not os.path.exists(_path):
                os.makedirs(_path)

            attention_map = attention_maps[i]
            attention_map = attention_map.reshape(batch_size, n_heads, 1 * pred_length, seq_length)  # [*, 8, 32, 96]

            interval = 96
            num = math.floor(loader_length * batch_size / interval)
            for j in tqdm(range(num), desc=f'pred {i}'):
                file_name = f'AM {exp_name} Pred {pred_length} Loader {i + 1} Data {j + 1}.png'
                for regex in replace_regex:
                    file_name = file_name.replace(regex[0], regex[1])

                draw_attention_map(attention_map[j], os.path.join(_path, file_name), cols=3)

    # draw attention map for every prediction step
    if selected_step_data is not None:
        for k in selected_step_data:
            i = k[0] - 1
            j = k[1] - 1
            only_average = k[2] if len(k) >= 3 else False

            _path = os.path.join(output_dir, folder, f'step {i + 1}', )
            if not os.path.exists(_path):
                os.makedirs(_path)

            attention_map = attention_maps[:, i, :, :, :, :]  # [61, 256, 8, 1, 96]
            attention_map = attention_map.reshape(loader_length * batch_size, n_heads, 1, seq_length)  # [*, 8, 1, 96]

            interval = 96

            _attention_map = attention_map[j * interval: (j + 1) * interval]  # [96, 8, 1, 96]
            _attention_map = _attention_map.reshape(n_heads, 1 * interval, seq_length)  # [8, 96, 96]

            file_name = f'AM {exp_name} Pred {pred_length} Step {i + 1} Step {j + 1}.png'
            for regex in replace_regex:
                file_name = file_name.replace(regex[0], regex[1])

            draw_attention_map(attention_map[j], os.path.join(_path, file_name), only_average=only_average, cols=3)
    else:
        for i in range(pred_length):
            _path = os.path.join(output_dir, folder, f'step {i + 1}', )
            if not os.path.exists(_path):
                os.makedirs(_path)

            attention_map = attention_maps[:, i, :, :, :, :]  # [61, 256, 8, 1, 96]
            attention_map = attention_map.reshape(loader_length * batch_size, n_heads, 1, seq_length)  # [*, 8, 1, 96]

            interval = 96
            num = math.floor(loader_length * batch_size / interval)
            for j in tqdm(range(num), desc=f'pred {i}'):
                _attention_map = attention_map[j * interval: (j + 1) * interval]  # [96, 8, 1, 96]
                _attention_map = _attention_map.reshape(n_heads, 1 * interval, seq_length)  # [8, 96, 96]

                file_name = f'AM {exp_name} Pred {pred_length} Step {i + 1} Step {j + 1}.png'
                for regex in replace_regex:
                    file_name = file_name.replace(regex[0], regex[1])

                draw_attention_map(attention_map[j], os.path.join(_path, file_name), cols=3)


draw_attention_map_figure(exp_name='LSTM-AQ_Electricity_96',
                          selected_loader_data=[[1, 11, True]],
                          selected_step_data=[],
                          folder='Electricity',
                          replace_regex=[['LSTM-AQ_Electricity_96', 'AL-QSQF Electricity']])

draw_attention_map_figure(exp_name='LSTM-AQ_Exchange_96',
                          selected_loader_data=[[1, 43, True]],
                          selected_step_data=[],
                          folder='Exchange',
                          replace_regex=[['LSTM-AQ_Exchange_96', 'AL-QSQF Electricity']])

draw_attention_map_figure(exp_name='LSTM-AQ_Traffic_96',
                          selected_loader_data=[[1, 17, True]],
                          selected_step_data=[],
                          folder='Traffic',
                          replace_regex=[['LSTM-AQ_Traffic_96', 'AL-QSQF Electricity']])
