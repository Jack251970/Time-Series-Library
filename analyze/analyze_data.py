import os

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

# 设置数据位置
root_path = '..'
data_dir = 'data'
data_folder = os.path.join(root_path, data_dir)
output_dir = 'table'


def output_table(file, output_file, source_file,
                 checked_fieldnames, target_fieldnames, core_target_fieldname, save_source,
                 row_label, column_label, value_label, replace_label=None,
                 rearrange_column_label=None, combine_column_label=False, add_table_appendix=True, replace_nan=True,
                 replace_regex=None):
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取数据
    global data_folder
    path = os.path.join(data_folder, file)
    data = pd.read_csv(path)

    # 扫描所有数据文件
    file_paths = []
    for root, dirs, files in os.walk(data_folder):
        for _file in files:
            if _file == 'jump_data.csv':
                continue
            if _file.endswith('.csv') and _file not in file_paths:
                _append_path = os.path.join(root, _file)
                file_paths.append(_append_path)
    print(f'scan {len(file_paths)} data files')

    # 读取所有数据文件
    all_data = pd.DataFrame()
    for _file_path in file_paths:
        all_data = pd.concat([all_data, pd.read_csv(_file_path)], ignore_index=True)
    print(f'load {len(all_data)} records')

    # 检查标准数据中是否需要更新：若有指标可以更优，则更新
    update_number = 0
    core_update_number = 0
    source_data = data.copy()
    for index, row in data.iterrows():
        model = row['model']
        data_path = row['data_path']
        pred_len = row['pred_len']

        # 获取检查数据和目标数据
        _check_data = {}
        for _column in checked_fieldnames:
            _check_data[_column] = row[_column]
        _target_data = {}
        for _column, _method in target_fieldnames:
            _target_data[_column] = row[_column]

        # 获取检查数据都相同的数据
        filtered_data = all_data.copy()
        for _column, _value in _check_data.items():
            # 如果数据相同，或者都是空值，则选择
            if pd.isna(_value):
                filtered_data = filtered_data[pd.isna(filtered_data[_column])]
            else:
                filtered_data = filtered_data[filtered_data[_column] == _value]
            if filtered_data.empty:
                break
        if filtered_data.empty:
            continue

        # 统计出最小的指标
        optimized_data = {}
        for _column, _method in target_fieldnames:
            if _method == 'min':
                optimized_data[_column] = (filtered_data[_column].min(), filtered_data[_column].idxmin(), _method)
            elif _method == 'max':
                optimized_data[_column] = (filtered_data[_column].max(), filtered_data[_column].idxmax(), _method)
            elif _method == 'none':
                pass
            else:
                raise ValueError(f"unknown method: {_method}")

        # 获取最小指标
        for _column, (_value, _index, _method) in optimized_data.items():
            _data_value = _target_data[_column]
            if _method == 'min':
                if not pd.isna(_data_value) and _value < _data_value:
                    data.loc[index, _column] = _value
                    if _column == core_target_fieldname:
                        source_data.loc[index] = filtered_data.loc[_index]
                        core_update_number += 1
                        print(f"update core {_column} for model {model}, data {data_path}, pred {pred_len}: "
                              f"{_data_value} -> {_value}")
                    else:
                        print(f"update {_column} for model {model}, data {data_path}, pred {pred_len}: "
                              f"{_data_value} -> {_value}")
                    update_number += 1
            elif _method == 'max':
                if not pd.isna(_data_value) and _value > _data_value:
                    data.loc[index, _column] = _value
                    if _column == core_target_fieldname:
                        source_data.loc[index] = filtered_data.loc[_index]
                        core_update_number += 1
                        print(f"update core {_column} for model {model}, data {data_path}, pred {pred_len}: "
                              f"{_data_value} -> {_value}")
                    else:
                        print(f"update {_column} for model {model}, data {data_path}, pred {pred_len}: "
                              f"{_data_value} -> {_value}")
                    update_number += 1
            elif _method == 'none':
                pass
            else:
                raise ValueError(f"unknown method: {_method}")

    print(f'update {update_number} cells')
    print(f'update {core_update_number} source rows')

    # 保存最佳数据
    if save_source:
        source_data.to_csv(os.path.join(output_dir, source_file), index=False)

    # 初始化替换规则
    if replace_regex is None:
        replace_regex = []

    # 替换列标签
    if replace_label is not None:
        for _replace_label, _ in replace_label:
            old_label = _replace_label[0]
            new_label = _replace_label[1]
            data.rename(columns={old_label: new_label}, inplace=True)
            if old_label in row_label:
                row_label[row_label.index(old_label)] = new_label
            if old_label in column_label:
                column_label[column_label.index(old_label)] = new_label
            if old_label in value_label:
                value_label[value_label.index(old_label)] = new_label

    # 创建一个二维数组，用于存储表格数据
    table_data = pd.pivot_table(data, values=value_label, index=row_label, columns=column_label,
                                aggfunc='mean')

    # 重新排序列标签
    if rearrange_column_label is not None:
        table_data.columns = table_data.columns.swaplevel().reorder_levels(rearrange_column_label)
        table_data.sort_index(axis=1, level=0, inplace=True)

    # 合并多层列标签
    if combine_column_label:
        table_data.columns = [' '.join(col).strip() for col in table_data.columns.values]

    # 将表格数据转换为latex格式，数字仅仅取出小数点后3位
    table_data = table_data.to_latex(float_format='%.4f')

    # 将nan替换为'-'
    if replace_nan:
        table_data = table_data.replace('NaN', '-')

    # 处理替换标签
    if replace_label is not None:
        for _replace_label, keep_in_latex in replace_label:
            old_label = _replace_label[0]
            new_label = _replace_label[1]
            if not keep_in_latex:
                table_data = table_data.replace(new_label, old_label)

    # 执行替换规则
    for regex in replace_regex:
        table_data = table_data.replace(regex[0], regex[1])

    # 新增表格前缀
    if add_table_appendix:
        table_data = table_data.replace('\\begin{tabular}',
                                        '\\begin{table}[htbp]\n\\centering\n\\caption{Table}\n\\begin{tabular}')
        table_data = table_data.replace('\\end{tabular}', '\\end{tabular}\n\\end{table}')

    # 写入文件
    with open(os.path.join(output_dir, output_file), 'w') as writer:
        writer.write(table_data)

    print('\n')


# # baseline MSE & MAE table
# output_table(file=os.path.join('probability_forecast', 'data_baseline_paper.csv'),
#              output_file='accuracy_table_baseline.txt',
#              source_file='reliability_source_data_baseline.csv',
#              checked_fieldnames=['model', 'data_path', 'custom_params', 'seed', 'task_name', 'model_id', 'data',
#                                  'features', 'target', 'scaler', 'seq_len', 'label_len', 'pred_len', 'inverse'],
#              target_fieldnames=[('mse', 'min'), ('mae', 'min'), ('crps', 'min'), ('pinaw', 'min')],
#              core_target_fieldname='mse',
#              save_source=False,
#              row_label=['data_path', 'pred_len'],
#              column_label=['model'],
#              value_label=['mse', 'mae'],
#              replace_label=[(['mse', 'amse'], False), (['mae', 'bmae'], False)],
#              rearrange_column_label=['model', None],
#              combine_column_label=False,
#              add_table_appendix=True,
#              replace_nan=True,
#              replace_regex=[['electricity/electricity.csv', 'Electricity'],
#                             ['exchange_rate/exchange_rate.csv', 'Exchange'],
#                             ['weather/weather.csv', 'Weather'],
#                             ['traffic/traffic.csv', 'Traffic'],
#                             ['data_path', ''],
#                             ['pred_len', ''],
#                             ['model', ''],
#                             ['LSTM-AQ', 'AL-QSQF']])
#
# # baseline CRPS & PINAW table
# output_table(file=os.path.join('probability_forecast', 'data_baseline_paper.csv'),
#              output_file='reliability_table_baseline.txt',
#              source_file='reliability_source_data_baseline.csv',
#              checked_fieldnames=['model', 'data_path', 'custom_params', 'seed', 'task_name', 'model_id', 'data',
#                                  'features', 'target', 'scaler', 'seq_len', 'label_len', 'pred_len', 'inverse'],
#              target_fieldnames=[('mse', 'min'), ('mae', 'min'), ('crps', 'min'), ('pinaw', 'min')],
#              core_target_fieldname='mse',
#              save_source=False,
#              row_label=['data_path', 'pred_len'],
#              column_label=['model'],
#              value_label=['crps', 'pinaw'],
#              rearrange_column_label=['model', None],
#              combine_column_label=False,
#              add_table_appendix=True,
#              replace_nan=True,
#              replace_regex=[['electricity/electricity.csv', 'Electricity'],
#                             ['exchange_rate/exchange_rate.csv', 'Exchange'],
#                             ['weather/weather.csv', 'Weather'],
#                             ['traffic/traffic.csv', 'Traffic'],
#                             ['data_path', ''],
#                             ['pred_len', ''],
#                             ['model', ''],
#                             ['LSTM-AQ', 'AL-QSQF']])

# comp1 MSE & MAE table
output_table(file=os.path.join('probability_forecast', 'data_comp_lf.csv'),
             output_file='accuracy_table_comp1.txt',
             source_file='reliability_source_data_comp_lf.csv',
             checked_fieldnames=['model', 'data_path', 'custom_params', 'seed', 'task_name', 'model_id', 'data',
                                 'features', 'target', 'scaler', 'seq_len', 'label_len', 'pred_len', 'inverse'],
             target_fieldnames=[('mse', 'min'), ('mae', 'min'), ('crps', 'min'), ('pinaw', 'min')],
             core_target_fieldname='mse',
             save_source=False,
             row_label=['data_path', 'pred_len'],
             column_label=['model'],
             value_label=['mse', 'mae'],
             replace_label=[(['mse', 'amse'], False), (['mae', 'bmae'], False)],
             rearrange_column_label=['model', None],
             combine_column_label=False,
             add_table_appendix=True,
             replace_nan=True,
             replace_regex=[['electricity/electricity.csv', 'Electricity'],
                            ['exchange_rate/exchange_rate.csv', 'Exchange'],
                            ['weather/weather.csv', 'Weather'],
                            ['traffic/traffic.csv', 'Traffic'],
                            ['data_path', ''],
                            ['pred_len', ''],
                            ['model', ''],
                            ['LSTM-AQ', 'AL-QSQF']])


# comp1 CRPS & PINAW table
output_table(file=os.path.join('probability_forecast', 'data_comp_lf.csv'),
             output_file='reliability_table_comp1.txt',
             source_file='reliability_source_data_comp_lf.csv',
             checked_fieldnames=['model', 'data_path', 'custom_params', 'seed', 'task_name', 'model_id', 'data',
                                 'features', 'target', 'scaler', 'seq_len', 'label_len', 'pred_len', 'inverse'],
             target_fieldnames=[('mse', 'min'), ('mae', 'min'), ('crps', 'min'), ('pinaw', 'min')],
             core_target_fieldname='mse',
             save_source=False,
             row_label=['data_path', 'pred_len'],
             column_label=['model'],
             value_label=['crps', 'pinaw'],
             rearrange_column_label=['model', None],
             combine_column_label=False,
             add_table_appendix=True,
             replace_nan=True,
             replace_regex=[['electricity/electricity.csv', 'Electricity'],
                            ['exchange_rate/exchange_rate.csv', 'Exchange'],
                            ['weather/weather.csv', 'Weather'],
                            ['traffic/traffic.csv', 'Traffic'],
                            ['data_path', ''],
                            ['pred_len', ''],
                            ['model', ''],
                            ['LSTM-AQ', 'AL-QSQF']])

# # comp qsqf MSE & MAE & CRPS & PINAW table
# output_table(file=os.path.join('probability_forecast', 'data_qsqf_comp.csv'),
#              output_file='accuracy_reliability_table_comp_qsqf.txt',
#              source_file='accuracy_reliability_source_data_qsqf.csv',
#              checked_fieldnames=['model', 'data_path', 'custom_params', 'seed', 'task_name', 'model_id', 'data',
#                                  'features', 'target', 'scaler', 'seq_len', 'label_len', 'pred_len', 'inverse'],
#              target_fieldnames=[('mse', 'min'), ('mae', 'min'), ('crps', 'min'), ('pinaw', 'min')],
#              core_target_fieldname='mse',
#              save_source=False,
#              row_label=['data_path', 'pred_len'],
#              column_label=['model'],
#              value_label=['mse', 'mae', 'crps', 'pinaw'],
#              replace_label=[(['mse', 'amse'], False), (['mae', 'bmae'], False)],
#              rearrange_column_label=['model', None],
#              combine_column_label=False,
#              add_table_appendix=True,
#              replace_nan=True,
#              replace_regex=[['electricity/electricity.csv', 'Electricity'],
#                             ['exchange_rate/exchange_rate.csv', 'Exchange'],
#                             ['weather/weather.csv', 'Weather'],
#                             ['traffic/traffic.csv', 'Traffic'],
#                             ['data_path', ''],
#                             ['pred_len', ''],
#                             ['model', ''],
#                             ['LSTM-AQ', 'AL-QSQF']])
#
#
# # comp aq MSE & MAE & CRPS & PINAW table
# output_table(file=os.path.join('probability_forecast', 'data_comp.csv'),
#              output_file='accuracy_reliability_table_comp_aq.txt',
#              source_file='accuracy_reliability_source_data_aq.csv',
#              checked_fieldnames=['model', 'data_path', 'custom_params', 'seed', 'task_name', 'model_id', 'data',
#                                  'features', 'target', 'scaler', 'seq_len', 'label_len', 'pred_len', 'inverse'],
#              target_fieldnames=[('mse', 'min'), ('mae', 'min'), ('crps', 'min'), ('pinaw', 'min')],
#              core_target_fieldname='mse',
#              save_source=False,
#              row_label=['data_path', 'pred_len'],
#              column_label=['model'],
#              value_label=['mse', 'mae', 'crps', 'pinaw'],
#              replace_label=[(['mse', 'amse'], False), (['mae', 'bmae'], False)],
#              rearrange_column_label=['model', None],
#              combine_column_label=False,
#              add_table_appendix=True,
#              replace_nan=True,
#              replace_regex=[['electricity/electricity.csv', 'Electricity'],
#                             ['exchange_rate/exchange_rate.csv', 'Exchange'],
#                             ['weather/weather.csv', 'Weather'],
#                             ['traffic/traffic.csv', 'Traffic'],
#                             ['data_path', ''],
#                             ['pred_len', ''],
#                             ['model', ''],
#                             ['LSTM-AQ', 'AL-QSQF']])
