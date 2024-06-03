import os

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

root_path = '..'
data_dir = 'data'
data_folder = os.path.join(root_path, data_dir)
file = os.path.join('probability_forecast', 'data_baseline_paper.csv')


def get_csv_data(path):
    global data_folder
    _path = os.path.join(data_folder, path)
    return pd.read_csv(_path)


baseline_data = get_csv_data(file)


# print(baseline_data.columns)


def update_data(_baseline_data, checked_columns, target_columns, _core_target_fieldname):
    global data_folder
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

    # 检查标准数据中是否需要更新：若MSE,NAE,CRPS,PINAW中有指标可以更小，则更新
    _update_number = 0
    _core_update_number = 0
    _baseline_source_data = _baseline_data.copy()
    for index, row in _baseline_data.iterrows():
        _model = row['model']
        _dataset = row['data_path']
        _pred_len = row['pred_len']

        # 获取检查数据和目标数据
        _check_data = {}
        for _column in checked_columns:
            _check_data[_column] = row[_column]
        _target_data = {}
        for _column, _method in target_columns:
            _target_data[_column] = row[_column]

        # 获取检查数据都相同的数据
        _data = all_data
        for _column, _value in _check_data.items():
            # 如果数据相同，或者都是空值，则选择
            if pd.isna(_value):
                _data = _data[pd.isna(_data[_column])]
            else:
                _data = _data[_data[_column] == _value]
            if _data.empty:
                break
        if _data.empty:
            continue

        # 统计出最小的指标
        _min_target_data = {}
        for _column, _method in target_columns:
            if _method == 'min':
                _min_target_data[_column] = (_data[_column].min(), _method)
            elif _method == 'max':
                _min_target_data[_column] = (_data[_column].max(), _method)
            else:
                raise ValueError(f"unknown method: {_method}")

        # 获取最小指标
        for _column, (_value, _method) in _min_target_data.items():
            _baseline_value = _target_data[_column]
            if _method == 'min':
                if not pd.isna(_baseline_value) and _value < _baseline_value:
                    _baseline_data.loc[index, _column] = _value
                    if _column == _core_target_fieldname:
                        _baseline_source_data.loc[index] = row
                        _core_update_number += 1
                    _update_number += 1
                    print(
                        f"update {_column} for model {_model}, data {_dataset}, pred {_pred_len}: {_baseline_value} -> {_value}")
            elif _method == 'max':
                if not pd.isna(_baseline_value) and _value > _baseline_value:
                    _baseline_data.loc[index, _column] = _value
                    if _column == _core_target_fieldname:
                        _baseline_source_data.loc[index] = row
                        _core_update_number += 1
                    _update_number += 1
                    print(
                        f"update {_column} for model {_model}, data {_dataset}, pred {_pred_len}: {_baseline_value} -> {_value}")
            else:
                raise ValueError(f"unknown method: {_method}")

    print(f'update {_update_number} cells')
    print(f'update {_core_update_number} source rows')

    return _baseline_data, _baseline_source_data


# 更新最佳数据
checked_fieldnames = ['model', 'data_path', 'custom_params', 'seed', 'task_name', 'model_id', 'data',
                      'features', 'target', 'scaler', 'seq_len', 'label_len', 'pred_len', 'inverse']
target_fieldnames = [('mse', 'min'), ('mae', 'min'), ('crps', 'min'), ('pinaw', 'min')]
core_target_fieldname = 'mse'
baseline_data, baseline_source_data = update_data(baseline_data, checked_fieldnames, target_fieldnames,
                                                  core_target_fieldname)
baseline_source_data.to_csv('baseline_source_data.csv', index=False)


def get_latex_table_data(_data, row_label, column_label, value_label, replace_label=None, rearrange_column_label=None,
                         combine_column_label=False, add_table_appendix=True, replace_nan=True, replace_regex=None):
    if replace_regex is None:
        replace_regex = []

    # 替换列标签
    if replace_label is not None:
        for _replace_label, _ in replace_label:
            old_label = _replace_label[0]
            new_label = _replace_label[1]
            _data.rename(columns={old_label: new_label}, inplace=True)
            if old_label in row_label:
                row_label[row_label.index(old_label)] = new_label
            if old_label in column_label:
                column_label[column_label.index(old_label)] = new_label
            if old_label in value_label:
                value_label[value_label.index(old_label)] = new_label

    # 创建一个二维数组，用于存储表格数据
    table_data = pd.pivot_table(_data, values=value_label, index=row_label, columns=column_label, aggfunc='mean')

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

    return table_data


# MSE & MAE table
latex_text = get_latex_table_data(baseline_data,
                                  row_label=['data_path', 'pred_len'],
                                  column_label=['model'],
                                  value_label=['mse', 'mae'],
                                  replace_label=[(['mse', 'amse'], False),
                                                 (['mae', 'bmae'], False)],
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
                                                 ['model', '']])

with open('accuracy_table.txt', 'w') as f:
    f.write(latex_text)

# CRPS & PINAW table
latex_text = get_latex_table_data(baseline_data,
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
                                                 ['model', '']])

with open('reliability_table.txt', 'w') as f:
    f.write(latex_text)
