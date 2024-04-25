import os

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

root_path = os.path.join('..', 'data')

file = os.path.join('probability_forecast', 'data_baseline_paper.csv')


def get_csv_data(path):
    global root_path
    _path = os.path.join(root_path, path)
    return pd.read_csv(_path)


baseline_data = get_csv_data(file)
# print(baseline_data.columns)


def update_data(_baseline_data, checked_columns):
    global root_path
    # 扫描所有数据文件
    file_paths = []
    for root, dirs, files in os.walk(root_path):
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
    for index, row in _baseline_data.iterrows():
        _check_data = {}
        for _column in checked_columns:
            _check_data[_column] = row[_column]
        _mse = row['mse']
        _mae = row['mae']
        _crps = row['crps']
        _pinaw = row['pinaw']

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
        _min_mse = _data['mse'].min()
        _min_mae = _data['mae'].min()
        _min_crps = _data['crps'].min()
        _min_pinaw = _data['pinaw'].min()

        # 获取最小指标
        if _min_mse < _mse:
            _baseline_data.loc[index, 'mse'] = _min_mse
            print(f"update mse for model {_check_data['model']}: {_mse} -> {_min_mse} in {_check_data}")
        if _min_mae < _mae:
            _baseline_data.loc[index, 'mae'] = _min_mae
            print(f"update mae for model {_check_data['model']}: {_mse} -> {_min_mse} in {_check_data}")
        if _min_crps < _crps:
            _baseline_data.loc[index, 'crps'] = _min_crps
            print(f"update crps for model {_check_data['model']}: {_mse} -> {_min_mse} in {_check_data}")
        if _min_pinaw < _pinaw:
            _baseline_data.loc[index, 'pinaw'] = _min_pinaw
            print(f"update pinaw for model {_check_data['model']}: {_mse} -> {_min_mse} in {_check_data}")

    return _baseline_data


# 更新最佳数据
checked_fieldnames = ['model', 'data_path', 'custom_params', 'seed', 'task_name', 'model_id', 'data',
                      'features', 'target', 'scaler', 'seq_len', 'label_len', 'pred_len', 'inverse']
baseline_data = update_data(baseline_data, checked_fieldnames)


def get_latex_table_data(_data, row_label, column_label, value_label, rearrange_column_label=None,
                         add_table_appendix=True, replace_nan=True, replace_regex=None):
    if replace_regex is None:
        replace_regex = []

    # 创建一个二维数组，用于存储表格数据
    table_data = pd.pivot_table(_data, values=value_label, index=row_label, columns=column_label, aggfunc='mean')

    # 重新排序列标签
    if rearrange_column_label is not None:
        table_data.columns = table_data.columns.swaplevel().reorder_levels(rearrange_column_label)
        table_data.sort_index(axis=1, level=0, inplace=True)

    # 合并多层列标签
    # table_data.columns = [' '.join(col).strip() for col in table_data.columns.values]

    # 将表格数据转换为latex格式，数字仅仅取出小数点后3位
    table_data = table_data.to_latex(float_format='%.3f')

    # 将nan替换为'-'
    if replace_nan:
        table_data = table_data.replace('NaN', '-')

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
                                  rearrange_column_label=['model', None],
                                  add_table_appendix=True,
                                  replace_nan=True,
                                  replace_regex=[['electricity/electricity.csv', 'Electricity'],
                                                 ['exchange_rate/exchange_rate.csv', 'Exchange'],
                                                 ['weather/weather.csv', 'Weather'],
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
                                  add_table_appendix=True,
                                  replace_nan=True,
                                  replace_regex=[['electricity/electricity.csv', 'Electricity'],
                                                 ['exchange_rate/exchange_rate.csv', 'Exchange'],
                                                 ['weather/weather.csv', 'Weather'],
                                                 ['data_path', ''],
                                                 ['pred_len', ''],
                                                 ['model', '']])

with open('reliability_table.txt', 'w') as f:
    f.write(latex_text)
