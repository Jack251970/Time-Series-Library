import os
import pandas as pd

file = 'probability_forecast/data_baseline_paper.csv'


def get_csv_data(path):
    root_path = '../data/'
    return pd.read_csv(os.path.join(root_path, path))


data = get_csv_data(file)
# print(data.columns)


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


latex_text = get_latex_table_data(data,
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

with open('table.txt', 'w') as f:
    f.write(latex_text)
