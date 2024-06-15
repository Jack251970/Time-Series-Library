import os

from analyze.analyze_data import output_table


# MSE & MAE table
output_table(file=os.path.join('probability_forecast', 'data_comp.csv'),
             output_file='accuracy_table_comp.txt',
             source_file='comp_source_data.csv',
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


# CRPS & PINAW table
output_table(file=os.path.join('probability_forecast', 'data_comp.csv'),
             output_file='reliability_table_comp.txt',
             source_file='comp_source_data.csv',
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