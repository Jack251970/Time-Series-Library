import os

import matplotlib.pyplot as plt

from analyze.test_data_factory import get_loss

from utils.tools import set_times_new_roman_font

set_times_new_roman_font()

output_dir = 'loss'


def output_loss_figure(exp_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loss, vali_loss, test_loss = get_loss(exp_name)

    plt.clf()
    plt.plot(train_loss.squeeze(), color='blue', label='Train Loss')
    plt.plot(vali_loss.squeeze(), color='red', label='Validation Loss ')
    plt.plot(test_loss.squeeze(), color='green', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{exp_name} loss.png'))


# output_loss_figure('LSTM-AQ(HLF)_Electricity_96')
# output_loss_figure('LSTM-AQ_Electricity_96')
# output_loss_figure('LSTM-AQ2_Electricity_96')
# output_loss_figure('LSTM-AQ3_Electricity_96')
# output_loss_figure('LSTM-AQ4_Electricity_96')


def output_multi_loss_figure(_list, _file, xlabel=None, ylabel=None, font_size=18):
    plt.figure(figsize=(12.8, 9.6))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.clf()
    for i in _list:
        exp_name, color, label = i
        train_loss, vali_loss, test_loss = get_loss(exp_name)
        plt.plot(train_loss.squeeze(), color=color, label=label)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    path = os.path.join(output_dir, _file)
    plt.savefig(path.replace('.png', '.pdf'), format='pdf')


output_multi_loss_figure([['LSTM-AQ(HLF)_Electricity_96', 'blue', 'Ours'],
                          ['LSTM-AQ_Electricity_96', 'red', 'CRPS'],
                          ['LSTM-AQ2_Electricity_96', 'green', 'MSE'],
                          ['LSTM-AQ3_Electricity_96', 'gray', 'MAE'],
                          ['LSTM-AQ4_Electricity_96', 'purple', 'MQL']],
                         _file='LF AL-QSQF Multi Electricity Pred 96.png',
                         xlabel='Epoch',
                         ylabel='Train Loss')
