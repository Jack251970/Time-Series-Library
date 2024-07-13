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


output_loss_figure('LSTM-AQ_Electricity_96')
output_loss_figure('LSTM-AQ2_Electricity_96')
output_loss_figure('LSTM-AQ3_Electricity_96')
output_loss_figure('LSTM-AQ4_Electricity_96')


def output_multi_loss_figure(_list):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.clf()
    for i in _list:
        exp_name, color, label = i
        train_loss, vali_loss, test_loss = get_loss(exp_name)
        plt.plot(train_loss.squeeze(), color=color, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'multi loss.png'))


output_multi_loss_figure([['LSTM-AQ_Electricity_96', 'blue', 'CRPS'],
                          ['LSTM-AQ2_Electricity_96', 'red', 'MSE'],
                          ['LSTM-AQ3_Electricity_96', 'green', 'MAE'],
                          ['LSTM-AQ4_Electricity_96', 'purple', 'MQL']])
