import os

import numpy as np

import matplotlib.pyplot as plt

settings_1 = 'probability_forecast_wind_96_16_LSTM-CQ_custom_ftMS_sl96_ll0_pl16_dm512_nh8_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-10 12-13-12'
settings_2 = 'probability_forecast_wind_96_16_LSTM-CQ_custom_ftMS_sl96_ll0_pl16_dm512_nh8_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-10 12-13-12'

files = ['train_loss.npy', 'vali_loss.npy', 'test_loss.npy']


def get_loss(settings):
    root = './../process/'

    _train_loss = None
    _vali_loss = None
    _test_loss = None

    for file in files:
        path = os.path.join(root, settings, file)
        data = np.load(path)
        if file == 'train_loss.npy':
            _train_loss = data
        elif file == 'vali_loss.npy':
            _vali_loss = data
        elif file == 'test_loss.npy':
            _test_loss = data

    print(_train_loss.shape)

    return _train_loss, _vali_loss, _test_loss


train_loss_1, vali_loss_1, test_loss_1 = get_loss(settings_1)

print(f'train_loss: {train_loss_1}')
print(f'vali_loss: {vali_loss_1}')
print(f'test_loss: {test_loss_1}')

train_loss_2, vali_loss_2, test_loss_2 = get_loss(settings_2)

print(f'train_loss: {train_loss_2}')
print(f'vali_loss: {vali_loss_2}')
print(f'test_loss: {test_loss_2}')

plt.clf()
plt.plot(train_loss_1.squeeze(), color='blue', label='Train Loss without FDR')
plt.plot(train_loss_2.squeeze(), color='red', label='Train Loss with FDR')
plt.legend()
plt.savefig('train_loss.png')

plt.clf()
plt.plot(vali_loss_1.squeeze(), color='blue', label='Validation Loss without FDR')
plt.plot(vali_loss_2.squeeze(), color='red', label='Validation Loss with FDR')
plt.legend()
plt.savefig('vali_loss.png')

plt.clf()
plt.plot(test_loss_1.squeeze(), color='blue', label='Test Loss without FDR')
plt.plot(test_loss_2.squeeze(), color='red', label='Test Loss with FDR')
plt.legend()
plt.savefig('test_loss.png')

settings = 'probability_forecast_OT_96_16_LSTM-ED-CQ_custom_ftMS_sl96_ll0_pl16_dm512_nh8_el1_dl1_dmavg_ma25_df2048_fc1_ebtimeF_dtTrue_deExp_2024-04-12 08-27-51'

train_loss, vali_loss, test_loss = get_loss(settings)

print(f'train_loss: {train_loss}')
print(f'vali_loss: {vali_loss}')
print(f'test_loss: {test_loss}')

plt.clf()
plt.plot(train_loss.squeeze(), color='blue', label='Train Loss')
plt.plot(vali_loss.squeeze(), color='red', label='Validation Loss')
plt.plot(test_loss.squeeze(), color='green', label='Test Loss')
plt.legend()
plt.savefig('loss.png')