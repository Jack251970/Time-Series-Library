import numpy as np

import matplotlib.pyplot as plt

from analyze.test_data_factory import get_exp_settings, get_loss_path

exp_name = 'LSTM-AQ_Electricity_96'
exp_settings = get_exp_settings(exp_name)
loss_paths = get_loss_path(exp_name)


def get_loss(_paths):
    _train_loss = None
    _vali_loss = None
    _test_loss = None

    for _path in _paths:
        data = np.load(_path)
        if 'train' in _path:
            _train_loss = data
        elif 'vali' in _path:
            _vali_loss = data
        elif 'test' in _path:
            _test_loss = data

    return _train_loss, _vali_loss, _test_loss


train_loss, vali_loss, test_loss = get_loss(loss_paths)

print(f'train_loss: {train_loss}')
print(f'vali_loss: {vali_loss}')
print(f'test_loss: {test_loss}')

plt.clf()
plt.plot(train_loss.squeeze(), color='blue', label='Train Loss')
plt.plot(vali_loss.squeeze(), color='red', label='Validation Loss')
plt.plot(test_loss.squeeze(), color='green', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
