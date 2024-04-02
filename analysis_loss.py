import os

import numpy as np

root_path = './'

files = ['train_loss.npy', 'vali_loss.npy', 'test_loss.npy']


def get_loss(root):
    _train_loss = None
    _vali_loss = None
    _test_loss = None

    for file in files:
        path = os.path.join(root, file)
        data = np.load(path)
        if file == 'train_loss.npy':
            _train_loss = data
        elif file == 'vali_loss.npy':
            _vali_loss = data
        elif file == 'test_loss.npy':
            _test_loss = data
        # print(data.shape)

    return _train_loss, _vali_loss, _test_loss


train_loss, vali_loss, test_loss = get_loss(root_path)

print(f'train_loss: {train_loss}')
print(f'vali_loss: {vali_loss}')
print(f'test_loss: {test_loss}')
