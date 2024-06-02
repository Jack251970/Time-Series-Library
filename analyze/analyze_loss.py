import numpy as np

import matplotlib.pyplot as plt

from analyze.test_data_factory import get_loss

exp_name = 'LSTM-AQ_Electricity_96'
train_loss, vali_loss, test_loss = get_loss(exp_name)

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
