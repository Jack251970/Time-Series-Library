import os

import numpy as np

from data_provider.data_loader import Dataset_Custom
from run_optimizer import h
from utils.metrics import metric

root_path = 'results/power_96_96_Autoformer_custom_ftM_sl96_ll96_pl96_dm512_nh12_el1_dl1_df2048_fc2_ebtimeF_dtTrue_Exp_0'

files = ['metrics.npy', 'pred.npy', 'true.npy']

pred_data = None
true_data = None

for file in files:
    path = os.path.join(root_path, file)
    data = np.load(path)
    if file == 'pred.npy':
        pred_data = data
    elif file == 'true.npy':
        true_data = data
    # print(data.shape)


def output_metrix(preds, trues):
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))

    pred_data_power = preds[:, :, -1]
    true_data_power = trues[:, :, -1]
    mae, mse, _, _, _ = metric(pred_data_power, true_data_power)
    print('mse_power:{}, mae_power:{}'.format(mse, mae))


def analysis_error(preds, trues, k=10, inverse=False, print_feature_num=True, print_all_features=False):
    """
    choose the k-top data of error_data and print the corresponding error_data, pred_data and true_data
    """
    # print data with 2 decimal places and not use scientific notation
    np.set_printoptions(suppress=True, precision=3)
    
    n, s, f = preds.shape

    error_data = preds - trues  # (5684, 96, 14)

    # convert to 2D (sample_num, feature_num) to adapt to scaler.inverse_transform
    error_data = np.reshape(error_data, (-1, error_data.shape[-1]))  # (545664, 14)
    preds = np.reshape(preds, (-1, preds.shape[-1]))
    trues = np.reshape(trues, (-1, trues.shape[-1]))

    if inverse:
        config = h.get_optimizer_settings()['search_space']['Autoformer']

        data_set = Dataset_Custom(
            root_path=config['root_path']['_value'],
            data_path=config['data_path']['_value'],
            flag='train',
            size=[config['seq_len']['_value'], config['label_len']['_value'], config['pred_len']['_value']],
            features=config['features']['_value'],
            target=config['target']['_value'],
            timeenc=(0 if config['embed']['_value'] != 'timeF' else 1),
            freq=config['freq']['_value'],
            scale=True,
            scaler='StandardScaler'
        )

        preds = data_set.inverse_transform(preds)
        trues = data_set.inverse_transform(trues)

    error_data = error_data.ravel()  # (7639296,)
    preds = preds.ravel()
    trues = trues.ravel()

    indices = abs(error_data).argsort()[-k:][::-1]  # get the top-k sort indices and reverse them
    indices_feature = indices % f
    if print_feature_num:
        print(indices_feature)

    if print_all_features:
        indices_sample_num = indices // f
        indices_sample_start = indices_sample_num * f
        indices_sample_end = indices_sample_start + f
        for i in range(len(indices_sample_start)):
            start = indices_sample_start[i]
            end = indices_sample_end[i]
            print(trues[start:end])

    top_k_error_data = error_data[indices]
    top_k_pred_data = preds[indices]
    top_k_true_data = trues[indices]

    print("Top K Error Data: ", top_k_error_data)
    print("Top K Predicted Data: ", top_k_pred_data)
    print("Top K True Data: ", top_k_true_data)


output_metrix(pred_data, true_data)
analysis_error(pred_data, true_data, k=3, inverse=True, print_all_features=True)

