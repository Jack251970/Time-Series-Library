import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sns

plt.switch_backend('agg')


def set_times_new_roman_font():
    from matplotlib import rcParams

    config = {
        "font.family": 'serif',
        "font.size": 12,
        "font.serif": ['Times New Roman'],
        # "mathtext.fontset": 'stix',
        # 'axes.unicode_minus': False
    }
    rcParams.update(config)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    else:
        raise NotImplementedError

    if epoch in lr_adjust.keys():
        # get lr in dictionary
        lr = lr_adjust[epoch]

        # update lr in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return 'Updating learning rate to {}'.format(lr)
    else:
        return None


class EarlyStopping:
    def __init__(self, checkpoints_file, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.checkpoints_file = checkpoints_file
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            _ = self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            _ = f'EarlyStopping counter: {self.counter} out of {self.patience}'
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            _ = self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        return _

    def save_checkpoint(self, val_loss, model, path):
        best_model_path = path + '/' + self.checkpoints_file
        if self.verbose:
            _ = (f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                 f'Saving model to {best_model_path}.')
        else:
            _ = None
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), best_model_path)
        self.val_loss_min = val_loss
        return _


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def draw_figure(x, pred, true, high, low, pred_range, path, xlim=None, ylim=None):
    plt.clf()
    plt.plot(pred.squeeze(), label='Predicted Value', color='red')
    plt.plot(true.squeeze(), label='True Value', color='blue')
    if pred_range is not None:
        for j in range(len(pred_range)):
            # plt.plot(high[j, :].squeeze(), label='High Value', color='green')
            # plt.plot(low[j, :].squeeze(), label='Low Value', color='green')
            plt.fill_between(x, high[j, :].squeeze(), low[j, :].squeeze(), color='gray',
                             alpha=1 - pred_range[j])
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.savefig(path)


def draw_density_figure(samples, true, path):  # [99], []
    plt.clf()

    # Plotting the density plot of the samples
    sns.kdeplot(samples.squeeze(), fill=True, label='Probability Density')

    # Adding a vertical line for the true value
    plt.axvline(true.squeeze(), color='r', linestyle='--', label='True Value')

    # Adding labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    plt.savefig(path)


def draw_attention_map(att_map, path, cols=4):
    """
    Draw attention map
    attn_map: (num_heads, seq_len_1, seq_len_2)
    """
    # prepare attention maps for each head
    to_shows = []
    n_heads = att_map.shape[0]
    for i in range(n_heads):
        to_shows.append((att_map[i], f'Head {i}'))
    if n_heads != 1:
        average_att_map = att_map.mean(axis=0)
        to_shows.append((average_att_map, 'Head Average'))

    # draw attention map
    plt.clf()
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    if rows == 1:
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[j].imshow(image)
            axs[j].set_title(title)
            axs[j].set_yticks([])
            axs[j].set_xticks([])
    else:
        for i in range(rows):
            for j in range(cols):
                try:
                    image, title = next(it)
                except StopIteration:
                    image = np.zeros_like(to_shows[0][0])
                    title = 'pad'
                axs[i, j].imshow(image)
                axs[i, j].set_title(title)
                axs[i, j].set_yticks([])
                axs[i, j].set_xticks([])
    plt.legend('')
    plt.savefig(path)


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
