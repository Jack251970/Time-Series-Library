import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import pad

from models.QSQF.net_qspline_C import ConvLayer


class Model(nn.Module):
    def __init__(self, params, use_cnn=True, use_qrnn=False):
        """
        We define a recurrent network that predicts the future values
        of a time-dependent variable based on past inputs and covariances.

        Use cun for feature extraction.
        Use qrnn for choosing the qrnn to replace lstm.
        """
        super(Model, self).__init__()
        self.use_cnn = use_cnn
        self.use_qrnn = use_qrnn
        self.task_name = params.task_name
        input_size = params.enc_in + params.lag - 1  # take lag into account
        if use_cnn:
            input_size = input_size + 2 * 2 - (3 - 1) - 1 + 1  # take conv into account
            input_size = (input_size + 2 * 1 - (3 - 1) - 1) // 2 + 1  # take maxPool into account
        self.lstm_input_size = input_size
        self.lstm_hidden_dim = 40
        self.lstm_layers = 2
        self.sample_times = params.sample_times
        self.lstm_dropout = params.dropout
        self.num_spline = params.num_spline
        self.pred_start = params.seq_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start
        self.n_heads = 8

        # CNN
        self.cnn = ConvLayer(1)

        # LSTM
        if self.use_qrnn:
            from layers.pytorch_qrnn.torchqrnn import QRNN
            self.lstm = QRNN(input_size=self.lstm_input_size,
                             hidden_size=self.lstm_hidden_dim,
                             num_layers=self.lstm_layers,
                             dropout=self.lstm_dropout,
                             use_cuda=params.use_gpu)
        else:
            self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                                hidden_size=self.lstm_hidden_dim,
                                num_layers=self.lstm_layers,
                                bias=True,
                                batch_first=False,
                                bidirectional=False,
                                dropout=self.lstm_dropout)

            # initialize LSTM forget gate bias to be 1 as recommended by
            # http://proceedings.mlr.press/v37/jozefowicz15.pdf
            # noinspection PyProtectedMember
            for names in self.lstm._all_weights:
                for name in filter(lambda _n: "bias" in _n, names):
                    bias = getattr(self.lstm, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)

        # QSQM
        self.pre_beta_0 = nn.Linear(self.lstm_hidden_dim * self.lstm_layers, 1)
        self.pre_gamma = nn.Linear(self.lstm_hidden_dim * self.lstm_layers, self.num_spline)
        self.soft_plus = nn.Softplus()  # make sure parameter is positive\

        # Reindex
        self.new_index = [0]

    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'probability_forecast':
            # we don't need to use mark data because lstm can handle time series relation information
            batch = torch.cat((x_enc, y_enc), dim=1).float()
            train_batch = batch[:, :, :-1]
            labels_batch = batch[:, :, -1]
            return self.probability_forecast(train_batch, labels_batch)  # return loss list
        return None

    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, probability_range=None):
        if self.task_name == 'probability_forecast':
            batch = torch.cat((x_enc, y_enc), dim=1).float()
            train_batch = batch[:, :, :-1]
            if probability_range is None:
                probability_range = [0.5]
            return self.probability_forecast(train_batch, probability_range=probability_range)
        return None

    def run_lstm(self, x, hidden, cell):
        if self.use_cnn:
            x = self.cnn(x)  # [1, 256, 5]

        if self.use_qrnn:
            _, hidden = self.lstm(x, hidden)
        else:
            _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

        return hidden, cell

    @staticmethod
    def get_hidden_permute(hidden):
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0)  # [256, 2, 40]
        hidden_permute = hidden_permute.contiguous().view(hidden.shape[1], -1)  # [256, 80]

        return hidden_permute

    def get_qsqm_parameter(self, hidden_permute):
        pre_beta_0 = self.pre_beta_0(hidden_permute)  # [256, 1]
        beta_0 = self.soft_plus(pre_beta_0)  # [256, 1]
        pre_gamma = self.pre_gamma(hidden_permute)  # [256, 20]
        gamma = self.soft_plus(pre_gamma)  # [256, 20]

        return beta_0, gamma

    # noinspection DuplicatedCode
    def probability_forecast(self, train_batch, labels_batch=None, sample=False,
                             probability_range=None):  # [256, 112, 7], [256, 112,]
        if probability_range is None:
            probability_range = [0.5]

        batch_size = train_batch.shape[0]  # 256
        device = train_batch.device

        assert isinstance(probability_range, list)
        probability_range_len = len(probability_range)
        probability_range = torch.Tensor(probability_range).to(device)  # [3]

        train_batch = train_batch.permute(1, 0, 2)  # [112, 256, 7]
        if labels_batch is not None:
            labels_batch = labels_batch.permute(1, 0)  # [112, 256]

        # hidden and cell are initialized to zero
        hidden = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)  # [2, 256, 40]
        cell = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)  # [2, 256, 40]

        if labels_batch is not None:
            # train mode or validate mode
            hidden_permutes = torch.zeros(batch_size, self.train_window, self.lstm_hidden_dim * self.lstm_layers,
                                          device=device)
            for t in range(self.train_window):
                hidden, cell = self.run_lstm(train_batch[t].unsqueeze_(0).clone(), hidden, cell)
                hidden_permute = self.get_hidden_permute(hidden)
                hidden_permutes[:, t, :] = hidden_permute

                # check if hidden contains NaN
                if torch.isnan(hidden).sum() > 0:
                    break

            # get loss list
            stop_flag = False
            loss_list = []
            for t in range(self.train_window):
                hidden_permute = hidden_permutes[:, t, :]  # [256, 80]
                if torch.isnan(hidden_permute).sum() > 0:
                    stop_flag = True
                    break
                beta_0, gamma = self.get_qsqm_parameter(hidden_permute)  # [256, 1], [256, 20]
                loss_list.append((beta_0, gamma, labels_batch[t].clone()))

            return loss_list, stop_flag
        else:
            # test mode
            # initialize cdf range
            min_cdf = torch.Tensor([0.0]).to(device)  # [1]
            max_cdf = torch.Tensor([1.0]).to(device)  # [1]
            cdf_low = (1 - probability_range) / 2  # [3]
            cdf_high = 1 - (1 - probability_range) / 2  # [3]
            low_cdf = cdf_low.unsqueeze(0).expand(batch_size, -1)  # [256, 3]
            high_cdf = cdf_high.unsqueeze(0).expand(batch_size, -1)  # [256, 3]

            # initialize samples
            samples_low = torch.zeros(probability_range_len, batch_size, self.pred_steps, device=device)  # [3, 256, 16]
            samples_high = samples_low.clone()  # [3, 256, 16]
            samples = torch.zeros(self.sample_times, batch_size, self.pred_steps, device=device)  # [99, 256, 12]

            # condition range
            for t in range(self.pred_start):
                hidden, cell = self.run_lstm(train_batch[t].unsqueeze(0), hidden, cell)  # [2, 256, 40], [2, 256, 40]
            hidden_init = hidden.clone()
            cell_init = cell.clone()

            for j in range(self.sample_times + probability_range_len * 2):
                # clone test batch
                test_batch = train_batch.clone()  # [112, 256, 7]

                # initialize hidden and cell
                hidden, cell = hidden_init.clone(), cell_init.clone()

                # prediction range
                for t in range(self.pred_steps):
                    hidden, cell = self.run_lstm(test_batch[self.pred_start + t].unsqueeze(0), hidden, cell)
                    hidden_permute = self.get_hidden_permute(hidden)
                    beta_0, gamma = self.get_qsqm_parameter(hidden_permute)

                    if j < probability_range_len:
                        pred_cdf = low_cdf[:, j].unsqueeze(-1)  # [256, 1]
                    elif j < 2 * probability_range_len:
                        pred_cdf = high_cdf[:, j - probability_range_len].unsqueeze(-1)  # [256, 1]
                    else:
                        # pred_cdf is a uniform distribution
                        uniform = torch.distributions.uniform.Uniform(
                            torch.tensor([0.0], device=device),
                            torch.tensor([1.0], device=device))
                        pred_cdf = uniform.sample(torch.Size([batch_size]))  # [256, 1]

                    pred = sample_qsqm(beta_0, gamma, pred_cdf, min_cdf, max_cdf)
                    if j < probability_range_len:
                        samples_low[j, :, t] = pred
                    elif j < 2 * probability_range_len:
                        samples_high[j - probability_range_len, :, t] = pred
                    else:
                        samples[j - probability_range_len * 2, :, t] = pred

                    # predict value at t-1 is as a covars for t,t+1,...,t+lag
                    # for lag in range(self.lag):
                    #     z = self.lag - lag
                    #     if self.pred_start + t + z < self.train_window:
                    #         test_batch[self.pred_start + t + z, :, self.lag_index[lag]] = pred
                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            test_batch[self.pred_start + t + 1, :, self.new_index[0]] = pred

            samples_mu = torch.mean(samples, dim=0).unsqueeze(-1)  # mean or median ? # [256, 12, 1]
            samples_std = samples.std(dim=0).unsqueeze(-1)  # [256, 12, 1]

            # use integral to calculate the mean
            if not sample:
                # hidden and cell are initialized to zero
                hidden = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
                cell = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)

                # condition range
                test_batch = train_batch.clone()
                for t in range(self.pred_start):
                    hidden, cell = self.run_lstm(test_batch[t].unsqueeze(0), hidden, cell)

                # prediction range
                # sample
                samples_mu = torch.zeros(batch_size, self.pred_steps, 1, device=device)

                for t in range(self.pred_steps):
                    hidden, cell = self.run_lstm(test_batch[self.pred_start + t].unsqueeze(0), hidden, cell)
                    hidden_permute = self.get_hidden_permute(hidden)
                    beta_0, gamma = self.get_qsqm_parameter(hidden_permute)

                    pred = sample_qsqm(beta_0, gamma, None, min_cdf, max_cdf)
                    samples_mu[:, t, 0] = pred

                    # predict value at t-1 is as a covars for t,t+1,...,t+lag
                    # for lag in range(self.lag):
                    #     z = self.lag - lag
                    #     if self.pred_start + t + z < self.train_window:
                    #         test_batch[self.pred_start + t + z, :, self.lag_index[lag]] = pred
                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            test_batch[self.pred_start + t + 1, :, self.new_index[0]] = pred

            return samples, samples_mu, samples_std, samples_high, samples_low


def phase_beta_0_and_gamma(beta_0, gamma):
    sigma = torch.full_like(gamma, 1.0 / gamma.shape[1])  # [256, 20]

    beta = pad(gamma, (1, 0))[:, :-1]  # [256, 20]
    beta[:, 0] = beta_0[:, 0]
    beta = (gamma - beta) / (2 * sigma)
    beta = beta - pad(beta, (1, 0))[:, :-1]
    beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)  # [256, 20]
    ksi = torch.cumsum(sigma, dim=1)  # [256, 20]

    return sigma, beta, ksi


def sample_qsqm(beta_0, gamma, pred_cdf, min_cdf, max_cdf):
    sigma, beta, ksi = phase_beta_0_and_gamma(beta_0, gamma)  # [256, 20], [256, 20]
    ksi = pad(ksi, (1, 0))[:, :-1]  # [256, 20]

    if pred_cdf is not None:
        indices = ksi < pred_cdf  # [256, 20] # if smaller than pred_cdf, True
        # Q(alpha) = beta_0 * pred_cdf + sum(beta * (pred_cdf - ksi) ^ 2)
        pred = (beta_0 * pred_cdf).sum(dim=1)  # [256,]
        pred = pred + ((pred_cdf - ksi).pow(2) * beta * indices).sum(dim=1)  # [256,]

        return pred
    else:
        # get min pred and max pred
        # indices = ksi < min_cdf  # [256, 20] # True
        # min_pred = (beta_0 * min_cdf).sum(dim=1)  # [256,]
        # min_pred = min_pred + ((min_cdf - ksi).pow(2) * beta * indices).sum(dim=1)  # [256,]
        # indices = ksi < max_cdf  # [256, 20] # True
        # max_pred = (beta_0 * max_cdf).sum(dim=1)  # [256,]
        # max_pred = max_pred + ((max_cdf - ksi).pow(2) * beta * indices).sum(dim=1)  # [256,]
        # total_area = ((max_cdf - min_cdf) * (max_pred - min_pred))  # [256,]

        # calculate integral
        # itg Q(alpha) = 1/2 * beta_0 * (max_cdf ^ 2 - min_cdf ^ 2) + sum(1/3 * beta * (max_cdf - ksi) ^ 3)
        integral1 = 0.5 * beta_0.squeeze() * (max_cdf.pow(2) - min_cdf.pow(2))  # [256,]
        integral2 = 1 / 3 * ((max_cdf - ksi).pow(3) * beta).sum(dim=1)  # [256,]
        integral = integral1 + integral2  # [256,]
        pred_mu = integral / (max_cdf - min_cdf)  # [256,]

        return pred_mu


def loss_fn(list_param, crps=True):
    beta_0, gamma, labels = list_param  # [256, 1], [256, 20], [256,]

    if not crps:
        # MSE
        mseLoss = get_mse(beta_0, gamma, labels)

        return mseLoss
    else:
        # CRPS
        labels = labels.unsqueeze(1)  # [256, 1]
        crpsLoss = get_crps(beta_0, gamma, labels)

        return crpsLoss


def get_mse(beta_0, gamma, labels):
    device = beta_0.device
    min_cdf = torch.Tensor([0]).to(device)  # [256, 1]
    max_cdf = torch.Tensor([1]).to(device)  # [256, 1]

    sigma, beta, ksi = phase_beta_0_and_gamma(beta_0, gamma)  # [256, 20], [256, 20]
    ksi = pad(ksi, (1, 0))[:, :-1]  # [256, 20]

    # calculate integral
    # itg Q(alpha) = 1/2 * beta_0 * (max_cdf ^ 2 - min_cdf ^ 2) + sum(1/3 * beta * (max_cdf - ksi) ^ 3)
    integral1 = 0.5 * beta_0.squeeze() * (max_cdf.pow(2) - min_cdf.pow(2))  # [256,]
    integral2 = 1 / 3 * ((max_cdf - ksi).pow(3) * beta).sum(dim=1)  # [256,]
    integral = integral1 + integral2  # [256,]
    pred = integral / (max_cdf - min_cdf)  # [256,]

    loss = nn.MSELoss()
    mseLoss = loss(pred, labels)

    return mseLoss


def get_crps(beta_0, gamma, labels):
    sigma, beta, ksi = phase_beta_0_and_gamma(beta_0, gamma)  # [256, 20], [256, 20]

    # calculate the maximum for each segment of the spline
    df1 = ksi.expand(sigma.shape[1], sigma.shape[0], sigma.shape[1]).T.clone()
    df2 = ksi.T.unsqueeze(2)
    ksi = pad(ksi, (1, 0))[:, :-1]
    knots = df1 - ksi
    knots[knots < 0] = 0
    knots = (df2 * beta_0).sum(dim=2) + (knots.pow(2) * beta).sum(dim=2)
    knots = pad(knots.T, (1, 0))[:, :-1]  # F(ksi_1~K)=0~max

    diff = labels - knots
    labels = labels.squeeze()
    alpha_l = diff > 0
    alpha_A = torch.sum(alpha_l * beta, dim=1)
    alpha_B = beta_0[:, 0] - 2 * torch.sum(alpha_l * beta * ksi, dim=1)
    alpha_C = -labels + torch.sum(alpha_l * beta * ksi * ksi, dim=1)

    # since A may be zero, roots can be from different methods.
    not_zero = (alpha_A != 0)
    alpha = torch.zeros_like(alpha_A)
    # since there may be numerical calculation error,#0
    idx = (alpha_B ** 2 - 4 * alpha_A * alpha_C) < 0  # 0
    diff = diff.abs()
    index = diff == (diff.min(dim=1)[0].view(-1, 1))
    index[~idx, :] = False
    # index=diff.abs()<1e-4#0,1e-4 is a threshold
    # idx=index.sum(dim=1)>0#0
    alpha[idx] = ksi[index]  # 0
    alpha[~not_zero] = -alpha_C[~not_zero] / alpha_B[~not_zero]
    not_zero = ~(~not_zero | idx)  # 0
    delta = alpha_B[not_zero].pow(2) - 4 * alpha_A[not_zero] * alpha_C[not_zero]
    alpha[not_zero] = (-alpha_B[not_zero] + torch.sqrt(delta)) / (2 * alpha_A[not_zero])

    # formula for CRPS is here!
    gamma_0 = torch.zeros_like(labels)
    crps_1 = (gamma_0 - labels) * (1 - 2 * alpha)
    crps_2 = beta_0[:, 0] * (1 / 3 - alpha.pow(2))
    crps_3 = torch.sum(beta / 6 * (1 - ksi).pow(4), dim=1)
    crps_4 = torch.sum(alpha_l * 2 / 3 * beta * (alpha.unsqueeze(1) - ksi).pow(3), dim=1)
    crps = crps_1 + crps_2 + crps_3 - crps_4

    crps = torch.mean(crps)
    return crps
