import torch
import torch.nn as nn
from torch.nn.functional import pad

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:52:22 2020

@author: 18096
"""

'''Defines the neural network, loss function and metrics'''


class Model(nn.Module):
    def __init__(self, params):
        """
        We define a recurrent network that predicts the future values
        of a time-dependent variable based on past inputs and covariances.
        """
        super(Model, self).__init__()
        self.task_name = params.task_name
        self.lstm_input_size = params.enc_in + params.lag - 1  # take lag dimension into account
        self.lstm_hidden_dim = 40
        self.lstm_layers = 2
        self.sample_times = params.sample_times
        self.lstm_dropout = params.dropout
        self.num_spline = params.num_spline
        self.pred_start = params.seq_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start

        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_dim,
                            num_layers=self.lstm_layers,
                            bias=True,
                            batch_first=False,
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

        # Plan C:
        self.pre_beta_0 = nn.Linear(self.lstm_hidden_dim * self.lstm_layers, 1)
        self.pre_gamma = nn.Linear(self.lstm_hidden_dim * self.lstm_layers, self.num_spline)

        self.beta_0 = nn.Softplus()
        # soft-plus to make sure gamma is positive
        # self.gamma=nn.ReLU()
        self.gamma = nn.Softplus()

    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'probability_forecast':
            # we don't need to use mark data because lstm can handle time series relation information
            batch = torch.cat((x_enc, y_enc), dim=1).float()
            train_batch = batch[:, :, :-1]
            labels_batch = batch[:, :, -1]
            return self.probability_forecast(train_batch, labels_batch)  # return loss list
        return None

    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, probability_range=0.95):
        if self.task_name == 'probability_forecast':
            batch = torch.cat((x_enc, y_enc), dim=1).float()
            train_batch = batch[:, :, :-1]
            return self.probability_forecast(train_batch, probability_range=probability_range)
        return None

    def probability_forecast(self, train_batch, labels_batch=None, sample=False, probability_range=0.4):  # [256, 108, 7], [256, 108,]
        batch_size = train_batch.shape[0]  # 256
        device = train_batch.device

        train_batch = train_batch.permute(1, 0, 2)  # [108, 256, 7]
        if labels_batch is not None:
            labels_batch = labels_batch.permute(1, 0)  # [108, 256]

        # hidden and cell are initialized to zero
        hidden = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)  # [2, 256, 40]
        cell = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)  # [2, 256, 40]

        if labels_batch is not None:
            # train mode or validate mode
            hidden_permutes = torch.zeros(batch_size, self.train_window, self.lstm_hidden_dim * self.lstm_layers, device=device)
            for t in range(self.train_window):
                # {[256, 1], [256, 20]}, [2, 256, 40], [2, 256, 40]
                x = train_batch[t].unsqueeze_(0).clone()  # [1, 256, 7]

                _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]
                # use h from all three layers to calculate mu and sigma
                hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)  # [256, 80]
                hidden_permutes[:, t - self.pred_start, :] = hidden_permute

                # check if hidden contains NaN
                if torch.isnan(hidden).sum() > 0:
                    raise ValueError(f'Backward Error! Process Stop!')

            # get loss list
            loss_list = []
            for t in range(self.train_window):
                hidden_permute = hidden_permutes[:, t, :]  # [256, 80]

                # Plan C:
                pre_beta_0 = self.pre_beta_0(hidden_permute)  # [256, 1]
                beta_0 = self.beta_0(pre_beta_0)  # [256, 1]
                pre_gamma = self.pre_gamma(hidden_permute)  # [256, 20]
                gamma = self.gamma(pre_gamma)  # [256, 20]

                loss_list.append((beta_0, gamma, labels_batch[t].clone()))

            return loss_list
        else:
            # test mode
            # condition range
            test_batch = train_batch  # [108, 256, 7]
            for t in range(self.pred_start):
                x = test_batch[t].unsqueeze(0)  # [1, 256, 7]

                _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

            # prediction range
            cdf_high = 1 - (1 - probability_range) / 2
            cdf_low = (1 - probability_range) / 2

            # sample
            samples_high = torch.zeros(1, batch_size, self.pred_steps, device=device, requires_grad=False)  # [1, 256, 16]
            samples_low = torch.zeros(1, batch_size, self.pred_steps, device=device, requires_grad=False)  # [1, 256, 16]
            samples = torch.zeros(self.sample_times, batch_size, self.pred_steps, device=device)  # [99, 256, 12]
            for j in range(self.sample_times + 2):
                hidden_permutes = torch.zeros(batch_size, self.pred_steps, self.lstm_hidden_dim * self.lstm_layers, device=device)
                for t in range(self.pred_steps):
                    x = test_batch[self.pred_start + t].unsqueeze(0)  # [1, 256, 7]

                    _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]
                    # use h from all three layers to calculate mu and sigma
                    hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)  # [256, 80]
                    hidden_permutes[:, t, :] = hidden_permute

                    # Plan C:
                    pre_beta_0 = self.pre_beta_0(hidden_permute)  # [256, 1]
                    beta_0 = self.beta_0(pre_beta_0)  # [256, 1]
                    pre_gamma = self.pre_gamma(hidden_permute)  # [256, 20]
                    gamma = self.gamma(pre_gamma)  # [256, 20]

                    # pred_cdf is a uniform distribution
                    if j == 0:  # high
                        pred_cdf = torch.Tensor([cdf_high]).to(device)
                    elif j == 1:  # low
                        pred_cdf = torch.Tensor([cdf_low]).to(device)
                    else:
                        uniform = torch.distributions.uniform.Uniform(
                            torch.tensor([0.0], device=device),
                            torch.tensor([1.0], device=device))
                        pred_cdf = uniform.sample([batch_size])  # [256, 1]

                    # Plan C
                    sigma = torch.full_like(gamma, 1.0 / gamma.shape[1])  # [256, 20]
                    beta = pad(gamma, (1, 0))[:, :-1]  # [256, 20]
                    beta[:, 0] = beta_0[:, 0]
                    beta = (gamma - beta) / (2 * sigma)
                    beta = beta - pad(beta, (1, 0))[:, :-1]
                    beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)  # [256, 20]

                    ksi = pad(torch.cumsum(sigma, dim=1), (1, 0))[:, :-1]  # [256, 20]
                    indices = ksi < pred_cdf  # [256, 20] # if smaller than pred_cdf, True
                    # Q(alpha) = beta_0 * pred_cdf + sum(beta * (pred_cdf - ksi) ^ 2)
                    pred = (beta_0 * pred_cdf).sum(dim=1)  # [256,]
                    pred = pred + ((pred_cdf - ksi).pow(2) * beta * indices).sum(dim=1)  # [256,] # Q(alpha)公式?

                    if j == 0:
                        samples_high[0, :, t] = pred
                    elif j == 1:
                        samples_low[0, :, t] = pred
                    else:
                        samples[j - 2, :, t] = pred

                    # predict value at t-1 is as a covars for t,t+1,...,t+lag
                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            test_batch[self.pred_start + t + 1, :, 0] = pred

            samples_mu = torch.mean(samples, dim=0).unsqueeze(-1)  # mean or median ? # [256, 12, 1]
            samples_std = samples.std(dim=0).unsqueeze(-1)  # [256, 12, 1]

            if not sample:
                samples_mu = torch.zeros(batch_size, self.pred_steps, 1, device=device)

                for t in range(self.pred_steps):
                    x = test_batch[self.pred_start + t].unsqueeze(0)  # [1, 256, 7]

                    _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]
                    # use h from all three layers to calculate mu and sigma
                    hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)  # [256, 80]

                    # Plan C:
                    pre_beta_0 = self.pre_beta_0(hidden_permute)  # [256, 1]
                    beta_0 = self.beta_0(pre_beta_0)  # [256, 1]
                    pre_gamma = self.pre_gamma(hidden_permute)  # [256, 20]
                    gamma = self.gamma(pre_gamma)  # [256, 20]

                    # Plan C
                    min_cdf = torch.Tensor([0]).to(device)  # [256, 1]
                    max_cdf = torch.Tensor([1]).to(device)  # [256, 1]

                    sigma = torch.full_like(gamma, 1.0 / gamma.shape[1])  # [256, 20]
                    beta = pad(gamma, (1, 0))[:, :-1]  # [256, 20]
                    beta[:, 0] = beta_0[:, 0]
                    beta = (gamma - beta) / (2 * sigma)
                    beta = beta - pad(beta, (1, 0))[:, :-1]
                    beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)  # [256, 20]
                    ksi = pad(torch.cumsum(sigma, dim=1), (1, 0))[:, :-1]  # [256, 20]

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

                    samples_mu[:, t, 0] = pred_mu

                    # predict value at t-1 is as a covars for t,t+1,...,t+lag
                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            test_batch[self.pred_start + t + 1, :, 0] = pred_mu

            return samples, samples_mu, samples_std, samples_high, samples_low


def loss_fn(list_param):
    beta_0, gamma, labels = list_param  # [256, 1], [256, 20], [256,]

    # Plan C
    device = beta_0.device
    min_cdf = torch.Tensor([0]).to(device)  # [256, 1]
    max_cdf = torch.Tensor([1]).to(device)  # [256, 1]

    sigma = torch.full_like(gamma, 1.0 / gamma.shape[1])  # [256, 20]
    beta = pad(gamma, (1, 0))[:, :-1]  # [256, 20]
    beta[:, 0] = beta_0[:, 0]
    beta = (gamma - beta) / (2 * sigma)
    beta = beta - pad(beta, (1, 0))[:, :-1]
    beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)  # [256, 20]
    ksi = pad(torch.cumsum(sigma, dim=1), (1, 0))[:, :-1]  # [256, 20]

    # calculate integral
    # itg Q(alpha) = 1/2 * beta_0 * (max_cdf ^ 2 - min_cdf ^ 2) + sum(1/3 * beta * (max_cdf - ksi) ^ 3)
    integral1 = 0.5 * beta_0.squeeze() * (max_cdf.pow(2) - min_cdf.pow(2))  # [256,]
    integral2 = 1 / 3 * ((max_cdf - ksi).pow(3) * beta).sum(dim=1)  # [256,]
    integral = integral1 + integral2  # [256,]
    pred = integral / (max_cdf - min_cdf)  # [256,]

    loss = nn.MSELoss()
    mseLoss = loss(pred, labels)

    labels = labels.unsqueeze(1)  # [256, 1]
    crpsLoss = get_crps(beta_0, gamma, labels)

    return crpsLoss


def get_crps(beta_0, gamma, labels):
    sigma = torch.full_like(gamma, 1.0 / gamma.shape[1], requires_grad=False)  # [256, 1], [256, 20]

    beta = pad(gamma, (1, 0))[:, :-1]  # [256, 20]
    beta[:, 0] = beta_0[:, 0]
    beta = (gamma - beta) / (2 * sigma)
    beta = beta - pad(beta, (1, 0))[:, :-1]
    beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)

    # calculate the maximum for each segment of the spline
    ksi = torch.cumsum(sigma, dim=1)
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