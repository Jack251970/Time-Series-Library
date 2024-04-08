import torch
import torch.nn as nn
from torch.nn.functional import pad

from models.QSQF.net_qspline_C import ConvLayer


class Model(nn.Module):
    def __init__(self, params, use_cnn=True, use_qrnn=False):
        """
        LSTM-CQ: Auto-Regressive LSTM with Convolution and QSpline to Provide Probabilistic Forecasting.

        params: parameters for the model.
        use_cnn: whether to use cnn for feature extraction.
        use_qrnn: whether to use qrnn to replace lstm.
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
        self.linear_gamma = nn.Linear(self.lstm_hidden_dim * self.lstm_layers, 1)
        self.linear_eta_k = nn.Linear(self.lstm_hidden_dim * self.lstm_layers, self.num_spline)
        self.soft_plus = nn.Softplus()  # make sure parameter is positive

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

    # noinspection DuplicatedCode
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
        candidate_gamma = self.linear_gamma(hidden_permute)  # [256, 1]
        gamma = self.soft_plus(candidate_gamma)  # [256, 1]
        candidate_eta_k = self.linear_eta_k(hidden_permute)  # [256, 20]
        eta_k = self.soft_plus(candidate_eta_k)  # [256, 20]

        return gamma, eta_k

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
                gamma, eta_k = self.get_qsqm_parameter(hidden_permute)  # [256, 1], [256, 20]
                y = labels_batch[t].clone()  # [256,]
                loss_list.append((gamma, eta_k, y))

            return loss_list, stop_flag
        else:
            # test mode
            # initialize cdf range
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
                    gamma, eta_k = self.get_qsqm_parameter(hidden_permute)

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

                    pred = sample_qsqm(gamma, eta_k, pred_cdf)
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
                    gamma, eta_k = self.get_qsqm_parameter(hidden_permute)

                    pred = sample_qsqm(gamma, eta_k, None)
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


def phase_gamma_and_eta_k(gamma, eta_k):
    """
    Formula
    beta_k = (eta_k - gamma) / (2 * alpha_prime_k), k = 1
    beta_k = (eta_k - eta_{k-1}) / (2 * alpha_prime_k) - (eta_{k-1} - eta_{k-2}) / (2 * alpha_prime_{k-1}), k > 1
    let x_k = (eta_k - gamma) / (2 * alpha_prime_k), and x_k = 0, and then beta_k = x_k - x_{k-1}
    """
    # use interval based on uniform distribution
    alpha_prime_k = torch.full_like(eta_k, 1.0 / eta_k.shape[1])  # [256, 20]

    # get alpha_k ([0, k])
    alpha_0_k = pad(torch.cumsum(alpha_prime_k, dim=1), pad=(1, 0))[:, :-1]  # [256, 20]

    # get x_k
    x_k = pad(eta_k, pad=(1, 0))[:, :-1]  # [256, 20]
    x_k[:, 0] = gamma[:, 0]
    x_k = (eta_k - x_k) / (2 * alpha_prime_k)

    # get beta_k
    beta_k = x_k - pad(x_k, pad=(1, 0))[:, :-1]  # [256, 20]

    # TODO: Check if need it?
    beta_k[:, -1] = eta_k[:, -1] - beta_k[:, :-1].sum(dim=1)  # [256, 20]

    return alpha_prime_k, alpha_0_k, beta_k


# noinspection DuplicatedCode
def get_y_hat(gamma, alpha_0_k, beta_k):
    """
    Formula
    int{Q(alpha)} = 1/2 * gamma * (max_alpha ^ 2 - min_alpha ^ 2) + sum(1/3 * beta_k * (max_alpha - ksi) ^ 3)
    y_hat = int{Q(alpha)} / (max_alpha - min_alpha)
    """
    # init min_alpha and max_alpha
    device = gamma.device
    min_alpha = torch.Tensor([0]).to(device)  # [1]
    max_alpha = torch.Tensor([1]).to(device)  # [1]

    # get min pred and max pred
    # indices = alpha_0_k < min_alpha  # [256, 20]
    # min_pred = (gamma * min_alpha).sum(dim=1)  # [256,]
    # min_pred = min_pred + ((min_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    # indices = alpha_0_k < max_alpha  # [256, 20]
    # max_pred = (gamma * max_alpha).sum(dim=1)  # [256,]
    # max_pred = max_pred + ((max_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    # total_area = ((max_alpha - min_alpha) * (max_pred - min_pred))  # [256,]

    # get int{Q(alpha)}
    integral1 = 0.5 * gamma.squeeze() * (max_alpha.pow(2) - min_alpha.pow(2))  # [256,]
    integral2 = 1 / 3 * ((max_alpha - alpha_0_k).pow(3) * beta_k).sum(dim=1)  # [256,]
    integral = integral1 + integral2  # [256,]
    y_hat = integral / (max_alpha - min_alpha)  # [256,]

    return y_hat


# noinspection DuplicatedCode
def sample_qsqm(gamma, eta_k, alpha):
    """
    Formula
    Q(alpha) = gamma * alpha + sum(beta_k * (alpha - alpha_k) ^ 2)
    """
    alpha_prime_k, alpha_0_k, beta_k = phase_gamma_and_eta_k(gamma, eta_k)  # [256, 20], [256, 20], [256, 20]

    if alpha is not None:
        # get Q(alpha)
        indices = alpha_0_k < alpha  # [256, 20]
        pred1 = (gamma * alpha).sum(dim=1)  # [256,]
        pred2 = (beta_k * (alpha - alpha_0_k).pow(2) * indices).sum(dim=1)  # [256,]
        pred = pred1 + pred2  # [256,]

        return pred
    else:
        # get pred mean value
        y_hat = get_y_hat(gamma, alpha_0_k, beta_k)  # [256,]

        return y_hat


def loss_fn(list_param, crps=True):
    gamma, eta_k, labels = list_param  # [256, 1], [256, 20], [256,]

    if not crps:
        # MSE
        return get_mse(gamma, eta_k, labels)
    else:
        # CRPS
        labels = labels.unsqueeze(1)  # [256, 1]
        return get_crps(torch.zeros_like(labels).to(labels.device), gamma, eta_k, labels)


def get_mse(gamma, eta_k, y):
    alpha_prime_k, alpha_0_k, beta_k = phase_gamma_and_eta_k(gamma, eta_k)  # [256, 20], [256, 20]

    # get y_hat
    y_hat = get_y_hat(gamma, alpha_0_k, beta_k)  # [256,]

    # calculate loss
    loss = nn.MSELoss()
    mseLoss = loss(y_hat, y)

    return mseLoss


# noinspection DuplicatedCode
def get_crps(_lambda, gamma, eta_k, y):
    alpha_prime_k, alpha_0_k, beta_k = phase_gamma_and_eta_k(gamma, eta_k)  # [256, 20], [256, 20]
    alpha_1_k1 = pad(alpha_0_k, pad=(0, 1), value=1)[:, 1:]  # [256, 20]

    # calculate the maximum for each segment of the spline
    df1 = alpha_1_k1.expand(alpha_prime_k.shape[1], alpha_prime_k.shape[0], alpha_prime_k.shape[1]).T.clone()
    df2 = alpha_1_k1.T.unsqueeze(2)
    alpha_0_k = pad(alpha_1_k1, (1, 0))[:, :-1]
    knots = df1 - alpha_0_k
    knots[knots < 0] = 0
    knots = (df2 * gamma).sum(dim=2) + (knots.pow(2) * beta_k).sum(dim=2)
    knots = pad(knots.T, (1, 0))[:, :-1]  # F(ksi_1~K)=0~max

    diff = y - knots
    y = y.squeeze()
    alpha_l = diff > 0
    alpha_A = torch.sum(alpha_l * beta_k, dim=1)
    alpha_B = gamma[:, 0] - 2 * torch.sum(alpha_l * beta_k * alpha_0_k, dim=1)
    alpha_C = -y + torch.sum(alpha_l * beta_k * alpha_0_k * alpha_0_k, dim=1)

    # since A may be zero, roots can be from different methods.
    not_zero = (alpha_A != 0)
    alpha_plus = torch.zeros_like(alpha_A)
    # since there may be numerical calculation error,#0
    idx = (alpha_B ** 2 - 4 * alpha_A * alpha_C) < 0  # 0
    diff = diff.abs()
    index = diff == (diff.min(dim=1)[0].view(-1, 1))
    index[~idx, :] = False
    # index=diff.abs()<1e-4#0,1e-4 is a threshold
    # idx=index.sum(dim=1)>0#0
    alpha_plus[idx] = alpha_0_k[index]  # 0
    alpha_plus[~not_zero] = -alpha_C[~not_zero] / alpha_B[~not_zero]
    not_zero = ~(~not_zero | idx)  # 0
    delta = alpha_B[not_zero].pow(2) - 4 * alpha_A[not_zero] * alpha_C[not_zero]
    alpha_plus[not_zero] = (-alpha_B[not_zero] + torch.sqrt(delta)) / (2 * alpha_A[not_zero])

    # formula for CRPS is here!
    crps_1 = (_lambda - y) * (1 - 2 * alpha_plus)
    crps_2 = gamma[:, 0] * (1 / 3 - alpha_plus.pow(2))
    crps_3 = torch.sum(beta_k / 6 * (1 - alpha_0_k).pow(4), dim=1)
    crps_4 = torch.sum(alpha_l * 2 / 3 * beta_k * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(3), dim=1)
    crps = crps_1 + crps_2 + crps_3 - crps_4

    crps = torch.mean(crps)
    return crps
