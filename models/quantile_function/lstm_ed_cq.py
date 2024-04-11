import torch
import torch.nn as nn
from torch.nn.functional import pad

from models.quantile_function.lstm_cq import ConvLayer


class Model(nn.Module):
    def __init__(self, params, algorithm_type="1+2"):
        """
        LSTM-ED-CQ: Auto-Regressive LSTM based on encoder-decoder architecture with convolution and spline to provide
        probabilistic forecasting.

        params: parameters for the model.
        algorithm_type: algorithm type, e.g. '1', '2', '1+2'
        """
        super(Model, self).__init__()
        self.algorithm_type = algorithm_type
        self.task_name = params.task_name
        self.encoder_lstm_input_size = params.enc_in - 1
        self.decoder_lstm_input_size = self.lag  # TODO: Check params.enc_in + params.lag - 1!!
        self.lstm_hidden_size = params.lstm_hidden_size
        self.encoder_lstm_layers = params.lstm_layers
        self.decoder_lstm_layers = 1  # TODO: Check self.encoder_lstm_layers!!
        self.sample_times = params.sample_times
        self.lstm_dropout = params.dropout
        self.num_spline = params.num_spline
        self.pred_start = params.seq_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start

        # LSTM
        self.lstm_encoder = nn.LSTM(input_size=self.encoder_lstm_input_size,
                                    hidden_size=self.lstm_hidden_size,
                                    num_layers=self.encoder_lstm_layers,
                                    bias=True,
                                    batch_first=False,
                                    bidirectional=False,
                                    dropout=self.lstm_dropout)
        self.lstm_decoder = nn.LSTM(input_size=self.decoder_lstm_input_size,
                                    hidden_size=self.lstm_hidden_size,
                                    num_layers=self.decoder_lstm_layers,
                                    bias=True,
                                    batch_first=False,
                                    bidirectional=False,
                                    dropout=self.lstm_dropout)
        self.init_lstm(self.lstm_encoder)
        self.init_lstm(self.lstm_decoder)

        # QSQM
        self._lambda = -1e-3  # make sure all data is not on the left point
        if self.algorithm_type == '2':
            self.linear_gamma = nn.Linear(self.lstm_hidden_size * self.decoder_lstm_input_size, 1)
        elif self.algorithm_type == '1+2':
            self.linear_gamma = nn.Linear(self.lstm_hidden_size * self.decoder_lstm_input_size, self.num_spline)
        elif self.algorithm_type == '1':
            self.linear_gamma = nn.Linear(self.lstm_hidden_size * self.decoder_lstm_input_size, self.num_spline)
        else:
            raise ValueError("algorithm_type must be '1', '2', or '1+2'")
        self.linear_eta_k = nn.Linear(self.lstm_hidden_size * self.decoder_lstm_input_size, self.num_spline)
        self.soft_plus = nn.Softplus()  # make sure parameter is positive
        device = torch.device("cuda" if params.use_gpu else "cpu")
        y = torch.ones(self.num_spline) / self.num_spline
        self.alpha_prime_k = y.repeat(params.batch_size, 1).to(device)  # [256, 20]

        # Reindex
        self.new_index = [0]

    @staticmethod
    def init_lstm(lstm):
        # initialize LSTM forget gate bias to be 1 as recommended by
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        # noinspection PyProtectedMember
        for names in lstm._all_weights:
            for name in filter(lambda _n: "bias" in _n, names):
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    # noinspection DuplicatedCode,PyUnusedLocal
    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'probability_forecast':
            # we don't need to use mark data because lstm can handle time series relation information
            batch = torch.cat((x_enc, y_enc), dim=1).float()
            train_batch = batch[:, :, :-1]
            labels_batch = batch[:, :, -1]
            return self.probability_forecast(train_batch, labels_batch)  # return loss list
        return None

    # noinspection PyUnusedLocal
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
        _, (hidden, cell) = self.lstm_encoder(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

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
        if self.algorithm_type == '1':
            return gamma, None

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
        hidden = torch.zeros(self.encoder_lstm_layers, batch_size, self.lstm_hidden_size, device=device)  # [2, 256, 40]
        cell = torch.zeros(self.encoder_lstm_layers, batch_size, self.lstm_hidden_size, device=device)  # [2, 256, 40]

        if labels_batch is not None:
            # train mode or validate mode
            hidden_permutes = torch.zeros(batch_size, self.train_window, self.lstm_hidden_size * self.encoder_lstm_layers,
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
                    loss_list.clear()
                    stop_flag = True
                    break
                gamma, eta_k = self.get_qsqm_parameter(hidden_permute)  # [256, 20], [256, 20]
                y = labels_batch[t].clone()  # [256,]
                loss_list.append((self.alpha_prime_k, self._lambda, gamma, eta_k, y, self.algorithm_type))

            return loss_list, stop_flag
        else:
            # test mode
            # initialize alpha range
            alpha_low = (1 - probability_range) / 2  # [3]
            alpha_high = 1 - (1 - probability_range) / 2  # [3]
            low_alpha = alpha_low.unsqueeze(0).expand(batch_size, -1)  # [256, 3]
            high_alpha = alpha_high.unsqueeze(0).expand(batch_size, -1)  # [256, 3]

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
                        pred_alpha = low_alpha[:, j].unsqueeze(-1)  # [256, 1]
                    elif j < 2 * probability_range_len:
                        pred_alpha = high_alpha[:, j - probability_range_len].unsqueeze(-1)  # [256, 1]
                    else:
                        # pred alpha is a uniform distribution
                        uniform = torch.distributions.uniform.Uniform(
                            torch.tensor([0.0], device=device),
                            torch.tensor([1.0], device=device))
                        pred_alpha = uniform.sample(torch.Size([batch_size]))  # [256, 1]

                    pred = sample_qsqm(self.alpha_prime_k, pred_alpha, self._lambda, gamma, eta_k, self.algorithm_type)
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
                hidden = torch.zeros(self.encoder_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
                cell = torch.zeros(self.encoder_lstm_layers, batch_size, self.lstm_hidden_size, device=device)

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

                    pred = sample_qsqm(self.alpha_prime_k, None, self._lambda, gamma, eta_k, self.algorithm_type)
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


def phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type):
    """
    Formula
    2:
    beta_k = (eta_k - gamma) / (2 * alpha_prime_k), k = 1
    let x_k = (eta_k - eta_{k-1}) / (2 * alpha_prime_k), and x_k = 0, eta_0 = gamma
    beta_k = x_k - x_{k-1}, k > 1

    1+2:
    beta_k = (eta_k - gamma_1) / (2 * alpha_prime_k), k = 1
    let x_k = (eta_k - eta_{k-1} - gamma_k) / (2 * alpha_prime_k), and x_k = 0, eta_0 = 0
    beta_k = x_k - x_{k-1}, k > 1

    1:
    none
    """
    # get alpha_k ([0, k])
    alpha_0_k = pad(torch.cumsum(alpha_prime_k, dim=1), pad=(1, 0))[:, :-1]  # [256, 20]

    # get x_k
    if algorithm_type == '2':
        eta_0_1k = pad(eta_k, pad=(1, 0))[:, :-1]  # [256, 20]
        eta_0_1k[:, 0] = gamma[:, 0]  # eta_0 = gamma
        x_k = (eta_k - eta_0_1k) / (2 * alpha_prime_k)
    elif algorithm_type == '1+2':
        eta_0_1k = pad(eta_k, pad=(1, 0))[:, :-1]  # [256, 20]
        x_k = (eta_k - eta_0_1k - gamma) / (2 * alpha_prime_k)
    elif algorithm_type == '1':
        return alpha_0_k, None
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")

    # get beta_k
    beta_k = x_k - pad(x_k, pad=(1, 0))[:, :-1]  # [256, 20]

    return alpha_0_k, beta_k


# noinspection DuplicatedCode
def get_y_hat(alpha_0_k, _lambda, gamma, beta_k, algorithm_type):
    """
    Formula
    2:
    int{Q(alpha)} = lambda * (max_alpha - min_alpha) + 1/2 * gamma * (max_alpha ^ 2 - min_alpha ^ 2)
    + sum(1/3 * beta_k * (max_alpha - alpha_0_k) ^ 3)
    y_hat = int{Q(alpha)} / (max_alpha - min_alpha)

    1+2:
    int{Q(alpha)} = lambda * (max_alpha - min_alpha) + sum(1/2 * gamma_k * (max_alpha - alpha_0_k) ^ 2)
    + sum(1/3 * beta_k * (max_alpha - alpha_0_k) ^ 3)
    y_hat = int{Q(alpha)} / (max_alpha - min_alpha)

    1:
    int {Q(alpha)} = lambda * (max_alpha - min_alpha) + sum(1/2 * gamma_k * (max_alpha - alpha_0_k) ^ 2)
    y_hat = int{Q(alpha)} / (max_alpha - min_alpha)
    """
    # init min_alpha and max_alpha
    device = gamma.device
    min_alpha = torch.Tensor([0]).to(device)  # [1]
    max_alpha = torch.Tensor([1]).to(device)  # [1]

    # get min pred and max pred
    # indices = alpha_0_k < min_alpha  # [256, 20]
    # min_pred0 = _lambda
    # if algorithm_type == '2':
    #     min_pred1 = (min_alpha * gamma).sum(dim=1)  # [256,]
    #     min_pred2 = ((min_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    #     min_pred = min_pred0 + min_pred1 + min_pred2  # [256,]
    # elif algorithm_type == '1+2':
    #     min_pred1 = ((min_alpha - alpha_0_k) * gamma * indices).sum(dim=1)  # [256,]
    #     min_pred2 = ((min_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    #     min_pred = min_pred0 + min_pred1 + min_pred2  # [256,]
    # elif algorithm_type == '1':
    #     min_pred1 = ((min_alpha - alpha_0_k) * gamma * indices).sum(dim=1)  # [256,]
    #     min_pred = min_pred0 + min_pred1  # [256,]
    # else:
    #     raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    # indices = alpha_0_k < max_alpha  # [256, 20]
    # max_pred0 = _lambda
    # if algorithm_type == '2':
    #     max_pred1 = (max_alpha * gamma).sum(dim=1)  # [256,]
    #     max_pred2 = ((max_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    #     max_pred = max_pred0 + max_pred1 + max_pred2  # [256,]
    # elif algorithm_type == '1+2':
    #     max_pred1 = ((max_alpha - alpha_0_k) * gamma * indices).sum(dim=1)  # [256,]
    #     max_pred2 = ((max_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    #     max_pred = max_pred0 + max_pred1 + max_pred2  # [256,]
    # elif algorithm_type == '1':
    #     max_pred1 = ((max_alpha - alpha_0_k) * gamma * indices).sum(dim=1)  # [256,]
    #     max_pred = max_pred0 + max_pred1  # [256,]
    # else:
    #     raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    # total_area = ((max_alpha - min_alpha) * (max_pred - min_pred))  # [256,]

    # get int{Q(alpha)}
    if algorithm_type == '2':
        integral0 = _lambda * (max_alpha - min_alpha)
        integral1 = 1 / 2 * gamma.squeeze() * (max_alpha.pow(2) - min_alpha.pow(2))  # [256,]
        integral2 = 1 / 3 * ((max_alpha - alpha_0_k).pow(3) * beta_k).sum(dim=1)  # [256,]
        integral = integral0 + integral1 + integral2  # [256,]
    elif algorithm_type == '1+2':
        integral0 = _lambda * (max_alpha - min_alpha)
        integral1 = 1 / 2 * ((max_alpha - alpha_0_k).pow(2) * gamma).sum(dim=1)  # [256,]
        integral2 = 1 / 3 * ((max_alpha - alpha_0_k).pow(3) * beta_k).sum(dim=1)  # [256,]
        integral = integral0 + integral1 + integral2  # [256,]
    elif algorithm_type == '1':
        integral0 = _lambda * (max_alpha - min_alpha)
        integral1 = 1 / 2 * ((max_alpha - alpha_0_k).pow(2) * gamma).sum(dim=1)  # [256,]
        integral = integral0 + integral1  # [256,]
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    y_hat = integral / (max_alpha - min_alpha)  # [256,]

    return y_hat


# noinspection DuplicatedCode
def sample_qsqm(alpha_prime_k, alpha, _lambda, gamma, eta_k, algorithm_type):
    """
    Formula
    2:
    Q(alpha) = lambda + gamma * alpha + sum(beta_k * (alpha - alpha_k) ^ 2)

    1+2:
    Q(alpha) = lambda + sum(beta_k * (alpha - alpha_k)) + sum(beta_k * (alpha - alpha_k) ^ 2)

    1:
    Q(alpha) = lambda + sum(beta_k * (alpha - alpha_k))
    """
    alpha_0_k, beta_k = phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type)

    if alpha is not None:
        # get Q(alpha)
        indices = alpha_0_k < alpha  # [256, 20]
        pred0 = _lambda
        if algorithm_type == '2':
            pred1 = (gamma * alpha).sum(dim=1)  # [256,]
            pred2 = (beta_k * (alpha - alpha_0_k).pow(2) * indices).sum(dim=1)  # [256,]
            pred = pred0 + pred1 + pred2  # [256,]
        elif algorithm_type == '1+2':
            pred1 = (gamma * (alpha - alpha_0_k) * indices).sum(dim=1)  # [256,]
            pred2 = (beta_k * (alpha - alpha_0_k).pow(2) * indices).sum(dim=1)  # [256,]
            pred = pred0 + pred1 + pred2  # [256,]
        elif algorithm_type == '1':
            pred1 = (gamma * (alpha - alpha_0_k) * indices).sum(dim=1)  # [256,]
            pred = pred0 + pred1  # [256,]
        else:
            raise ValueError("algorithm_type must be '1', '2', or '1+2'")

        return pred
    else:
        # get pred mean value
        y_hat = get_y_hat(alpha_0_k, _lambda, gamma, beta_k, algorithm_type)  # [256,]

        return y_hat


def loss_fn(tuple_param):
    alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type = tuple_param

    # CRPS
    labels = labels.unsqueeze(1)  # [256, 1]
    return get_crps(alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type)


# def get_mse(alpha_prime_k, _lambda, gamma, eta_k, y, algorithm_type):
#     alpha_0_k, beta_k = phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type)
#
#     # get y_hat
#     y_hat = get_y_hat(alpha_0_k, _lambda, gamma, beta_k, algorithm_type)  # [256,]
#
#     # calculate loss
#     loss = nn.MSELoss()
#     mseLoss = loss(y_hat, y)
#
#     return mseLoss


# noinspection DuplicatedCode
def get_crps(alpha_prime_k, _lambda, gamma, eta_k, y, algorithm_type):
    # [256, 1], [256, 20], [256, 20], [256, 20], [256, 1]
    alpha_0_k, beta_k = phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type)
    alpha_1_k1 = pad(alpha_0_k, pad=(0, 1), value=1)[:, 1:]  # [256, 20]

    # calculate the maximum for each segment of the spline and get l
    df1 = alpha_1_k1.expand(alpha_prime_k.shape[1], alpha_prime_k.shape[0],
                            alpha_prime_k.shape[1]).T.clone()  # [20, 256, 20]
    knots = df1 - alpha_0_k  # [20, 256, 20]
    knots[knots < 0] = 0  # [20, 256, 20]
    if algorithm_type == '2':
        df2 = alpha_1_k1.T.unsqueeze(2)
        knots = _lambda + (df2 * gamma).sum(dim=2) + (knots.pow(2) * beta_k).sum(dim=2)  # [20, 256]
    elif algorithm_type == '1+2':
        knots = _lambda + (knots * gamma).sum(dim=2) + (knots.pow(2) * beta_k).sum(dim=2)  # [20, 256]
    elif algorithm_type == '1':
        knots = _lambda + (knots * gamma).sum(dim=2)  # [20, 256]
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    knots = pad(knots.T, (1, 0), value=_lambda)[:, :-1]  # F(alpha_{1~K})=0~max  # [256, 20]
    diff = y - knots  # [256, 20]
    alpha_l = diff > 0  # [256, 20]

    # calculate the parameter for quadratic equation
    y = y.squeeze()  # [256,]
    if algorithm_type == '2':
        A = torch.sum(alpha_l * beta_k, dim=1)  # [256,]
        B = gamma[:, 0] - 2 * torch.sum(alpha_l * beta_k * alpha_0_k, dim=1)  # [256,]
        C = _lambda - y + torch.sum(alpha_l * beta_k * alpha_0_k * alpha_0_k, dim=1)  # [256,]
    elif algorithm_type == '1+2':
        A = torch.sum(alpha_l * beta_k, dim=1)  # [256,]
        B = torch.sum(alpha_l * gamma, dim=1) - 2 * torch.sum(alpha_l * beta_k * alpha_0_k, dim=1)  # [256,]
        C = _lambda - y - torch.sum(alpha_l * gamma * alpha_0_k, dim=1) + torch.sum(
            alpha_l * beta_k * alpha_0_k * alpha_0_k, dim=1)  # [256,]
    elif algorithm_type == '1':
        A = torch.zeros_like(y)  # [256,]
        B = torch.sum(alpha_l * gamma, dim=1)
        C = _lambda - y - torch.sum(alpha_l * gamma * alpha_0_k, dim=1)
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")

    # solve the quadratic equation: since A may be zero, roots can be from different methods.
    not_zero = (A != 0)  # [256,]
    alpha_plus = torch.zeros_like(A)  # [256,]
    # since there may be numerical calculation error  #0
    idx = (B ** 2 - 4 * A * C) < 0  # 0  # [256,]
    diff = diff.abs()  # [256,]
    index = diff == (diff.min(dim=1)[0].view(-1, 1))  # [256,]
    index[~idx, :] = False  # [256,]
    # index=diff.abs()<1e-4  # 0,1e-4 is a threshold
    # idx=index.sum(dim=1)>0  # 0
    alpha_plus[idx] = alpha_0_k[index]  # 0  # [256,]
    alpha_plus[~not_zero] = -C[~not_zero] / B[~not_zero]  # [256,]
    not_zero = ~(~not_zero | idx)  # 0  # [256,]
    delta = B[not_zero].pow(2) - 4 * A[not_zero] * C[not_zero]  # [232,]
    alpha_plus[not_zero] = (-B[not_zero] + torch.sqrt(delta)) / (2 * A[not_zero])  # [256,]

    # get CRPS
    crps_1 = (_lambda - y) * (1 - 2 * alpha_plus)  # [256,]
    if algorithm_type == '2':
        crps_2 = gamma[:, 0] * (1 / 3 - alpha_plus.pow(2))
        crps_3 = torch.sum(1 / 6 * beta_k * (1 - alpha_0_k).pow(4), dim=1)
        crps_4 = torch.sum(2 / 3 * alpha_l * beta_k * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(3), dim=1)
        crps = crps_1 + crps_2 + crps_3 - crps_4
    elif algorithm_type == '1+2':
        crps_2 = torch.sum(1 / 3 * gamma * (1 - alpha_0_k).pow(3), dim=1)  # [256,]
        crps_3 = torch.sum(alpha_l * gamma * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(2), dim=1)  # [256,]
        crps_4 = torch.sum(1 / 6 * beta_k * (1 - alpha_0_k).pow(4), dim=1)  # [256,]
        crps_5 = torch.sum(2 / 3 * alpha_l * beta_k * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(3), dim=1)  # [256,]
        crps = crps_1 + crps_2 - crps_3 + crps_4 - crps_5  # [256, 256]
    elif algorithm_type == '1':
        crps_2 = torch.sum(1 / 3 * gamma * (1 - alpha_0_k).pow(3), dim=1)  # [256,]
        crps_3 = torch.sum(alpha_l * gamma * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(2), dim=1)  # [256,]
        crps = crps_1 + crps_2 - crps_3  # [256,]
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")

    crps = torch.mean(crps)  # [256,]
    return crps
