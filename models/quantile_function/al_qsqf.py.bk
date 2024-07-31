import torch
import torch.nn as nn
from torch.nn.functional import pad

from layers.ALQSQF_EncDec import series_decomp
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention

from models.quantile_function.qf_func import phase_gamma_and_eta_k


class Model(nn.Module):
    def __init__(self, params, algorithm_type="1+2"):
        """
        AL-QSQF: Attention-based LSTM encoder-decoder network with spline function for non-parametric probabilistic
        forecasting.

        params: parameters for the model.
        algorithm_type: algorithm type, e.g. '1', '2', '1+2'
        """
        super(Model, self).__init__()
        self.algorithm_type = algorithm_type
        self.task_name = params.task_name
        self.batch_size = params.batch_size
        self.lstm_hidden_size = params.lstm_hidden_size
        self.enc_in = params.enc_in
        self.enc_lstm_layers = params.lstm_layers
        self.dec_lstm_layers = 1
        self.sample_times = params.sample_times
        self.dropout = params.dropout
        self.num_spline = params.num_spline
        self.pred_start = params.seq_len
        self.pred_len = params.pred_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start

        # phase custom_params
        custom_params = params.custom_params
        custom_params = custom_params.split('_')
        self.enc_lstm_input_size = self.enc_in + self.lag
        self.dec_lstm_input_size = self.enc_in + self.lag - 1
        self.n_heads = params.n_heads
        self.d_model = params.d_model
        if len(custom_params) > 0 and custom_params[0] in ('label', 'label1'):
            if custom_params[0] == 'label':
                self.pred_start = params.seq_len - params.label_len
            else:
                self.pred_steps = params.pred_len + params.label_len
                self.pred_start = params.seq_len - params.label_len
            custom_params.pop(0)
        if len(custom_params) > 0 and custom_params[0] != '':
            raise ValueError(f"Cannot parse these custom_params: {custom_params}")

        # LSTM: Encoder and Decoder
        self.lstm_enc = nn.LSTM(input_size=self.enc_lstm_input_size,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.enc_lstm_layers,
                                bias=True,
                                batch_first=False,
                                bidirectional=False,
                                dropout=self.dropout)
        self.lstm_dec = nn.LSTM(input_size=self.dec_lstm_input_size,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.dec_lstm_layers,
                                bias=True,
                                batch_first=False,
                                bidirectional=False,
                                dropout=self.dropout)
        self.init_lstm(self.lstm_enc)
        self.init_lstm(self.lstm_dec)

        # Series decomposition
        self.decomp_enc = series_decomp(kernel_size=params.moving_avg)
        self.decomp_dec1 = series_decomp(kernel_size=params.moving_avg)
        self.decomp_dec2 = series_decomp(kernel_size=params.moving_avg)

        # Attention
        self.attention = FullAttention(mask_flag=False, output_attention=True, attention_dropout=self.dropout)
        self.L_enc = self.pred_start
        self.L_dec = 1
        self.H = self.n_heads

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.dec_lstm_layers * self.lstm_hidden_size % self.n_heads != 0:
            raise ValueError("dec_lstm_layers * lstm_hidden_size must be divisible by n_heads")

        self.E_enc = self.d_model // self.n_heads
        self.E_dec = self.d_model // self.n_heads

        self.enc_embedding = DataEmbedding(self.enc_lstm_layers * self.lstm_hidden_size, self.d_model,
                                           params.embed, params.freq, 0)
        self.dec_embedding = DataEmbedding(self.dec_lstm_layers * self.lstm_hidden_size, self.d_model,
                                           params.embed, params.freq, 0)
        out_size = self.dec_lstm_layers * self.lstm_hidden_size // self.n_heads
        self.out_projection = nn.Linear(self.E_dec, out_size)
        self.enc_norm = nn.LayerNorm([self.L_enc, self.H, self.E_enc])
        self.dec_norm = nn.LayerNorm([self.L_dec, self.H, self.E_dec])
        self.out_norm = nn.LayerNorm([self.L_dec, self.H, out_size])

        # QSQM
        device = torch.device("cuda" if params.use_gpu else "cpu")
        self.qsqm_input_size = self.lstm_hidden_size * self.dec_lstm_layers
        self.linear_lambda = nn.Linear(self.qsqm_input_size, 1)
        self.linear_lambda.weight.data.fill_(0.0)
        if self.algorithm_type == '2':
            self.linear_gamma = nn.Linear(self.qsqm_input_size, 1)
        elif self.algorithm_type == '1+2':
            self.linear_gamma = nn.Linear(self.qsqm_input_size, self.num_spline)
        elif self.algorithm_type == '1':
            self.linear_gamma = nn.Linear(self.qsqm_input_size, self.num_spline)
        else:
            raise ValueError("algorithm_type must be '1', '2', or '1+2'")
        self.linear_eta_k = nn.Linear(self.qsqm_input_size, self.num_spline)
        self.soft_plus = nn.Softplus()  # make sure parameter is positive
        self.alpha_prime_k = torch.ones(self.batch_size, self.num_spline).to(device) / self.num_spline  # [256, 20]

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
            return self.probability_forecast(x_enc, y_enc[:, -self.pred_len:, :-1],
                                             x_mark_enc, x_mark_dec[:, -self.pred_len:, :],
                                             labels=y_enc[:, -self.pred_len:, -1:])  # return loss list
        return None

    # noinspection PyUnusedLocal
    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, probability_range=None):
        if self.task_name == 'probability_forecast':
            return self.probability_forecast(x_enc, y_enc[:, -self.pred_len:, :-1],
                                             x_mark_enc, x_mark_dec[:, -self.pred_len:, :],
                                             probability_range=probability_range)
        return None

    # noinspection DuplicatedCode
    def run_lstm_enc(self, x, hidden, cell):
        _, (hidden, cell) = self.lstm_enc(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]
        return hidden, cell

    # noinspection DuplicatedCode
    def run_lstm_dec(self, t, dec_hidden, x, x_mark_dec_step, hidden, cell, enc_hidden_attn):
        _, (hidden, cell) = self.lstm_dec(x, (hidden, cell))  # [1, 256, 64], [1, 256, 64]
        # series decomposition & embedding decoder
        dec_hidden_step = hidden.clone().view(self.batch_size, 1, self.dec_lstm_layers * self.lstm_hidden_size)
        if t == 0:
            dec_hidden = dec_hidden_step
        else:
            dec_hidden_front = dec_hidden[0:t].clone().view(self.batch_size, t,
                                                            self.dec_lstm_layers * self.lstm_hidden_size)
            dec_hidden = torch.cat([dec_hidden_front, dec_hidden_step], dim=1)
        dec_hidden_uncertainty, dec_hidden_attn_trend = self.decomp_dec1(dec_hidden)
        dec_hidden_uncertainty_step = dec_hidden_uncertainty[:, -1:, :]
        dec_hidden_attn_trend_step = dec_hidden_attn_trend[:, -1, :]
        dec_hidden_uncertainty_embed = self.dec_embedding(dec_hidden_uncertainty_step, x_mark_dec_step)
        # attention
        dec_hidden_attn = dec_hidden_uncertainty_embed.view(self.batch_size, self.L_dec, self.H, self.E_dec)
        enc_hidden_attn = self.enc_norm(enc_hidden_attn)
        dec_hidden_attn = self.dec_norm(dec_hidden_attn)
        y, attn = self.attention(dec_hidden_attn, enc_hidden_attn, enc_hidden_attn, None)
        y = self.out_projection(y)
        y = self.out_norm(y)
        y = y.view(self.dec_lstm_layers, self.batch_size, -1)
        return y, y, cell, attn, dec_hidden_attn_trend_step

    @staticmethod
    def get_hidden_permute(hidden):
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0)  # [256, 1, 40]
        hidden_permute = hidden_permute.contiguous().view(hidden.shape[1], -1)  # [256, 40]
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
    def initialize_sample_parameters(self, batch_size, device):
        samples_lambda = torch.zeros(self.pred_steps, batch_size, 1, device=device)
        if self.algorithm_type == '2':
            samples_gamma = torch.zeros(self.pred_steps, batch_size, 1, device=device)
            samples_eta_k = torch.zeros(self.pred_steps, batch_size, self.num_spline, device=device)
        elif self.algorithm_type == '1+2':
            samples_lambda = torch.zeros(self.pred_steps, batch_size, 1, device=device)
            samples_gamma = torch.zeros(self.pred_steps, batch_size, self.num_spline, device=device)
            samples_eta_k = torch.zeros(self.pred_steps, batch_size, self.num_spline, device=device)
        elif self.algorithm_type == '1':
            samples_lambda = torch.zeros(self.pred_steps, batch_size, 1, device=device)
            samples_gamma = torch.zeros(self.pred_steps, batch_size, self.num_spline, device=device)
            samples_eta_k = torch.zeros(self.pred_steps, batch_size, self.num_spline, device=device)
        else:
            raise ValueError("algorithm_type must be '1', '2', or '1+2'")
        return samples_lambda, samples_gamma, samples_eta_k

    # noinspection DuplicatedCode
    def probability_forecast(self, enc_in, dec_in, mark_enc, mark_dec, labels=None, sample=False,
                             probability_range=None):
        # [256, 96, 4], [256, 12, 7], [256, 12,]
        if probability_range is None:
            probability_range = [0.5]

        batch_size = enc_in.shape[0]  # 256
        device = enc_in.device

        assert isinstance(probability_range, list)
        probability_range_len = len(probability_range)
        probability_range = torch.Tensor(probability_range).to(device)  # [3]

        enc_in = enc_in.permute(1, 0, 2)  # [96, 256, 4]
        dec_in = dec_in.permute(1, 0, 2)  # [16, 256, 7]
        if labels is not None:
            labels = labels.permute(1, 0, 2)  # [12, 256]

        # initialize encoder hidden and cell
        enc_hidden_init = torch.zeros(self.enc_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        enc_cell_init = torch.zeros(self.enc_lstm_layers, batch_size, self.lstm_hidden_size, device=device)

        # initialize hidden and cell
        hidden, cell = enc_hidden_init.clone(), enc_cell_init.clone()

        # run encoder
        enc_hidden = torch.zeros(self.pred_start, self.enc_lstm_layers, batch_size, self.lstm_hidden_size,
                                 device=device)  # [96, 1, 256, 40]
        for t in range(self.pred_start):
            hidden, _ = self.run_lstm_enc(enc_in[t].unsqueeze_(0).clone(), hidden, cell)  # [2, 256, 40], [2, 256, 40]
            enc_hidden[t] = hidden

        # series decomposition & embedding
        enc_hidden = enc_hidden.view(batch_size, self.pred_start, self.enc_lstm_layers * self.lstm_hidden_size)
        enc_hidden_uncertainty, _ = self.decomp_enc(enc_hidden)  # [256, 96, 40]
        enc_hidden_uncertainty = self.enc_embedding(enc_hidden_uncertainty, mark_enc)
        enc_hidden_attn = enc_hidden_uncertainty.view(batch_size, self.L_enc, self.H, self.E_enc)  # [256, 96, 8, 5]

        # initialize decoder hidden and cell
        dec_hidden_init = torch.zeros(self.dec_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        dec_cell_init = torch.zeros(self.dec_lstm_layers, batch_size, self.lstm_hidden_size, device=device)

        if labels is not None:
            # train mode or validate mode
            hidden_permutes = torch.zeros(batch_size, self.pred_steps, self.qsqm_input_size, device=device)

            # initialize hidden and cell
            hidden, cell = dec_hidden_init.clone(), dec_cell_init.clone()

            # decoder
            dec_hidden = torch.zeros(self.pred_steps, self.enc_lstm_layers, batch_size, self.lstm_hidden_size,
                                     device=device)  # [16, 1, 256, 40]
            dec_out_trend = torch.zeros(self.pred_steps, batch_size, 1, device=device)  # [16, 256, 1]
            for t in range(self.pred_steps):
                x_mark_dec_step = mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                hidden_qsqm, hidden, cell, _, dec_trend = self.run_lstm_dec(t, dec_hidden,
                                                                            dec_in[t].unsqueeze_(0).clone(),
                                                                            x_mark_dec_step,
                                                                            hidden, cell, enc_hidden_attn)
                _lambda = self.linear_lambda(dec_trend)
                dec_out_trend[t] = _lambda
                dec_hidden[t] = hidden
                hidden_permute = self.get_hidden_permute(hidden_qsqm)
                hidden_permutes[:, t, :] = hidden_permute

                # check if hidden contains NaN
                if torch.isnan(hidden).sum() > 0 or torch.isnan(hidden_qsqm).sum() > 0:
                    break

            # get loss list
            stop_flag = False
            loss_list = []

            # decoder
            for t in range(self.pred_steps):
                hidden_permute = hidden_permutes[:, t, :]  # [256, 80]
                if torch.isnan(hidden_permute).sum() > 0:
                    loss_list.clear()
                    stop_flag = True
                    break
                gamma, eta_k = self.get_qsqm_parameter(hidden_permute)  # [256, 20], [256, 20]
                y = labels[t].clone()  # [256,]
                loss_list.append((self.alpha_prime_k, dec_out_trend[t], gamma, eta_k, y, self.algorithm_type))

            return loss_list, stop_flag
        else:
            # test mode
            # initialize alpha range
            alpha_low = (1 - probability_range) / 2  # [3]
            alpha_high = 1 - (1 - probability_range) / 2  # [3]
            low_alpha = alpha_low.unsqueeze(0).expand(batch_size, -1)  # [256, 3]
            high_alpha = alpha_high.unsqueeze(0).expand(batch_size, -1)  # [256, 3]

            # initialize samples
            samples_low = torch.zeros(probability_range_len, batch_size, self.pred_steps,
                                      device=device)  # [3, 256, 16]
            samples_high = samples_low.clone()  # [3, 256, 16]
            samples = torch.zeros(self.sample_times, batch_size, self.pred_steps, device=device)  # [99, 256, 12]

            label_len = self.pred_steps - self.pred_len

            for j in range(self.sample_times + probability_range_len * 2):
                # clone test batch
                x_dec_clone = dec_in.clone()  # [16, 256, 7]

                # initialize hidden and cell
                hidden, cell = dec_hidden_init.clone(), dec_cell_init.clone()

                # decoder
                dec_hidden = torch.zeros(self.pred_steps, self.enc_lstm_layers, batch_size, self.lstm_hidden_size,
                                         device=device)  # [16, 1, 256, 40]
                dec_out_trend = torch.zeros(self.pred_steps, batch_size, 1, device=device)  # [16, 256, 1]
                for t in range(self.pred_steps):
                    x_mark_dec_step = mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                    hidden_qsqm, hidden, cell, _, dec_trend = self.run_lstm_dec(t, dec_hidden,
                                                                                dec_in[t].unsqueeze_(0).clone(),
                                                                                x_mark_dec_step,
                                                                                hidden, cell, enc_hidden_attn)
                    dec_hidden[t] = hidden
                    _lambda = self.linear_lambda(dec_trend)
                    dec_out_trend[t] = _lambda
                    hidden_permute = self.get_hidden_permute(hidden_qsqm)
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

                    pred = sample_pred(self.alpha_prime_k, pred_alpha, dec_out_trend[t], gamma, eta_k,
                                       self.algorithm_type)
                    if j < probability_range_len:
                        samples_low[j, :, t] = pred
                    elif j < 2 * probability_range_len:
                        samples_high[j - probability_range_len, :, t] = pred
                    else:
                        samples[j - probability_range_len * 2, :, t] = pred

                    if t >= label_len:
                        for lag in range(self.lag):
                            if t < self.pred_steps - lag - 1:
                                x_dec_clone[t + 1, :, 0] = pred

            samples_mu = torch.mean(samples, dim=0).unsqueeze(-1)  # mean or median ? # [256, 12, 1]
            samples_std = samples.std(dim=0).unsqueeze(-1)  # [256, 12, 1]

            # get attention map using integral to calculate the pred value
            attention_map = torch.zeros(self.pred_steps, batch_size, self.H, self.L_dec, self.L_enc,
                                        device=device)

            # clone test batch
            x_dec_clone = dec_in.clone()  # [16, 256, 7]

            # sample
            samples_mu1 = torch.zeros(batch_size, self.pred_steps, 1, device=device)

            # initialize parameters
            samples_lambda, samples_gamma, samples_eta_k = self.initialize_sample_parameters(batch_size, device)

            # initialize hidden and cell
            hidden, cell = dec_hidden_init.clone(), dec_cell_init.clone()

            # decoder
            dec_hidden = torch.zeros(self.pred_steps, self.enc_lstm_layers, batch_size, self.lstm_hidden_size,
                                     device=device)  # [16, 1, 256, 40]
            dec_out_trend = torch.zeros(self.pred_steps, batch_size, 1, device=device)  # [16, 256, 1]
            for t in range(self.pred_steps):
                x_mark_dec_step = mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                hidden_qsqm, hidden, cell, attn, dec_trend = self.run_lstm_dec(t, dec_hidden,
                                                                               dec_in[t].unsqueeze_(0).clone(),
                                                                               x_mark_dec_step,
                                                                               hidden, cell, enc_hidden_attn)
                dec_hidden[t] = hidden
                _lambda = self.linear_lambda(dec_trend)
                dec_out_trend[t] = _lambda
                hidden_permute = self.get_hidden_permute(hidden_qsqm)
                gamma, eta_k = self.get_qsqm_parameter(hidden_permute)
                attention_map[t] = attn

                pred = sample_pred(self.alpha_prime_k, None, dec_out_trend[t], gamma, eta_k, self.algorithm_type)
                samples_lambda[t] = dec_out_trend[t]
                samples_gamma[t] = gamma
                samples_eta_k[t] = eta_k
                samples_mu1[:, t, 0] = pred

                if t >= label_len:
                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            x_dec_clone[t + 1, :, 0] = pred

            if not sample:
                # use integral to calculate the mean
                return (samples, samples_mu1, samples_std, samples_high, samples_low, attention_map,
                        (samples_lambda, samples_gamma, samples_eta_k))
            else:
                # use uniform samples to calculate the mean
                return (samples, samples_mu, samples_std, samples_high, samples_low, attention_map,
                        (samples_lambda, samples_gamma, samples_eta_k))


# noinspection DuplicatedCode
def get_y_hat(alpha_0_k, _lambda, gamma, beta_k, algorithm_type):
    # init min_alpha and max_alpha
    device = gamma.device
    min_alpha = torch.Tensor([0]).to(device)  # [1]
    max_alpha = torch.Tensor([1]).to(device)  # [1]

    # get int{Q(alpha)}
    _lambda = _lambda.squeeze()  # [256,]
    if algorithm_type == '2':
        integral0 = _lambda * (max_alpha - min_alpha)  # [256,]
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
def sample_pred(alpha_prime_k, alpha, _lambda, gamma, eta_k, algorithm_type):
    # phase parameter
    alpha_0_k, beta_k = phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type)

    if alpha is not None:
        # get Q(alpha)
        indices = alpha_0_k < alpha  # [256, 20]
        pred0 = _lambda.squeeze()  # [256,]
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


def loss_fn_crps(tuple_param):
    alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type = tuple_param

    # calculate loss
    crpsLoss = get_crps(alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type)

    return crpsLoss


# noinspection DuplicatedCode
def get_crps(alpha_prime_k, _lambda, gamma, eta_k, y, algorithm_type, punish=2.0):
    # [256, 1], [256, 20], [256, 20], [256, 20], [256, 1]
    alpha_0_k, beta_k = phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type)
    alpha_1_k1 = pad(alpha_0_k, pad=(0, 1), value=1)[:, 1:]  # [256, 20]

    # calculate the maximum for each segment of the spline and get l
    df1 = alpha_1_k1.expand(alpha_prime_k.shape[1], alpha_prime_k.shape[0],
                            alpha_prime_k.shape[1]).T.clone()  # [20, 256, 20]
    knots = df1 - alpha_0_k  # [20, 256, 20]
    knots[knots < 0] = 0  # [20, 256, 20]
    num_spline = knots.shape[0]  # 20
    _lambda = _lambda.permute(1, 0).repeat(num_spline, 1)  # [20, 256]
    if algorithm_type == '2':
        df2 = alpha_1_k1.T.unsqueeze(2)
        knots = _lambda + (df2 * gamma).sum(dim=2) + (knots.pow(2) * beta_k).sum(dim=2)  # [20, 256]
    elif algorithm_type == '1+2':
        knots = _lambda + (knots * gamma).sum(dim=2) + (knots.pow(2) * beta_k).sum(dim=2)  # [20, 256]
    elif algorithm_type == '1':
        knots = _lambda + (knots * gamma).sum(dim=2)  # [20, 256]
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    _lambda = _lambda.permute(1, 0)  # [256, 20]
    knots = pad(knots.T, (1, 0), value=float('-Infinity'))[:, :-1]  # F(alpha_{1~K})=0~max  # [256, 20]
    knots[knots == float('-Infinity')] = _lambda[knots == float('-Infinity')]  # [256, 20]
    diff = y - knots  # [256, 20]
    alpha_l = diff > 0  # [256, 20]

    # calculate the parameter for quadratic equation
    y = y.squeeze()  # [256,]
    _lambda = _lambda[:, 0]  # [256,]
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
    a_not_zero = (A != 0)  # [256,]
    alpha_plus = torch.zeros_like(A)  # [256,]
    # get alpha_plus when A is not zero and delta is smaller than zero
    delta_smaller_zero = (B ** 2 - 4 * A * C) < 0  # [256,]
    delta_error = delta_smaller_zero & a_not_zero  # [256,]
    diff = diff.abs()  # [256,]
    # fix: use argmin instead of min to make sure only one minimum value is selected
    min_indices = torch.argmin(diff, dim=1)
    index = torch.zeros_like(diff, dtype=torch.bool)
    for i, min_index in enumerate(min_indices):
        index[i, min_index] = True
    index[~delta_error, :] = False  # [256,]
    # index=diff.abs()<1e-4  # 0,1e-4 is a threshold
    # delta_smaller_zero=index.sum(dim=1)>0  # 0
    alpha_plus[delta_error] = alpha_0_k[index]  # 0  # [256,]
    # get alpha_plus when A is zero
    alpha_plus[~a_not_zero] = -C[~a_not_zero] / B[~a_not_zero]  # [256,]
    # get alpha_plus when A is not zero and delta is larger than zero
    a_not_zero = ~(~a_not_zero | delta_smaller_zero)  # 0  # [256,]
    delta = B[a_not_zero].pow(2) - 4 * A[a_not_zero] * C[a_not_zero]  # [232,]
    alpha_plus[a_not_zero] = (-B[a_not_zero] + torch.sqrt(delta)) / (2 * A[a_not_zero])  # [256,]

    # get CRPS
    crps_1 = (_lambda - y) * (1 - 2 * alpha_plus)  # [256,]
    lambda_punish = punish * _lambda.abs()  # [256,]
    if algorithm_type == '2':
        crps_2 = gamma[:, 0] * (1 / 3 - alpha_plus.pow(2))
        crps_3 = torch.sum(1 / 6 * beta_k * (1 - alpha_0_k).pow(4), dim=1)
        crps_4 = torch.sum(2 / 3 * alpha_l * beta_k * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(3), dim=1)
        crps = crps_1 + crps_2 + crps_3 - crps_4 + lambda_punish  # [256,]
    elif algorithm_type == '1+2':
        crps_2 = torch.sum(1 / 3 * gamma * (1 - alpha_0_k).pow(3), dim=1)  # [256,]
        crps_3 = torch.sum(alpha_l * gamma * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(2), dim=1)  # [256,]
        crps_4 = torch.sum(1 / 6 * beta_k * (1 - alpha_0_k).pow(4), dim=1)  # [256,]
        crps_5 = torch.sum(2 / 3 * alpha_l * beta_k * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(3), dim=1)  # [256,]
        crps = crps_1 + crps_2 - crps_3 + crps_4 - crps_5 + lambda_punish  # [256,]
    elif algorithm_type == '1':
        crps_2 = torch.sum(1 / 3 * gamma * (1 - alpha_0_k).pow(3), dim=1)  # [256,]
        crps_3 = torch.sum(alpha_l * gamma * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(2), dim=1)  # [256,]
        crps = crps_1 + crps_2 - crps_3 + lambda_punish # [256,]
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")

    crps = torch.mean(crps)  # [256,]
    return crps
