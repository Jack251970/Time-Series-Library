import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention
from models.quantile_function.lstm_yjqr import sample_yjqr


class Model(nn.Module):
    def __init__(self, params):
        """
        LSTM-ED-YJQR
        """
        super(Model, self).__init__()
        self.task_name = params.task_name
        self.batch_size = params.batch_size
        self.lstm_hidden_size = params.lstm_hidden_size
        self.enc_lstm_layers = params.lstm_layers
        self.dec_lstm_layers = 1
        self.sample_times = params.sample_times
        self.lstm_dropout = params.dropout
        self.num_spline = params.num_spline
        self.pred_start = params.seq_len
        self.pred_len = params.pred_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start

        self.enc_in = params.enc_in
        input_size = self.enc_in + self.lag

        self.enc_lstm_input_size = input_size
        input_size = self.enc_in + self.lag - 1

        self.dec_lstm_input_size = input_size

        self.n_heads = params.n_heads
        self.d_model = params.d_model

        # LSTM: Encoder and Decoder
        self.lstm_enc = nn.LSTM(input_size=self.enc_lstm_input_size,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.enc_lstm_layers,
                                bias=True,
                                batch_first=False,
                                bidirectional=False,
                                dropout=self.lstm_dropout)
        self.lstm_dec = nn.LSTM(input_size=self.dec_lstm_input_size,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.dec_lstm_layers,
                                bias=True,
                                batch_first=False,
                                bidirectional=False,
                                dropout=self.lstm_dropout)
        self.init_lstm(self.lstm_enc)
        self.init_lstm(self.lstm_dec)

        # Attention
        self.attention = FullAttention(mask_flag=False, output_attention=True, attention_dropout=0.1)
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

        # YJQM
        self.yjqm_input_size = self.lstm_hidden_size * self.dec_lstm_layers
        self.pre_lamda = nn.Linear(self.yjqm_input_size, 1)
        self.pre_mu = nn.Linear(self.yjqm_input_size, 1)
        self.pre_sigma = nn.Linear(self.yjqm_input_size, 1)

        self.lamda = nn.LeakyReLU(negative_slope=0.5)  # TODO

        self.mu = nn.Sigmoid()

        self.sigma = nn.Softplus()
        # self.sigma = nn.Sigmoid()
        # self.sigma = nn.ReLU()

        # Reindex
        self.new_index = None
        self.lag_index = None
        self.cov_index = None

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

    # noinspection DuplicatedCode
    def get_input_data(self, x_enc, y_enc):
        if self.lag_index is None:
            self.lag_index = []
            self.cov_index = []
            if self.new_index is None:
                self.new_index = list(range(self.enc_in + self.lag))
            for i in range(self.lag):
                self.lag_index.append(self.new_index[i])
            for i in range(self.enc_in + self.lag - 1):
                lag = False
                for j in self.lag_index:
                    if i == j:
                        lag = True
                        break
                if not lag:
                    self.cov_index.append(i)

        batch = torch.cat((x_enc, y_enc[:, -self.pred_len:, :]), dim=1)

        # s = seq_len
        enc_in = batch[:, :self.pred_start, :]

        # s = label_len + pred_len
        dec_in = batch[:, -self.pred_steps:, :-1]
        labels = batch[:, -self.pred_steps:, -1]

        return enc_in, dec_in, labels

    # noinspection DuplicatedCode,PyUnusedLocal
    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'probability_forecast':
            # we don't need to use mark data because lstm can handle time series relation information
            enc_in, dec_in, labels = self.get_input_data(x_enc, y_enc)
            return self.probability_forecast(enc_in, dec_in, x_mark_enc, x_mark_dec, labels)  # return loss list
        return None

    # noinspection PyUnusedLocal
    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, probability_range=None):
        if self.task_name == 'probability_forecast':
            enc_in, dec_in, _ = self.get_input_data(x_enc, y_enc)
            return self.probability_forecast(enc_in, dec_in, x_mark_enc, x_mark_dec,
                                             probability_range=probability_range)
        return None

    # noinspection DuplicatedCode
    def run_lstm_enc(self, x, hidden, cell):
        _, (hidden, cell) = self.lstm_enc(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

        return hidden, cell

    # noinspection DuplicatedCode
    def run_lstm_dec(self, x, x_mark_dec_step, hidden, cell, enc_hidden_attn):
        _, (hidden, cell) = self.lstm_dec(x, (hidden, cell))  # [1, 256, 64], [1, 256, 64]

        dec_hidden_attn = hidden.clone().view(self.batch_size, 1, self.dec_lstm_layers * self.lstm_hidden_size)
        dec_hidden_attn = self.dec_embedding(dec_hidden_attn, x_mark_dec_step)
        dec_hidden_attn = dec_hidden_attn.view(self.batch_size, self.L_dec, self.H, self.E_dec)  # [256, 1, 2, 20]

        enc_hidden_attn = self.enc_norm(enc_hidden_attn)
        dec_hidden_attn = self.dec_norm(dec_hidden_attn)

        y, attn = self.attention(dec_hidden_attn, enc_hidden_attn, enc_hidden_attn, None)

        if self.out_projection is not None:
            y = self.out_projection(y)

        y = self.out_norm(y)

        y = y.view(self.dec_lstm_layers, self.batch_size, -1)

        return y, y, cell, attn

    @staticmethod
    def get_hidden_permute(hidden):
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0)  # [256, 1, 40]
        hidden_permute = hidden_permute.contiguous().view(hidden.shape[1], -1)  # [256, 40]

        return hidden_permute

    def get_yjqm_parameter(self, hidden_permute):
        pre_lamda = self.pre_lamda(hidden_permute)
        lamda = pre_lamda

        pre_mu = self.pre_mu(hidden_permute)
        mu = self.mu(pre_mu)

        pre_sigma = self.pre_sigma(hidden_permute)
        sigma = self.sigma(pre_sigma)

        return lamda, mu, sigma

    # noinspection DuplicatedCode
    def probability_forecast(self, x_enc, x_dec, x_mark_enc, x_mark_dec, labels=None, sample=True,
                             probability_range=None):
        # [256, 96, 4], [256, 12, 7], [256, 12,]
        if probability_range is None:
            probability_range = [0.5]

        device = x_enc.device

        assert isinstance(probability_range, list)
        probability_range_len = len(probability_range)
        probability_range = torch.Tensor(probability_range).to(device)  # [3]

        x_enc = x_enc.permute(1, 0, 2)  # [96, 256, 4]
        x_dec = x_dec.permute(1, 0, 2)  # [16, 256, 7]
        if labels is not None:
            labels = labels.permute(1, 0)  # [12, 256]

        # hidden and cell are initialized to zero
        hidden = torch.zeros(self.enc_lstm_layers, self.batch_size, self.lstm_hidden_size,
                             device=device)  # [2, 256, 40]
        cell = torch.zeros(self.enc_lstm_layers, self.batch_size, self.lstm_hidden_size, device=device)  # [2, 256, 40]

        # run encoder
        enc_hidden = torch.zeros(self.pred_start, self.enc_lstm_layers, self.batch_size, self.lstm_hidden_size,
                                 device=device)  # [96, 1, 256, 40]
        enc_cell = torch.zeros(self.pred_start, self.enc_lstm_layers, self.batch_size, self.lstm_hidden_size,
                               device=device)  # [96, 1, 256, 40]
        for t in range(self.pred_start):
            hidden, cell = self.run_lstm_enc(x_enc[t].unsqueeze_(0).clone(), hidden, cell)  # [2, 256, 40], [2, 256, 40]
            enc_hidden[t] = hidden
            enc_cell[t] = cell

        # only select the last hidden state
        # embedding encoder
        enc_hidden = enc_hidden.view(self.batch_size, self.pred_start,
                                     self.enc_lstm_layers * self.lstm_hidden_size)
        enc_hidden = self.enc_embedding(enc_hidden, x_mark_enc)
        enc_hidden_attn = enc_hidden.view(self.batch_size, self.L_enc, self.H, self.E_enc)  # [256, 96, 8, 5]

        dec_hidden = torch.zeros(self.dec_lstm_layers, self.batch_size, self.lstm_hidden_size, device=device)
        dec_cell = torch.zeros(self.dec_lstm_layers, self.batch_size, self.lstm_hidden_size, device=device)

        if labels is not None:
            # train mode or validate mode
            hidden_permutes = torch.zeros(self.batch_size, self.pred_steps, self.yjqm_input_size, device=device)

            # initialize hidden and cell
            hidden, cell = dec_hidden.clone(), dec_cell.clone()

            # decoder
            for t in range(self.pred_steps):
                x_mark_dec_step = x_mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                hidden_yjqm, hidden, cell, _ = self.run_lstm_dec(x_dec[t].unsqueeze_(0).clone(), x_mark_dec_step,
                                                                 hidden, cell, enc_hidden_attn)
                hidden_permute = self.get_hidden_permute(hidden_yjqm)
                hidden_permutes[:, t, :] = hidden_permute

                # check if hidden contains NaN
                if torch.isnan(hidden).sum() > 0 or torch.isnan(hidden_yjqm).sum() > 0:
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
                lamda, mu, sigma = self.get_yjqm_parameter(hidden_permute)  # [256, 1], [256, 1], [256, 1]
                y = labels[t].clone()  # [256,]
                # lambda,mu，sigma->(256,)
                loss_list.append((lamda.squeeze(), mu.squeeze(), sigma.squeeze(), y))

            return loss_list, stop_flag
        else:
            # test mode
            # initialize alpha range
            alpha_low = (1 - probability_range) / 2  # [3]
            alpha_high = 1 - (1 - probability_range) / 2  # [3]
            low_alpha = alpha_low.unsqueeze(0).expand(self.batch_size, -1)  # [256, 3]
            high_alpha = alpha_high.unsqueeze(0).expand(self.batch_size, -1)  # [256, 3]

            # initialize samples
            samples_low = torch.zeros(probability_range_len, self.batch_size, self.pred_steps,
                                      device=device)  # [3, 256, 16]
            samples_high = samples_low.clone()  # [3, 256, 16]
            samples = torch.zeros(self.sample_times, self.batch_size, self.pred_steps, device=device)  # [99, 256, 12]

            label_len = self.pred_steps - self.pred_len

            for j in range(self.sample_times + probability_range_len * 2):
                # clone test batch
                x_dec_clone = x_dec.clone()  # [16, 256, 7]

                # initialize hidden and cell
                hidden, cell = dec_hidden.clone(), dec_cell.clone()

                # decoder
                for t in range(self.pred_steps):
                    x_mark_dec_step = x_mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                    hidden_yjqm, hidden, cell, _ = self.run_lstm_dec(x_dec[t].unsqueeze_(0).clone(), x_mark_dec_step,
                                                                     hidden, cell, enc_hidden_attn)
                    hidden_permute = self.get_hidden_permute(hidden_yjqm)
                    lamda, mu, sigma = self.get_yjqm_parameter(hidden_permute)

                    if j < probability_range_len:
                        pred_alpha = low_alpha[:, j].unsqueeze(-1)  # [256, 1]
                    elif j < 2 * probability_range_len:
                        pred_alpha = high_alpha[:, j - probability_range_len].unsqueeze(-1)  # [256, 1]
                    else:
                        # pred alpha is a uniform distribution
                        uniform = torch.distributions.uniform.Uniform(
                            torch.tensor([0.0001], device=device),
                            torch.tensor([0.9999], device=device))
                        pred_alpha = uniform.sample(torch.Size([self.batch_size]))  # [256, 1]

                    pred = sample_yjqr(lamda, mu, sigma, pred_alpha)
                    if j < probability_range_len:
                        samples_low[j, :, t] = pred
                    elif j < 2 * probability_range_len:
                        samples_high[j - probability_range_len, :, t] = pred
                    else:
                        samples[j - probability_range_len * 2, :, t] = pred

                    if t >= label_len:
                        for lag in range(self.lag):
                            if t < self.pred_steps - lag - 1:
                                x_dec_clone[t + 1, :, self.lag_index[0]] = pred

            samples_mu = torch.mean(samples, dim=0).unsqueeze(-1)  # mean or median ? # [256, 12, 1]
            samples_std = samples.std(dim=0).unsqueeze(-1)  # [256, 12, 1]

            # get attention map using integral to calculate the pred value
            attention_map = torch.zeros(self.pred_steps, self.batch_size, self.H, self.L_dec, self.L_enc,
                                        device=device)

            # clone test batch
            x_dec_clone = x_dec.clone()  # [16, 256, 7]

            # sample
            samples_mu1 = torch.zeros(self.batch_size, self.pred_steps, 1, device=device)

            # initialize hidden and cell
            hidden, cell = dec_hidden.clone(), dec_cell.clone()

            # decoder
            for t in range(self.pred_steps):
                x_mark_dec_step = x_mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                hidden_yjqm, hidden, cell, attn = self.run_lstm_dec(x_dec[t].unsqueeze_(0).clone(), x_mark_dec_step,
                                                                    hidden, cell, enc_hidden_attn)
                attention_map[t] = attn
                hidden_permute = self.get_hidden_permute(hidden_yjqm)
                lamda, mu, sigma = self.get_yjqm_parameter(hidden_permute)

                if not sample:
                    pred = sample_yjqr(lamda, mu, sigma, None)
                    samples_mu1[:, t, 0] = pred

                    if t >= label_len:
                        for lag in range(self.lag):
                            if t < self.pred_steps - lag - 1:
                                x_dec_clone[t + 1, :, self.lag_index[0]] = pred

            if not sample:
                # use integral to calculate the mean
                return samples, samples_mu1, samples_std, samples_high, samples_low, attention_map
            else:
                # use uniform samples to calculate the mean
                return samples, samples_mu, samples_std, samples_high, samples_low, attention_map
