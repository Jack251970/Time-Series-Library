import torch
import torch.nn as nn

from layers.AutoCorrelation import AutoCorrelation
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention
from models.quantile_function.lstm_cq import ConvLayer, sample_pred


class Model(nn.Module):
    def __init__(self, params, algorithm_type="1+2"):
        """
        LSTM-AQ: Auto-Regressive LSTM based on encoder-decoder architecture with convolution and spline to provide
        probabilistic forecasting.

        params: parameters for the model.
        algorithm_type: algorithm type, e.g. '1', '2', '1+2'
        """
        super(Model, self).__init__()
        self.algorithm_type = algorithm_type
        self.task_name = params.task_name
        self.batch_size = params.batch_size
        self.lstm_hidden_size = params.lstm_hidden_size
        self.enc_lstm_layers = params.lstm_layers
        self.dec_lstm_layers = 1  # TODO: Check self.enc_lstm_layers!!
        self.sample_times = params.sample_times
        self.lstm_dropout = params.dropout
        self.num_spline = params.num_spline
        self.pred_start = params.seq_len
        self.pred_len = params.pred_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start

        # phase custom_params
        custom_params = params.custom_params
        custom_params = custom_params.split('_')
        if len(custom_params) > 0 and custom_params[0] == 'cnn':
            self.use_cnn = True
            custom_params.pop(0)
        else:
            self.use_cnn = False
        if len(custom_params) > 0 and all([len(custom_params[0]) == 2, all([i in 'ACLH' for i in custom_params[0]])]):
            features = custom_params.pop(0)
            self.enc_feature = features[0]
            self.dec_feature = features[1]
            self.enc_in = params.enc_in
            input_size = self.get_input_size(self.enc_feature, True)
            if self.use_cnn:
                input_size = input_size + 2 * 2 - (3 - 1) - 1 + 1  # take conv into account
                input_size = (input_size + 2 * 1 - (3 - 1) - 1) // 2 + 1  # take maxPool into account
            self.enc_lstm_input_size = input_size
            input_size = self.get_input_size(self.dec_feature, False)
            if self.use_cnn:
                input_size = input_size + 2 * 2 - (3 - 1) - 1 + 1
                input_size = (input_size + 2 * 1 - (3 - 1) - 1) // 2 + 1
            self.dec_lstm_input_size = input_size
        if len(custom_params) > 0 and custom_params[0] in ('attn', 'corr'):
            self.use_attn = custom_params[0]
            self.n_heads = params.n_heads
            self.d_model = params.d_model
            custom_params.pop(0)
        else:
            self.use_attn = None
        if len(custom_params) > 0 and custom_params[0] == 'dhz':
            self.dec_hidden_zero = True
            custom_params.pop(0)
        else:
            self.dec_hidden_zero = False
        if len(custom_params) > 0 and custom_params[0] == 'dhd1':
            self.dec_hidden_difference1 = True
            custom_params.pop(0)
        else:
            self.dec_hidden_difference1 = False
        if len(custom_params) > 0 and custom_params[0] in ('ap', 'ap1', 'ap2'):
            self.attention_projection = custom_params[0]
            custom_params.pop(0)
        else:
            self.attention_projection = None
        if len(custom_params) > 0 and custom_params[0] == 'dhs':
            self.dec_hidden_separate = True
            custom_params.pop(0)
        else:
            self.dec_hidden_separate = False
        if len(custom_params) > 0 and custom_params[0] == 'norm':
            self.use_norm = True
            custom_params.pop(0)
        else:
            self.use_norm = False
        if len(custom_params) > 0 and custom_params[0] == 'label':
            self.pred_steps = params.pred_len + params.label_len
            self.pred_start = params.seq_len - params.label_len
            custom_params.pop(0)
        if len(custom_params) > 0:
            raise ValueError(f"Cannot parse these custom_params: {custom_params}")

        # CNN
        if self.use_cnn:
            self.cnn_enc = ConvLayer(1)
            self.cnn_dec = ConvLayer(1)

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
        if self.use_attn is not None:
            if self.use_attn == 'attn':
                self.attention = FullAttention(mask_flag=False, output_attention=True, attention_dropout=0.1)
            else:
                self.attention = AutoCorrelation(mask_flag=False, output_attention=True, agg_mode='full',
                                                 attention_dropout=0.1)
            self.L_enc = self.pred_start
            self.L_dec = 1
            self.H = self.n_heads

            if self.attention_projection is not None:
                if self.d_model % self.n_heads != 0:
                    raise ValueError("d_model must be divisible by n_heads")
                if self.dec_lstm_layers * self.lstm_hidden_size % self.n_heads != 0:
                    raise ValueError("dec_lstm_layers * lstm_hidden_size must be divisible by n_heads")

                self.E_enc = self.d_model // self.n_heads
                self.E_dec = self.d_model // self.n_heads
            else:
                if self.enc_lstm_layers * self.lstm_hidden_size % self.n_heads != 0:
                    raise ValueError("enc_lstm_layers * lstm_hidden_size must be divisible by n_heads")
                self.E_enc = self.enc_lstm_layers * self.lstm_hidden_size // self.n_heads

                if self.dec_lstm_layers * self.lstm_hidden_size % self.n_heads != 0:
                    raise ValueError("dec_lstm_layers * lstm_hidden_size must be divisible by n_heads")
                self.E_dec = self.dec_lstm_layers * self.lstm_hidden_size // self.n_heads

            out_size = self.E_dec
            if self.attention_projection is not None:
                if self.attention_projection == 'ap':
                    self.enc_embedding = nn.Linear(self.enc_lstm_layers * self.lstm_hidden_size, self.d_model)
                    self.dec_embedding = nn.Linear(self.dec_lstm_layers * self.lstm_hidden_size, self.d_model)
                else:
                    self.enc_embedding = DataEmbedding(self.enc_lstm_layers * self.lstm_hidden_size, self.d_model,
                                                       params.embed, params.freq, 0)
                    self.dec_embedding = DataEmbedding(self.dec_lstm_layers * self.lstm_hidden_size, self.d_model,
                                                       params.embed, params.freq, 0)
                if not self.dec_hidden_separate:
                    out_size = self.dec_lstm_layers * self.lstm_hidden_size // self.n_heads
                    self.out_projection = nn.Linear(self.E_dec, out_size)
                else:
                    self.out_projection = None
            else:
                self.out_projection = None
            if self.use_norm:
                self.enc_norm = nn.LayerNorm([self.L_enc, self.H, self.E_enc])
                self.dec_norm = nn.LayerNorm([self.L_dec, self.H, self.E_dec])
                self.out_norm = nn.LayerNorm([self.L_dec, self.H, out_size])

        # QSQM
        if self.attention_projection is not None and self.dec_hidden_separate:
            self.qsqm_input_size = self.d_model
        else:
            self.qsqm_input_size = self.lstm_hidden_size * self.dec_lstm_layers
        self._lambda = -1e-3  # make sure all data is not on the left point
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
        device = torch.device("cuda" if params.use_gpu else "cpu")
        y = torch.ones(self.num_spline) / self.num_spline
        self.alpha_prime_k = y.repeat(self.batch_size, 1).to(device)  # [256, 20]

        # Reindex
        self.new_index = None
        self.lag_index = None
        self.cov_index = None

    def get_input_size(self, feature, enc):
        if enc:
            if feature == 'A':
                return self.enc_in + self.lag
            elif feature == 'C':
                return self.enc_in - 1
            elif feature == 'L':
                return self.lag
            elif feature == 'H':
                return self.lag + 1
            else:
                raise ValueError("enc_feature must be 'A', 'C', 'L', or 'H'")
        else:
            if feature == 'A':
                return self.enc_in + self.lag - 1
            elif feature == 'C':
                return self.enc_in - 1
            elif feature == 'L':
                return self.lag
            else:
                raise ValueError("dec_feature must be 'A', 'C', or 'L'")

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
    def get_input_data(self, x_enc, y_enc, x_mark_enc, x_mark_dec):
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

        # contact sample
        batch = torch.cat((x_enc, y_enc[:, -self.pred_len:, :]), dim=1)
        x_mark = torch.cat((x_mark_enc, x_mark_dec[:, -self.pred_len:, :]), dim=1)

        # divide sample
        # s = seq_len
        if self.enc_feature == 'A':
            enc_in = batch[:, :self.pred_start, :]
        elif self.enc_feature == 'C':
            enc_in = batch[:, :self.pred_start, self.cov_index]
        elif self.enc_feature == 'L':
            enc_in = batch[:, :self.pred_start, self.lag_index]
        elif self.enc_feature == 'H':
            _index = self.lag_index.copy()
            _index.append(-1)
            enc_in = batch[:, :self.pred_start, _index]
        else:
            raise ValueError("enc_feature must be 'A', 'C', or 'L'")
        mark_enc = x_mark[:, :self.pred_start, :]

        # s = label_len + pred_len
        if self.dec_feature == 'A':
            dec_in = batch[:, -self.pred_steps:, :-1]
        elif self.dec_feature == 'C':
            dec_in = batch[:, -self.pred_steps:, self.cov_index]
        elif self.dec_feature == 'L':
            dec_in = batch[:, -self.pred_steps:, self.lag_index]
        else:
            raise ValueError("dec_feature must be 'A', 'C', or 'L'")
        mark_dec = x_mark[:, -self.pred_steps:, :]
        labels = batch[:, -self.pred_steps:, -1]

        return enc_in, dec_in, mark_enc, mark_dec, labels

    # noinspection DuplicatedCode,PyUnusedLocal
    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'probability_forecast':
            # we don't need to use mark data because lstm can handle time series relation information
            enc_in, dec_in, mark_enc, mark_dec, labels = self.get_input_data(x_enc, y_enc, x_mark_enc, x_mark_dec)
            return self.probability_forecast(enc_in, dec_in, mark_enc, mark_dec, labels=labels)  # return loss list
        return None

    # noinspection PyUnusedLocal
    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, probability_range=None):
        if self.task_name == 'probability_forecast':
            enc_in, dec_in, mark_enc, mark_dec, _ = self.get_input_data(x_enc, y_enc, x_mark_enc, x_mark_dec)
            return self.probability_forecast(enc_in, dec_in, mark_enc, mark_dec, probability_range=probability_range)
        return None

    # noinspection DuplicatedCode
    def run_lstm_enc(self, x, hidden, cell):
        if self.use_cnn:
            x = self.cnn_enc(x)  # [96, 256, 5]

        _, (hidden, cell) = self.lstm_enc(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

        return hidden, cell

    # noinspection DuplicatedCode
    def run_lstm_dec(self, x, x_mark_dec_step, hidden, cell, enc_hidden_attn):
        if self.use_cnn:
            x = self.cnn_dec(x)  # [96, 256, 5]

        _, (hidden, cell) = self.lstm_dec(x, (hidden, cell))  # [1, 256, 64], [1, 256, 64]

        if self.use_attn is not None:
            if self.attention_projection is not None:
                dec_hidden_attn = hidden.clone().view(self.batch_size, 1, self.dec_lstm_layers * self.lstm_hidden_size)
                if self.attention_projection == 'ap1':
                    dec_hidden_attn = self.dec_embedding(dec_hidden_attn, x_mark_dec_step)
                else:
                    dec_hidden_attn = self.dec_embedding(dec_hidden_attn)
            else:
                dec_hidden_attn = hidden.clone()

            dec_hidden_attn = dec_hidden_attn.view(self.batch_size, self.L_dec, self.H, self.E_dec)  # [256, 1, 2, 20]

            if self.use_norm:
                enc_hidden_attn = self.enc_norm(enc_hidden_attn)
                dec_hidden_attn = self.dec_norm(dec_hidden_attn)

            y, attn = self.attention(dec_hidden_attn, enc_hidden_attn, enc_hidden_attn, None)

            if self.out_projection is not None:
                y = self.out_projection(y)

            if self.use_norm:
                y = self.out_norm(y)

            y = y.view(self.dec_lstm_layers, self.batch_size, -1)

            if self.dec_hidden_separate:
                return y, hidden, cell, attn
            else:
                return y, y, cell, attn
        else:
            return hidden, hidden, cell, None

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
            labels = labels.permute(1, 0)  # [12, 256]

        # hidden and cell are initialized to zero
        hidden = torch.zeros(self.enc_lstm_layers, batch_size, self.lstm_hidden_size,
                             device=device)  # [2, 256, 40]
        cell = torch.zeros(self.enc_lstm_layers, batch_size, self.lstm_hidden_size, device=device)  # [2, 256, 40]

        # run encoder
        enc_hidden = torch.zeros(self.pred_start, self.enc_lstm_layers, batch_size, self.lstm_hidden_size,
                                 device=device)  # [96, 1, 256, 40]
        enc_cell = torch.zeros(self.pred_start, self.enc_lstm_layers, batch_size, self.lstm_hidden_size,
                               device=device)  # [96, 1, 256, 40]
        for t in range(self.pred_start):
            hidden, cell = self.run_lstm_enc(enc_in[t].unsqueeze_(0).clone(), hidden, cell)  # [2, 256, 40], [2, 256, 40]
            enc_hidden[t] = hidden
            enc_cell[t] = cell

        # only select the last hidden state
        if self.use_attn:
            if self.dec_hidden_difference1:
                enc_hidden_1_n = enc_hidden  # [96, 1, 256, 40]
                enc_hidden_0_1n = torch.concat(
                    (torch.zeros(1, 1, batch_size, self.lstm_hidden_size, device=device),
                     enc_hidden[:-1]), dim=0)
                enc_hidden = enc_hidden_1_n - enc_hidden_0_1n  # [96, 1, 256, 40]
                cell_hidden_1_n = enc_cell  # [96, 1, 256, 40]
                cell_hidden_0_1n = torch.concat(
                    (torch.zeros(1, 1, batch_size, self.lstm_hidden_size, device=device),
                     enc_cell[:-1]), dim=0)
                enc_cell = cell_hidden_1_n - cell_hidden_0_1n  # [96, 1, 256, 40]

            # embedding encoder
            if self.attention_projection is not None:
                enc_hidden = enc_hidden.view(batch_size, self.pred_start,
                                             self.enc_lstm_layers * self.lstm_hidden_size)
                if self.attention_projection == 'ap1':
                    enc_hidden = self.enc_embedding(enc_hidden, mark_enc)
                else:
                    enc_hidden = self.enc_embedding(enc_hidden)
                enc_hidden_attn = enc_hidden.view(batch_size, self.L_enc, self.H, self.E_enc)  # [256, 96, 8, 5]
            else:
                enc_hidden_attn = enc_hidden.view(batch_size, self.L_enc, self.H, self.E_enc)  # [256, 96, 8, 5]

            if self.dec_hidden_zero:
                dec_hidden = torch.zeros(self.dec_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
                dec_cell = torch.zeros(self.dec_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
            else:
                dec_hidden = enc_hidden[-1, -self.dec_lstm_layers:, :, :]  # [1, 256, 40]
                dec_cell = enc_cell[-1, -self.dec_lstm_layers:, :, :]  # [1, 256, 40]
        else:
            enc_hidden_attn = None

            dec_hidden = enc_hidden[-1, -self.dec_lstm_layers:, :, :]  # [1, 256, 40]
            dec_cell = enc_cell[-1, -self.dec_lstm_layers:, :, :]  # [1, 256, 40]

        if labels is not None:
            # train mode or validate mode
            hidden_permutes = torch.zeros(batch_size, self.pred_steps, self.qsqm_input_size, device=device)

            # initialize hidden and cell
            hidden, cell = dec_hidden.clone(), dec_cell.clone()

            # decoder
            for t in range(self.pred_steps):
                x_mark_dec_step = mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                hidden_qsqm, hidden, cell, _ = self.run_lstm_dec(dec_in[t].unsqueeze_(0).clone(), x_mark_dec_step,
                                                                 hidden, cell, enc_hidden_attn)
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
            samples_low = torch.zeros(probability_range_len, batch_size, self.pred_steps,
                                      device=device)  # [3, 256, 16]
            samples_high = samples_low.clone()  # [3, 256, 16]
            samples = torch.zeros(self.sample_times, batch_size, self.pred_steps, device=device)  # [99, 256, 12]

            label_len = self.pred_steps - self.pred_len

            for j in range(self.sample_times + probability_range_len * 2):
                # clone test batch
                x_dec_clone = dec_in.clone()  # [16, 256, 7]

                # initialize hidden and cell
                hidden, cell = dec_hidden.clone(), dec_cell.clone()

                # decoder
                for t in range(self.pred_steps):
                    x_mark_dec_step = mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                    hidden_qsqm, hidden, cell, _ = self.run_lstm_dec(dec_in[t].unsqueeze_(0).clone(), x_mark_dec_step,
                                                                     hidden, cell, enc_hidden_attn)
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

                    pred = sample_pred(self.alpha_prime_k, pred_alpha, self._lambda, gamma, eta_k, self.algorithm_type)
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
            if self.use_attn:
                attention_map = torch.zeros(self.pred_steps, batch_size, self.H, self.L_dec, self.L_enc,
                                            device=device)
            else:
                attention_map = None

            # clone test batch
            x_dec_clone = dec_in.clone()  # [16, 256, 7]

            # sample
            samples_mu1 = torch.zeros(batch_size, self.pred_steps, 1, device=device)

            # initialize parameters
            samples_lambda, samples_gamma, samples_eta_k = self.initialize_sample_parameters(batch_size, device)

            # initialize hidden and cell
            hidden, cell = dec_hidden.clone(), dec_cell.clone()

            # decoder
            for t in range(self.pred_steps):
                x_mark_dec_step = mark_dec[:, t, :].unsqueeze(1).clone()  # [256, 1, 5]
                hidden_qsqm, hidden, cell, attn = self.run_lstm_dec(dec_in[t].unsqueeze_(0).clone(), x_mark_dec_step,
                                                                    hidden, cell, enc_hidden_attn)
                if self.use_attn:
                    attention_map[t] = attn
                hidden_permute = self.get_hidden_permute(hidden_qsqm)
                gamma, eta_k = self.get_qsqm_parameter(hidden_permute)

                pred = sample_pred(self.alpha_prime_k, None, self._lambda, gamma, eta_k, self.algorithm_type)
                samples_lambda[t] = self._lambda
                samples_gamma[t] = gamma
                samples_eta_k[t] = eta_k
                samples_mu1[:, t, 0] = pred

                if t >= label_len:
                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            x_dec_clone[t + 1, :, self.lag_index[0]] = pred

            if not sample:
                # use integral to calculate the mean
                return (samples, samples_mu1, samples_std, samples_high, samples_low, attention_map,
                        (samples_lambda, samples_gamma, samples_eta_k))
            else:
                # use uniform samples to calculate the mean
                return (samples, samples_mu, samples_std, samples_high, samples_low, attention_map,
                        (samples_lambda, samples_gamma, samples_eta_k))
