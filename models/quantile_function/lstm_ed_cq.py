import torch
import torch.nn as nn

from models.quantile_function.lstm_cq import ConvLayer, sample_qsqm


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

        self.lstm_hidden_size = params.lstm_hidden_size
        self.enc_lstm_layers = params.lstm_layers
        self.dec_lstm_layers = 1  # TODO: Check self.enc_lstm_layers!!
        self.sample_times = params.sample_times
        self.lstm_dropout = params.dropout
        self.num_spline = params.num_spline
        self.pred_start = params.seq_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start

        custom_params = params.custom_params
        assert isinstance(custom_params, str)
        if custom_params.startswith('cnn'):
            self.use_cnn = True
            custom_params = custom_params[3:]
        else:
            self.use_cnn = False
        self.enc_feature = custom_params[0]
        self.dec_feature = custom_params[1]
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

        # QSQM
        self._lambda = -1e-3  # make sure all data is not on the left point
        if self.algorithm_type == '2':
            self.linear_gamma = nn.Linear(self.lstm_hidden_size * self.dec_lstm_layers, 1)
        elif self.algorithm_type == '1+2':
            self.linear_gamma = nn.Linear(self.lstm_hidden_size * self.dec_lstm_layers, self.num_spline)
        elif self.algorithm_type == '1':
            self.linear_gamma = nn.Linear(self.lstm_hidden_size * self.dec_lstm_layers, self.num_spline)
        else:
            raise ValueError("algorithm_type must be '1', '2', or '1+2'")
        self.linear_eta_k = nn.Linear(self.lstm_hidden_size * self.dec_lstm_layers, self.num_spline)
        self.soft_plus = nn.Softplus()  # make sure parameter is positive
        device = torch.device("cuda" if params.use_gpu else "cpu")
        y = torch.ones(self.num_spline) / self.num_spline
        self.alpha_prime_k = y.repeat(params.batch_size, 1).to(device)  # [256, 20]

        # Reindex
        self.new_index = None
        self.lag_index = None
        self.cov_index = None

    def get_input_size(self, feature, enc):
        if feature == 'A':
            return self.enc_in + self.lag if enc else self.enc_in + self.lag - 1
        elif feature == 'C':
            return self.enc_in - 1
        elif feature == 'L':
            return self.lag
        else:
            raise ValueError("feature must be 'A', 'C', or 'L'")

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

        batch = torch.cat((x_enc, y_enc), dim=1)

        if self.enc_feature == 'A':
            enc_in = batch[:, :self.pred_start, :]
        elif self.enc_feature == 'C':
            enc_in = batch[:, :self.pred_start, self.cov_index]
        elif self.enc_feature == 'L':
            enc_in = batch[:, :self.pred_start, self.lag_index]
        else:
            raise ValueError("enc_feature must be 'A', 'C', or 'L'")

        if self.dec_feature == 'A':
            dec_in = batch[:, self.pred_start:, :-1]
        elif self.dec_feature == 'C':
            dec_in = batch[:, self.pred_start:, self.cov_index]
        elif self.dec_feature == 'L':
            dec_in = batch[:, self.pred_start:, self.lag_index]
        else:
            raise ValueError("dec_feature must be 'A', 'C', or 'L'")

        labels = batch[:, self.pred_start:, -1]

        return enc_in, dec_in, labels

    # noinspection DuplicatedCode,PyUnusedLocal
    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'probability_forecast':
            # we don't need to use mark data because lstm can handle time series relation information
            enc_in, dec_in, labels = self.get_input_data(x_enc, y_enc)
            return self.probability_forecast(enc_in, dec_in, labels)  # return loss list
        return None

    # noinspection PyUnusedLocal
    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, probability_range=None):
        if self.task_name == 'probability_forecast':
            enc_in, dec_in, _ = self.get_input_data(x_enc, y_enc)
            return self.probability_forecast(enc_in, dec_in, probability_range=probability_range)
        return None

    # noinspection DuplicatedCode
    def run_lstm_enc(self, x, hidden, cell):
        if self.use_cnn:
            x = self.cnn_enc(x)  # [96, 256, 5]

        _, (hidden, cell) = self.lstm_enc(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

        return hidden, cell

    # noinspection DuplicatedCode
    def run_lstm_dec(self, x, hidden, cell):
        if self.use_cnn:
            x = self.cnn_dec(x)  # [96, 256, 5]

        _, (hidden, cell) = self.lstm_dec(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

        return hidden, cell

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
    def probability_forecast(self, x_enc, x_dec, labels=None, sample=False, probability_range=None):
        # [256, 96, 4], [256, 12, 7], [256, 12,]
        if probability_range is None:
            probability_range = [0.5]

        batch_size = x_enc.shape[0]  # 256
        device = x_enc.device

        assert isinstance(probability_range, list)
        probability_range_len = len(probability_range)
        probability_range = torch.Tensor(probability_range).to(device)  # [3]

        x_enc = x_enc.permute(1, 0, 2)  # [96, 256, 4]
        x_dec = x_dec.permute(1, 0, 2)  # [16, 256, 7]
        if labels is not None:
            labels = labels.permute(1, 0)  # [12, 256]

        # hidden and cell are initialized to zero
        hidden = torch.zeros(self.enc_lstm_layers, batch_size, self.lstm_hidden_size, device=device)  # [2, 256, 40]
        cell = torch.zeros(self.enc_lstm_layers, batch_size, self.lstm_hidden_size, device=device)  # [2, 256, 40]

        # run encoder
        enc_hidden = torch.zeros(self.pred_start, self.enc_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        enc_cell = torch.zeros(self.pred_start, self.enc_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        for t in range(self.pred_start):
            hidden, cell = self.run_lstm_enc(x_enc[t].unsqueeze_(0).clone(), hidden, cell)  # [2, 256, 40], [2, 256, 40]
            enc_hidden[t] = hidden
            enc_cell[t] = cell

        # only select the last hidden state
        hidden = enc_hidden[-1, -self.dec_lstm_layers:, :, :]  # [1, 256, 40]
        cell = enc_cell[-1, -self.dec_lstm_layers:, :, :]  # [1, 256, 40]

        if labels is not None:
            # train mode or validate mode
            hidden_permutes = torch.zeros(batch_size, self.pred_steps, self.lstm_hidden_size * self.dec_lstm_layers,
                                          device=device)
            for t in range(self.pred_steps):
                hidden, cell = self.run_lstm_dec(x_dec[t].unsqueeze_(0).clone(), hidden, cell)
                hidden_permute = self.get_hidden_permute(hidden)
                hidden_permutes[:, t, :] = hidden_permute

                # check if hidden contains NaN
                if torch.isnan(hidden).sum() > 0:
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
            samples_low = torch.zeros(probability_range_len, batch_size, self.pred_steps, device=device)  # [3, 256, 16]
            samples_high = samples_low.clone()  # [3, 256, 16]
            samples = torch.zeros(self.sample_times, batch_size, self.pred_steps, device=device)  # [99, 256, 12]

            hidden_init = hidden.clone()
            cell_init = cell.clone()

            for j in range(self.sample_times + probability_range_len * 2):
                # clone test batch
                x_dec_clone = x_dec.clone()  # [16, 256, 7]

                # initialize hidden and cell
                hidden, cell = hidden_init.clone(), cell_init.clone()

                # decoder
                for t in range(self.pred_steps):
                    hidden, cell = self.run_lstm_dec(x_dec_clone[t].unsqueeze(0), hidden, cell)
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

                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            x_dec_clone[t + 1, :, self.lag_index[0]] = pred

            samples_mu = torch.mean(samples, dim=0).unsqueeze(-1)  # mean or median ? # [256, 12, 1]
            samples_std = samples.std(dim=0).unsqueeze(-1)  # [256, 12, 1]

            # use integral to calculate the mean
            if not sample:
                # clone test batch
                x_dec_clone = x_dec.clone()  # [16, 256, 7]

                # sample
                samples_mu = torch.zeros(batch_size, self.pred_steps, 1, device=device)

                # initialize hidden and cell
                hidden, cell = hidden_init, cell_init

                # decoder
                for t in range(self.pred_steps):
                    hidden, cell = self.run_lstm_dec(x_dec_clone[t].unsqueeze(0), hidden, cell)
                    hidden_permute = self.get_hidden_permute(hidden)
                    gamma, eta_k = self.get_qsqm_parameter(hidden_permute)

                    pred = sample_qsqm(self.alpha_prime_k, None, self._lambda, gamma, eta_k, self.algorithm_type)
                    samples_mu[:, t, 0] = pred

                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            x_dec_clone[t + 1, :, self.lag_index[0]] = pred

            return samples, samples_mu, samples_std, samples_high, samples_low
