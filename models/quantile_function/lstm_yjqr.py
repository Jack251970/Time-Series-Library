import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, params):
        """
        LSTM-YJQR
        """
        super(Model, self).__init__()
        self.task_name = params.task_name
        input_size = params.enc_in + params.lag - 1  # take lag into account
        self.lstm_input_size = input_size
        self.lstm_hidden_dim = params.lstm_hidden_size
        self.lstm_layers = params.lstm_layers
        self.sample_times = params.sample_times
        self.lstm_dropout = params.dropout
        self.num_spline = params.num_spline
        self.pred_start = params.seq_len
        self.pred_len = params.pred_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start

        # LSTM
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

        # YJQM
        self.qsqm_input_size = self.lstm_hidden_dim * self.lstm_layers
        self.pre_lamda = nn.Linear(self.qsqm_input_size, 1)
        self.pre_mu = nn.Linear(self.qsqm_input_size, 1)
        self.pre_sigma = nn.Linear(self.qsqm_input_size, 1)

        self.lamda = nn.LeakyReLU(negative_slope=0.5)  # TODO

        self.mu = nn.Sigmoid()

        self.sigma = nn.Softplus()
        # self.sigma = nn.Sigmoid()
        # self.sigma = nn.ReLU()

        # Reindex
        self.new_index = [0]

    # noinspection DuplicatedCode
    def get_input_data(self, x_enc, y_enc):
        batch = torch.cat((x_enc, y_enc[:, -self.pred_len:, :]), dim=1)
        train_batch = batch[:, :, :-1]
        labels_batch = batch[:, :, -1]
        return train_batch, labels_batch

    # noinspection DuplicatedCode,PyUnusedLocal
    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'probability_forecast':
            # we don't need to use mark data because lstm can handle time series relation information
            train_batch, labels_batch = self.get_input_data(x_enc, y_enc)
            return self.probability_forecast(train_batch, labels_batch)  # return loss list
        return None

    # noinspection PyUnusedLocal,DuplicatedCode
    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, probability_range=None):
        if self.task_name == 'probability_forecast':
            train_batch, _ = self.get_input_data(x_enc, y_enc)
            if probability_range is None:
                probability_range = [0.5]
            return self.probability_forecast(train_batch, probability_range=probability_range)
        return None

    # noinspection DuplicatedCode
    def run_lstm(self, x, hidden, cell):
        _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

        return hidden, cell

    @staticmethod
    def get_hidden_permute(hidden):
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(1, 2, 0)  # [256, 2, 40]
        hidden_permute = hidden_permute.contiguous().view(hidden.shape[1], -1)  # [256, 80]

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
    def probability_forecast(self, train_batch, labels_batch=None, sample=True,
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
                    loss_list.clear()
                    stop_flag = True
                    break
                lamda, mu, sigma = self.get_yjqm_parameter(hidden_permute)  # [256, 1], [256, 1], [256, 1]
                y = labels_batch[t].clone()  # [256,]
                # lamba,mu，sigma->(256,)
                loss_list.append((lamda.squeeze(), mu.squeeze(), sigma.squeeze(), y))

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
                        pred_alpha = uniform.sample(torch.Size([batch_size]))  # [256, 1]

                    pred = sample_yjqr(lamda, mu, sigma, pred_alpha)
                    if j < probability_range_len:
                        samples_low[j, :, t] = pred
                    elif j < 2 * probability_range_len:
                        samples_high[j - probability_range_len, :, t] = pred
                    else:
                        samples[j - probability_range_len * 2, :, t] = pred

                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            test_batch[self.pred_start + t + 1, :, self.new_index[0]] = pred

            samples_mu = torch.mean(samples, dim=0).unsqueeze(-1)  # mean or median ? # [256, 12, 1]
            samples_std = samples.std(dim=0).unsqueeze(-1)  # [256, 12, 1]

            # use integral to calculate the mean
            if not sample:
                # sample
                samples_mu = torch.zeros(batch_size, self.pred_steps, 1, device=device)

                # initialize hidden and cell
                hidden, cell = hidden_init, cell_init

                # prediction range
                test_batch = train_batch
                for t in range(self.pred_steps):
                    hidden, cell = self.run_lstm(test_batch[self.pred_start + t].unsqueeze(0), hidden, cell)
                    hidden_permute = self.get_hidden_permute(hidden)
                    lamda, mu, sigma = self.get_yjqm_parameter(hidden_permute)

                    pred = sample_yjqr(lamda, mu, sigma, None)
                    samples_mu[:, t, 0] = pred

                    for lag in range(self.lag):
                        if t < self.pred_steps - lag - 1:
                            test_batch[self.pred_start + t + 1, :, self.new_index[0]] = pred

            return samples, samples_mu, samples_std, samples_high, samples_low, None, None


def loss_fn(tuple_param):
    lamda, mu, sigma, labels = tuple_param

    # 计算损失函数
    # lambda,mu,log_sigma,labels=(256,)
    batch_size = labels.shape[0]
    trans_y = torch.zeros_like(labels, device=mu.device)
    y = labels.squeeze()

    # 使用 torch 的条件语句进行批处理
    mask_y_ge_0 = (y >= 0).squeeze()
    mask_y_lt_0 = (~mask_y_ge_0).squeeze()
    mask_lamda_ne_0 = (lamda != 0)
    mask_lamda_ne_2 = (lamda != 2)

    # 计算 trans_y
    trans_y[mask_y_ge_0 & mask_lamda_ne_0] = ((y[mask_y_ge_0 & mask_lamda_ne_0] + 1).pow(
        lamda[mask_y_ge_0 & mask_lamda_ne_0]) - 1) / lamda[mask_y_ge_0 & mask_lamda_ne_0]
    trans_y[mask_y_ge_0 & ~mask_lamda_ne_0] = torch.log(y[mask_y_ge_0 & ~mask_lamda_ne_0] + 1)
    trans_y[mask_y_lt_0 & mask_lamda_ne_2] = -(
            (1 - y[mask_y_lt_0 & mask_lamda_ne_2]).pow(2 - lamda[mask_y_lt_0 & mask_lamda_ne_2]) - 1) / (
                                                     2 - lamda[mask_y_lt_0 & mask_lamda_ne_2])
    trans_y[mask_y_lt_0 & ~mask_lamda_ne_2] = -torch.log(1 - y[mask_y_lt_0 & ~mask_lamda_ne_2])

    # 计算损失
    L1 = batch_size * 0.5 * torch.log(torch.tensor(2 * torch.pi))
    L2 = batch_size * 0.5 * 2 * torch.log(sigma)
    L3 = 0.5 * sigma.pow(-2) * (trans_y - mu).pow(2)
    L4 = (lamda - 1) * torch.sum(torch.sign(labels) * torch.log(torch.abs(labels) + 1))
    Ln = L1 + L2 + L3 - L4

    # loss=() 为一个数
    loss = torch.mean(Ln)
    return loss


def sample_yjqr(lamda, mu, sigma, alpha):
    device = mu.device

    if alpha is not None:
        # 如果输入分位数值，则直接计算对应分位数的预测值
        normal_dist = torch.distributions.Normal(0, 1)
        pred_cdf = normal_dist.icdf(alpha).to(device)  # TODO: 增加一个放大因子

        # pred_cdf = alpha_new * torch.ones(lamda.shape[0], device=device)
        y_deal = (mu + sigma * pred_cdf)
        pred = pred_output(y_deal.squeeze(), lamda.squeeze(), mu.squeeze())

        # pred=(256,)
        return pred
    else:
        # 如果未输入分位数值，则从积分值获取预测值的平均
        raise NotImplementedError


def pred_output(y_deal, lamda, mu):
    mask_y_ge_0 = y_deal >= 0
    mask_y_lt_0 = ~mask_y_ge_0
    mask_lamda_ne_0 = lamda != 0
    mask_lamda_ne_2 = lamda != 2

    # 初始化 y_pred
    y_pred = torch.zeros_like(y_deal, device=mu.device)

    # 计算 y_pred
    y_pred[mask_y_ge_0 & mask_lamda_ne_0] = ((
            y_deal[mask_y_ge_0 & mask_lamda_ne_0] * lamda[mask_y_ge_0 & mask_lamda_ne_0] + 1).pow(
        1 / lamda[mask_y_ge_0 & mask_lamda_ne_0])) - 1
    y_pred[mask_y_ge_0 & ~mask_lamda_ne_0] = torch.exp(y_deal[mask_y_ge_0 & ~mask_lamda_ne_0]) - 1
    y_pred[mask_y_lt_0 & mask_lamda_ne_2] = 1 - (
        ((lamda[mask_y_lt_0 & mask_lamda_ne_2] - 2) * y_deal[mask_y_lt_0 & mask_lamda_ne_2] + 1).pow(
            1 / (2 - lamda[mask_y_lt_0 & mask_lamda_ne_2])))
    y_pred[mask_y_lt_0 & ~mask_lamda_ne_2] = 1 - torch.exp(-y_deal[mask_y_lt_0 & ~mask_lamda_ne_2])
    return y_pred
