import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.task_name = params.task_name
        self.enc_in = params.enc_in
        self.c_out = params.c_out
        self.lstm_input_size = params.enc_in + params.lag - 1  # take lag into account
        self.lstm_hidden_size = params.lstm_hidden_size
        self.lstm_layers = params.lstm_layers
        self.lstm_dropout = params.dropout
        self.pred_start = params.seq_len
        self.pred_len = params.pred_len
        self.pred_steps = params.pred_len
        self.lag = params.lag
        self.train_window = self.pred_steps + self.pred_start

        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bias=True,
                            batch_first=False,
                            bidirectional=False,
                            dropout=self.lstm_dropout)
        self.init_lstm(self.lstm)

        if self.task_name == 'short_term_forecast' or 'long_term_forecast':
            self.out_projection = nn.Linear(self.lstm_layers * self.lstm_hidden_size, self.c_out)
        else:
            raise NotImplementedError()

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
        batch = torch.cat((x_enc, y_enc[:, -self.pred_len:, :]), dim=1)
        train_batch = batch[:, :, :-1]
        labels_batch = batch[:, :, :-self.lag+1]
        return train_batch, labels_batch

    # noinspection DuplicatedCode,PyUnusedLocal
    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'short_term_forecast' or 'long_term_forecast':
            train_batch, labels_batch = self.get_input_data(x_enc, y_enc)
            return self.probability_forecast(train_batch, labels_batch)  # [B, L1, D]
        else:
            raise NotImplementedError()

    # noinspection PyUnusedLocal
    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        if self.task_name == 'short_term_forecast' or 'long_term_forecast':
            train_batch, _ = self.get_input_data(x_enc, y_enc)
            return self.probability_forecast(train_batch)  # [B, L, D]
        return None

    # noinspection DuplicatedCode
    def run_lstm(self, x, hidden, cell):
        _, (hidden, cell) = self.lstm(x, (hidden, cell))  # [2, 256, 40], [2, 256, 40]

        return hidden, cell

    @staticmethod
    def get_hidden_permute(hidden):
        hidden_permute = hidden.permute(1, 2, 0)  # [256, 2, 40]
        hidden_permute = hidden_permute.contiguous().view(hidden.shape[1], -1)  # [256, 80]

        return hidden_permute

    # noinspection DuplicatedCode
    def probability_forecast(self, batch, labels_batch=None):  # [256, 112, 7], [256, 112,]
        batch_size = batch.shape[0]  # 256
        device = batch.device

        batch = batch.permute(1, 0, 2)  # [112, 256, 7]

        # hidden and cell are initialized to zero
        hidden = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)  # [2, 256, 40]
        cell = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)  # [2, 256, 40]

        if labels_batch is not None:
            # train mode or validate mode
            hidden_permutes = torch.zeros(batch_size, self.train_window, self.lstm_hidden_size * self.lstm_layers,
                                          device=device)
            for t in range(self.train_window):
                hidden, cell = self.run_lstm(batch[t].unsqueeze_(0).clone(), hidden, cell)
                hidden_permute = self.get_hidden_permute(hidden)  # [256, 80]
                hidden_permutes[:, t, :] = hidden_permute

                # check if hidden contains NaN
                if torch.isnan(hidden).sum() > 0:
                    print('Raise error!')
                    break

            # get loss list
            stop_flag = False
            output_steps = torch.zeros(batch_size, self.train_window, self.c_out, device=device)  # [112, 256, 7]
            for t in range(self.train_window):
                hidden_permute = hidden_permutes[:, t, :]  # [256, 80]
                if torch.isnan(hidden_permute).sum() > 0:
                    stop_flag = True
                    print('Raise error!')
                    break
                output_steps[:, t, :] = self.out_projection(hidden_permute)  # [256, 7]

            return (output_steps, labels_batch), stop_flag
        else:
            # condition range
            for t in range(self.pred_start):
                hidden, cell = self.run_lstm(batch[t].unsqueeze(0), hidden, cell)  # [2, 256, 40], [2, 256, 40]

            # prediction range
            output_steps = torch.zeros(batch_size, self.pred_steps, self.c_out, device=device)  # [16, 256, 7]
            for t in range(self.pred_steps):
                hidden, cell = self.run_lstm(batch[self.pred_start + t].unsqueeze(0), hidden, cell)
                hidden_permute = self.get_hidden_permute(hidden)
                pred = self.out_projection(hidden_permute)
                output_steps[:, t, :] = pred

                for lag in range(self.lag):
                    if t < self.pred_steps - lag - 1:
                        batch[self.pred_start + t + 1, :, 0] = pred

            return output_steps
