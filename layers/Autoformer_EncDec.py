import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Special designed layer normalization for the seasonal part
    """

    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):  # [32, 16, 512]
        x_hat = self.layer_norm(x)  # [32, 16, 512]
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)  # [32, 16, 512]
        return x_hat - bias  # [32, 16, 512]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size=25, stride=1, padding=0):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    # noinspection DuplicatedCode
    def forward(self, x):  # x: shape [32, 24, 14]
        # Select the first time step from each sequence in the batch and repeat it along the time axis
        # This is done to pad the sequence at the beginning.
        front = x[:, 0:1, :].repeat(1, self.kernel_size - (self.kernel_size + 1) // 2, 1)  # front: shape [32, 12, 14]

        # Select the last time step from each sequence in the batch and repeat it along the time axis
        # This is done to pad the sequence at the end.
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)  # end: shape [32, 12, 14]

        # concatenates the front, x, and end tensors along the time axis
        # This results in a sequence which has padding on the both ends of time series
        x = torch.cat([front, x, end], dim=1)  # x: shape [32, 48, 14]

        # This line first permutes the dimensions of x to bring the time axis to the last dimension
        x = x.permute(0, 2, 1)  # x: shape [32, 14, 48]

        # Apply the moving average operation on the permuted x
        x = self.avg(x)  # x: shape [32, 14, 24]

        # This line permutes the dimensions of x back to their original order
        x = x.permute(0, 2, 1)  # x: shape: [32, 24, 14]

        return x


class exponential_smoothing(nn.Module):
    """
    Exponential smoothing block to highlight the trend of time series
    """

    def __init__(self, alpha=0.5):
        super(exponential_smoothing, self).__init__()
        self.alpha = alpha

    def forward(self, x):  # x: shape [32, 24, 14]
        # This line first permutes the dimensions of x to bring the time axis to the last dimension
        x = x.permute(0, 2, 1)  # x: shape [32, 14, 24]

        # Apply the exponential smoothing operation on the permuted x
        out = torch.zeros_like(x)
        out[:, :, 0] = x[:, :, 0]
        for i in range(1, x.shape[2]):
            out[:, :, i] = (1 - self.alpha) * out[:, :, i - 1] + self.alpha * x[:, :, i]

        # This line permutes the dimensions of x back to their original order
        out = out.permute(0, 2, 1)  # x: shape: [32, 24, 14]

        return out


class stl_decomposition(nn.Module):
    """
    Stl decomposition block to highlight the trend of time series
    """

    def __init__(self, period=25):
        super(stl_decomposition, self).__init__()
        self.period = period

    def forward(self, x):  # x: shape [32, 24, 14]
        # This line first permutes the dimensions of x to bring the time axis to the last dimension
        x = x.permute(0, 2, 1)  # x: shape [32, 14, 16]

        # Apply the stl decomposition from statsmodels.tsa.seasonal to operate on the permuted seasonal, trend, resid
        seasonal = torch.zeros_like(x)
        trend = torch.zeros_like(x)
        resid = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                data = x[i, j, :]  # {Tensor: (16,)}
                data = data.tolist()  # {List: (16,)}
                data = pd.Series(data, index=pd.date_range("2018-09-14 00:00:00", periods=len(data), freq="15T"),
                                 name="data")  # {DataFrame: (16,)}
                stl = STL(data, period=self.period)
                data = stl.fit()

                seasonal_data = data.seasonal  # {Series: (16,)}
                trend_data = data.trend  # {Series: (16,)}
                resid_data = data.resid  # {Series: (16,)}

                seasonal_data = torch.from_numpy(seasonal_data.values)  # {Tensor: (16,)}
                trend_data = torch.from_numpy(trend_data.values)  # {Tensor: (16,)}
                resid_data = torch.from_numpy(resid_data.values)  # {Tensor: (16,)}

                # convert to torch.float32
                seasonal_data = torch.clone(seasonal_data).detach().to(torch.float32)
                trend_data = torch.clone(trend_data).detach().to(torch.float32)
                resid_data = torch.clone(resid_data).detach().to(torch.float32)

                seasonal[i, j, :] = seasonal_data
                trend[i, j, :] = trend_data
                resid[i, j, :] = resid_data

        # This line permutes the dimensions of seasonal, trend, resid back to their original order
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        resid = resid.permute(0, 2, 1)

        return seasonal, trend, resid


class adapt_smoothing(nn.Module):
    """
    Adaptive smoothing block to highlight the trend of time series
    """

    def __init__(self, kernel_size=25, stride=1, padding=0, norm_weights=True, norm_operation=True):
        super(adapt_smoothing, self).__init__()
        self.kernel_size = kernel_size
        self.norm_operation = norm_operation

        self.template = nn.Linear(in_features=kernel_size, out_features=1, bias=False)

        # TODO: Use the Conv1d instead.
        # self.template = nn.Conv1d(in_channels=kernel_size, out_channels=1, kernel_size=1, bias=False)

        # init the weights of the template
        if norm_weights:
            self.template.weight.data.fill_(1.0 / kernel_size)
        else:
            self.template.weight.data.fill_(1.0)

    # noinspection DuplicatedCode
    def forward(self, x):  # kernel_size: 3, x: shape [1, 4, 3]
        # Get the size of the input data
        batch, sequence, feature = x.shape

        # Select the first time step from each sequence in the batch and repeat it along the time axis
        # This is done to pad the sequence at the beginning.
        front = x[:, 0:1, :].repeat(1, self.kernel_size - (self.kernel_size + 1) // 2, 1)

        # Select the last time step from each sequence in the batch and repeat it along the time axis
        # This is done to pad the sequence at the end.
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        # concatenates the front, x, and end tensors along the time axis.
        # This results in a sequence which has padding on the both ends of time series
        x = torch.cat([front, x, end], dim=1)  # x: shape [1, 6, 3]

        # This line first permutes the dimensions of x to bring the time axis to the last dimension
        x = x.permute(0, 2, 1)  # x: shape [1, 3, 6]

        # Make sure the dimensions of the input is 4
        x = x.unsqueeze(-1)  # x: shape [1, 3, 6, 1]

        # Apply the adaptive moving average operation on the permuted x

        # 1. Extracts sliding local blocks from a batched input data
        x = F.unfold(x, (sequence, 1), stride=1)  # shape: [1, 12, 3]
        # output shape: [batch, channel * kernel_size[0] * kernel_size[1], ...]
        # reference: https://blog.csdn.net/dongjinkun/article/details/116671168

        # 2. Apply adaptive template to the input data
        x = self.template(x)  # shape: [1, 12, 1]

        # 3. Normalize the output of the adaptive template
        if self.norm_operation:
            x = x / torch.sum(self.template.weight, dim=1)  # shape: [1, 12, 1]

        # 4. Restore the unfolded data to the original shape
        x = x.view(batch, feature, sequence)  # shape: [1, 3, 4]

        # This line permutes the dimensions of x back to their original order
        x = x.permute(0, 2, 1)  # shape: [1, 4, 3]

        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    moe: Mixture of Experts for Seasonal-Trend Decomposition in FEDformer
    """

    def __init__(self, kernel_size, series_decomp_mode='avg'):
        super(series_decomp, self).__init__()
        self.series_decomp_mode = series_decomp_mode
        # CHANGE: We use adaptive smoothing mechanism to extract trend data from the raw data
        if series_decomp_mode == 'avg':
            self.smoothing = moving_avg(kernel_size)
        elif series_decomp_mode == 'exp':
            self.smoothing = exponential_smoothing(kernel_size)
        elif series_decomp_mode == 'adp_avg':
            self.smoothing = adapt_smoothing(kernel_size)
        elif series_decomp_mode == 'stl':
            self.decomposition = stl_decomposition(period=kernel_size)
        elif series_decomp_mode == 'moe':
            self.decompositions = [moving_avg(kernel, stride=1) for kernel in kernel_size]
            self.layer = torch.nn.Linear(1, len(kernel_size))
        elif series_decomp_mode == 'none':
            self.decomposition = None

    def forward(self, x):  # shape: {Tensor: (32, 16, 14)}
        if self.series_decomp_mode == 'avg' or self.series_decomp_mode == 'exp' or self.series_decomp_mode == 'adp_avg':
            trend = self.smoothing(x)  # use smoothing to eliminate seasonal component
            seasonal = x - trend
            return seasonal, trend
        elif self.series_decomp_mode == 'stl':
            seasonal, trend, resid = self.decomposition(x)
            # for test
            # return seasonal, trend, resid
            # for experiment
            return seasonal, trend + resid
        elif self.series_decomp_mode == 'moe1':
            trend = []
            for smoothing in self.decompositions:
                _trend = smoothing(x)
                trend.append(_trend.unsqueeze(-1))
            trend = torch.cat(trend, dim=-1)
            trend = torch.sum(trend * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
            seasonal = x - trend
            return seasonal, trend
        elif self.series_decomp_mode == 'moe2':
            # By ZDandsomSP <1115854107@qq.com>
            trend_list = []
            season_list = []
            for smoothing in self.decompositions:
                trend = smoothing(x)
                season = x - trend
                trend_list.append(trend)
                season_list.append(season)
            season = sum(season_list) / len(season_list)
            trend = sum(trend_list) / len(trend_list)
            return season, trend
        elif self.series_decomp_mode == 'none':
            zero = torch.zeros(x.shape).to(x.device)
            # none seasonal
            # return zero, x
            # none trend
            return x, zero


# noinspection DuplicatedCode
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, _moving_avg=25, series_decomp_mode='avg', dropout=0.1,
                 activation="relu"):
        """

        :param attention:
        :param d_model:
        :param d_ff: numbers of feed forward layer
        :param _moving_avg:
        :param series_decomp_mode:
        :param dropout:
        :param activation:
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(_moving_avg, series_decomp_mode=series_decomp_mode)
        self.decomp2 = series_decomp(_moving_avg, series_decomp_mode=series_decomp_mode)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_en, attn_mask=None):
        # note: encoder focuses on the seasonal part modeling, so the seasonal output of the decomp will be deprecated.

        # attention calculation
        attn_x_en, attn = self.attention(
            x_en, x_en, x_en,
            attn_mask=attn_mask
        )

        # series decomposition
        s_en_1, _ = self.decomp1(x_en + self.dropout(attn_x_en))

        # feed forward: linear & activation
        ff_s_en_1 = s_en_1.transpose(-1, 1)
        ff_s_en_1 = self.conv1(ff_s_en_1)
        ff_s_en_1 = self.activation(ff_s_en_1)
        ff_s_en_1 = self.dropout(ff_s_en_1)
        ff_s_en_1 = self.conv2(ff_s_en_1)
        ff_s_en_1 = ff_s_en_1.transpose(-1, 1)

        # series decomposition
        s_en_2, _ = self.decomp2(s_en_1 + self.dropout(ff_s_en_1))

        return s_en_2, attn


# noinspection DuplicatedCode
class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # many encoder layers
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """

        :param x: [32, ]
        :param attn_mask:
        :return:
        """
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# noinspection DuplicatedCode
class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None, _moving_avg=25,
                 series_decomp_mode='avg', dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention  # here we use AutoCorrelation as the attention mechanism
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(_moving_avg, series_decomp_mode=series_decomp_mode)
        self.decomp2 = series_decomp(_moving_avg, series_decomp_mode=series_decomp_mode)
        self.decomp3 = series_decomp(_moving_avg, series_decomp_mode=series_decomp_mode)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_de, cross, x_mask=None, cross_mask=None):
        # attention calculation
        attn_x_de = self.dropout(self.self_attention(
            x_de, x_de, x_de,
            attn_mask=x_mask
        )[0])

        # series decomposition
        s_de_1, t_de_1 = self.decomp1(x_de + attn_x_de)

        # attention calculation
        attn_s_de_1_x_en = self.dropout(self.cross_attention(
            s_de_1, cross, cross,
            attn_mask=cross_mask
        )[0])

        # series decomposition
        s_de_2, t_de_2 = self.decomp2(s_de_1 + attn_s_de_1_x_en)

        # feed forward
        ff_s_de_2 = s_de_2.transpose(-1, 1)
        ff_s_de_2 = self.conv1(ff_s_de_2)
        ff_s_de_2 = self.activation(ff_s_de_2)
        ff_s_de_2 = self.dropout(ff_s_de_2)
        ff_s_de_2 = self.conv2(ff_s_de_2)
        ff_s_de_2 = ff_s_de_2.transpose(-1, 1)

        # series decomposition
        s_de_3, t_de_3 = self.decomp3(ff_s_de_2 + s_de_2)

        # sum trend component as the output
        t_de = t_de_1 + t_de_2 + t_de_3

        # project the deep transformed seasonal component to the target dimension.
        t_de = self.projection(t_de.permute(0, 2, 1)).transpose(1, 2)

        # return two decomposed component
        return s_de_3, t_de


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

            # sum trend component as the output
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend
