import math

import torch
import torch.nn as nn
from einops import rearrange


# You can know more in https://zhuanlan.zhihu.com/p/374936725.
# The function of embedding is to add more information of the original data into the model.
# It will expand the dimension.

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding,
                                   padding_mode='circular', bias=False)

        # After the convolutional layer is created, it is initialized with Kaiming normal initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: shape [32, 16, 14]
        # 32 is the batch size, 16 is the sequence length (time steps), and 14 is the feature dimension.
        x = x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x = x.transpose(1, 2)  # back to original dimension sequence
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        This class is used to embed various time-related features such as minute, hour, weekday, day, and month. The
        size of each feature is predefined (e.g., minute_size = 4, hour_size = 24, etc.). Depending on the embed_type
        parameter, it uses either FixedEmbedding or PyTorch's nn.Embedding to create the embeddings.
        """
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        # calculate each embedding
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # sum of all these embeddings
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        This class is used to embed time-related features based on a frequency map. The frequency map is a dictionary
        that maps different frequency types (e.g., 'h', 't', 's', etc.) to their corresponding dimensions. The
        embed_type parameter is stored but not used in this class.
        """
        super(TimeFeatureEmbedding, self).__init__()
        self.embed_type = embed_type
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        # x: [32, 16, 14]
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        The series-wise connection inherently contains the sequential information.
        Thus, we can discard the position embedding of transformers.
        """
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DSWEmbedding(nn.Module):
    """
    Dimension-Segment-Wise (DSW) embedding
    Paper: CROSSFORMER: TRANSFORMER UTILIZING CROSS-DIMENSION DEPENDENCY FOR MULTIVARIATE TIME SERIES FORECASTING
    """

    def __init__(self, seg_len, d_model, pos_embed=False):
        super(DSWEmbedding, self).__init__()
        self.seg_len = seg_len
        self.d_model = d_model

        # CHANGE: We set the bias to False compared to the codes of the original paper.
        self.value_embedding = nn.Linear(seg_len, d_model, bias=False)

        # CHANGE: The codes of the original paper do not have the positional embedding.
        self.pos_embed = pos_embed
        if pos_embed:
            self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x):
        """
        x_padding : [B, L, F]
        B: batch size, L: (sequence length + padding length), F: feature dimension
        """
        B, _, F = x.shape  # [32, 24, 14]

        if self.pos_embed:
            # segment the input sequence and flatten the batch and feature dimensions
            x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d) seg_num seg_len', b=B, d=F,
                                  seg_len=self.seg_len)  # [32, (2 * 12), 14] -> [448, 2, 12]

            # get the value embedding
            e_v = self.value_embedding(x_segment)  # [448, 2, 512]

            # get the position embedding
            e_p = self.position_embedding(x_segment)  # [1, 2, 512]

            x_embed = e_v + e_p  # [448, 2, 512]

            # reshape the embedded sequence
            x_embed = rearrange(x_embed, '(b d) seg_num d_model -> b d seg_num d_model', b=B, d=F,
                                d_model=self.d_model)  # [(32 * 14), 2, 512] -> [32, 14, 2, 512]
        else:
            # segment the input sequence and flatten the batch, feature and segment length dimensions
            # [32, (2 * 12), 14] -> [896, 12]
            x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)

            # get the value embedding
            x_embed = self.value_embedding(x_segment)  # [896, 512]

            # reshape the embedded sequence: [(32 * 14 * 2), 512] -> [32, 14, 2, 512]
            x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=B, d=F)

        return x_embed


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : [B, F, S]
        B: batch size, F: feature dimension, S: sequence length
        """
        B, F, S = x.shape  # [32, 14, 16]

        # padding in the end of the sequence
        x = self.padding_patch_layer(x)  # [32, 14, 24]

        # unfold the sequence (segment it)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [32, 14, 2, 12]

        # flatten the batch dimension and feature dimension
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [32, 14, 2, 12] -> [448, 2, 12]

        # input encoding
        e_v = self.value_embedding(x)  # [448, 2, 512]
        e_p = self.position_embedding(x)  # [1, 2, 512]
        x = e_v + e_p

        return self.dropout(x), F
