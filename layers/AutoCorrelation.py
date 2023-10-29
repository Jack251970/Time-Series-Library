import math

import torch
import torch.nn as nn

from layers.SelfAttention_Family import AttentionLayer


class SeqNorm(nn.Module):
    def __init__(self, dims, var=False):
        super(SeqNorm, self).__init__()
        self.dims = dims
        self.var = var

    # noinspection DuplicatedCode
    def forward(self, x):
        out = x
        for dim in self.dims:
            mean = torch.mean(x, dim=dim)
            mean = mean.unsqueeze(dim)
            if dim == 3:
                mean = mean.repeat(1, 1, 1, x.shape[3])
            elif dim == 2:
                mean = mean.repeat(1, 1, x.shape[2], 1)
            elif dim == 1:
                mean = mean.repeat(1, x.shape[1], 1, 1)
            else:
                raise NotImplementedError
            out = out - mean
            if self.var:
                var = torch.var(x, dim=dim)
                var = var.unsqueeze(dim)
                if dim == 3:
                    var = var.repeat(1, 1, 1, x.shape[3])
                elif dim == 2:
                    var = var.repeat(1, 1, x.shape[2], 1)
                elif dim == 1:
                    var = var.repeat(1, x.shape[1], 1, 1)
                else:
                    raise NotImplementedError
                out = out / torch.sqrt(var + 1e-8)
        return out


# noinspection PyMethodMayBeStatic
class SeqMean(nn.Module):
    def __init__(self):
        super(SeqMean, self).__init__()

    # noinspection DuplicatedCode
    def forward(self, x):
        mean = torch.mean(x, dim=1)
        mean = mean.unsqueeze(1)
        mean = mean.repeat(1, x.shape[1], 1, 1)
        return x - mean


# noinspection DuplicatedCode
class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False,
                 agg_mode='same_head', norm=False, mean_dim=None, norm_var=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.agg_mode = agg_mode

        self.norm = norm
        if norm is True:
            if mean_dim is None:
                mean_dim = [1]
            if mean_dim == [1] and norm_var is False:
                self.q_norm_layer = SeqMean()
                self.k_norm_layer = SeqMean()
                self.v_norm_layer = SeqMean()
            else:
                self.q_norm_layer = SeqNorm(mean_dim, norm_var)
                self.k_norm_layer = SeqNorm(mean_dim, norm_var)
                self.v_norm_layer = SeqNorm(mean_dim, norm_var)

    def time_delay_agg_full(self, values, r_qk):
        """
        Standard version of AutoCorrelation for learning
        """
        # init size
        batch, head, channel, length = values.shape  # 32, 8, 64, 16

        # init index
        init_index = torch.arange(length)  # shape: [16] (from 0 to 15)
        init_index = init_index.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 1, 16]
        init_index = init_index.repeat(batch, head, channel, 1)  # shape: [32, 8, 64, 16]
        init_index = init_index.to(values.device)

        # calculate the number of top elements to select, based on the logarithm of the sequence length
        k = int(self.factor * math.log(length))  # 5

        # select the largest elements from corr along the last dimension
        r_qk_top, tao = torch.topk(r_qk, k, dim=-1, largest=True)
        # [32, 8, 64, 5] (values), [32, 8, 64, 5] (indices)

        # apply the softmax function to r_qk_top along the last dimension, converting them into probabilities
        r_qk_softmax = torch.softmax(r_qk_top, dim=-1)  # [32, 8, 64, 5]

        # duplicate values along the last dimension
        tmp_values = values.repeat(1, 1, 1, 2)  # [32, 8, 64, 32]

        # create a tensor filled with zeros, having the same shape as values
        delays_agg = torch.zeros_like(values).float()  # [32, 8, 64, 16]

        # iterate over the range k
        for i in range(k):
            # get the index of the i-th element
            tao_k = tao[..., i].unsqueeze(-1)  # [32, 8, 64, 1]

            # calculate a temporary index by adding the i-th element of delay to init_index
            tmp_index = init_index + tao_k  # [32, 8, 64, 16]

            # gather elements from tmp_values along the last dimension, according to tmp_delay
            roll_v_tao = torch.gather(tmp_values, dim=-1, index=tmp_index)  # [32, 8, 64, 16]

            # update delays_agg by adding the product of rolled value and the i-th element of r_qk after softmax
            delays_agg = delays_agg + roll_v_tao * (r_qk_softmax[..., i].unsqueeze(-1))

        return delays_agg

    def time_delay_agg_same_head(self, values, r_qk):
        """
        Time Delay Aggregation.
        SpeedUp version of AutoCorrelation (a batch-normalization style design).
        Same for the training phase or the inference phase and every head.
        :param values: [32, 8, 64, 16]
        :param r_qk: [32, 8, 64, 16]
        :return:
        """

        # init size
        batch, head, channel, length = values.shape  # 32, 8, 64, 16

        # init index
        init_index = torch.arange(length)  # shape: [16] (from 0 to 15)
        init_index = init_index.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 1, 16]
        init_index = init_index.repeat(batch, head, channel, 1)  # shape: [32, 8, 64, 16]
        init_index = init_index.to(values.device)

        # calculate the number of top elements to select, based on the logarithm of the sequence length
        k = int(self.factor * math.log(length))  # 5

        # calculate the mean value of r_qk in every head
        mean_r_qk = torch.mean(r_qk, dim=1)  # [32, 64, 16]

        # select the largest elements from mean_r_qk along the time sequence dimension
        r_qk_top, tao = torch.topk(mean_r_qk, k, dim=-1, largest=True)  # [32, 64, 5] (values), [32, 64, 5] (indices)

        # apply the softmax function to r_qk_top along the time sequence dimension, converting them into probabilities
        r_qk_softmax = torch.softmax(r_qk_top, dim=-1)  # [32, 64, 5]

        # duplicate values along the last dimension
        tmp_values = values.repeat(1, 1, 1, 2)  # [32, 8, 64, 32]

        # create a tensor filled with zeros, having the same shape as values
        delays_agg = torch.zeros_like(values).float()  # [32, 8, 64, 16]

        for i in range(k):
            # get the index of the i-th element
            tao_k = tao[..., i]  # [32, 64]
            tao_k = tao_k.unsqueeze(1).unsqueeze(-1)  # [32, 1, 64, 1]
            tao_k = tao_k.repeat(1, head, 1, length)  # [32, 8, 64, 16]

            # calculate a temporary index by adding the i-th element of delay to init_index
            tmp_index = init_index + tao_k  # [32, 8, 64, 16] * 3

            # gather elements from tmp_values along the last dimension, according to tmp_index
            roll_v_tao = torch.gather(tmp_values, dim=-1, index=tmp_index)  # [32, 8, 64, 16]

            # get the i-th element of r_qk_softmax
            r_qk_softmax_i = r_qk_softmax[..., i]  # [32, 64]
            r_qk_softmax_i = r_qk_softmax_i.unsqueeze(1).unsqueeze(-1)  # [32, 1, 64, 1]
            r_qk_softmax_i = r_qk_softmax_i.repeat(1, head, 1, length)  # [32, 8, 64, 16]

            # update delays_agg by adding the product of rolled value and r_qk_softmax_i
            delays_agg = delays_agg + roll_v_tao * r_qk_softmax_i  # [32, 8, 64, 16] * 3

        return delays_agg

    def time_delay_agg_same_all(self, values, r_qk):
        """
        Time Delay Aggregation.
        SpeedUp version of AutoCorrelation (a batch-normalization style design).
        Same for the training phase or the inference phase.
        :param values: [32, 8, 64, 16]
        :param r_qk: [32, 8, 64, 16]
        :return:
        """

        # init size
        batch, head, channel, length = values.shape  # 32, 8, 64, 16

        # init index
        init_index = torch.arange(length)  # shape: [16] (from 0 to 15)
        init_index = init_index.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 1, 16]
        init_index = init_index.repeat(batch, head, channel, 1)  # shape: [32, 8, 64, 16]
        init_index = init_index.to(values.device)

        # calculate the number of top elements to select, based on the logarithm of the sequence length
        k = int(self.factor * math.log(length))  # 5

        # calculate the mean value of r_qk in every head
        mean_r_qk = torch.mean(r_qk, dim=1)  # [32, 64, 16]

        # calculate the mean value of mean_r_qk in every head dimensions
        mean_r_qk = torch.mean(mean_r_qk, dim=1)  # [32, 16]

        # select the largest elements from mean_r_qk along the time sequence dimension
        r_qk_top, tao = torch.topk(mean_r_qk, k, dim=-1, largest=True)  # [32, 5] (values), [32, 5] (indices)

        # apply the softmax function to r_qk_top along the time sequence dimension, converting them into probabilities
        r_qk_softmax = torch.softmax(r_qk_top, dim=-1)  # [32, 5]

        # duplicate values along the last dimension
        tmp_values = values.repeat(1, 1, 1, 2)  # [32, 8, 64, 32]

        # create a tensor filled with zeros, having the same shape as values
        delays_agg = torch.zeros_like(values).float()  # [32, 8, 64, 16]

        for i in range(k):
            # get the index of the i-th element
            tao_k = tao[:, i]  # [32]
            tao_k = tao_k.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [32, 1, 1, 1]
            tao_k = tao_k.repeat(1, head, channel, length)  # [32, 8, 64, 16]

            # calculate a temporary index by adding the i-th element of delay to init_index
            tmp_index = init_index + tao_k  # [32, 8, 64, 16] * 3

            # gather elements from tmp_values along the last dimension, according to tmp_index
            roll_v_tao = torch.gather(tmp_values, dim=-1, index=tmp_index)  # [32, 8, 64, 16]

            # get the i-th element of r_qk_softmax
            r_qk_softmax_i = r_qk_softmax[:, i]  # [32]
            r_qk_softmax_i = r_qk_softmax_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [32, 1, 1, 1]
            r_qk_softmax_i = r_qk_softmax_i.repeat(1, head, channel, length)  # [32, 8, 64, 16]

            # update delays_agg by adding the product of rolled value and r_qk_softmax_i
            delays_agg = delays_agg + roll_v_tao * r_qk_softmax_i  # [32, 8, 64, 16] * 3

        return delays_agg

    def time_delay_agg_speed(self, values, r_qk, train):
        """
        Time Delay Aggregation.
        SpeedUp version of AutoCorrelation (a batch-normalization style design).
        Different for the training phase or the inference phase.
        :param values: [32, 8, 64, 16]
        :param r_qk: [32, 8, 64, 16]
        :param train: train (True) or inference (False)
        :return:
        """

        # init size
        batch, head, channel, length = values.shape  # 32, 8, 64, 16

        # init index
        if not train:
            init_index = torch.arange(length)  # shape: [16] (from 0 to 15)
            init_index = init_index.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 1, 16]
            init_index = init_index.repeat(batch, head, channel, 1)  # shape: [32, 8, 64, 16]
            init_index = init_index.to(values.device)
        else:
            init_index = None

        # calculate the number of top elements to select, based on the logarithm of the sequence length
        k = int(self.factor * math.log(length))  # 5

        # calculate the mean value of r_qk in every head
        mean_r_qk = torch.mean(r_qk, dim=1)  # [32, 64, 16]

        # calculate the mean value of mean_r_qk in every head dimensions
        mean_r_qk = torch.mean(mean_r_qk, dim=1)  # [32, 16]

        # select the largest elements from mean_r_qk along the time sequence dimension
        if not train:
            r_qk_top, tao = torch.topk(mean_r_qk, k, dim=-1, largest=True)  # [32, 5] (values), [32, 5] (indices)
        else:
            tao = torch.topk(torch.mean(mean_r_qk, dim=0), k, dim=-1, largest=True)[1]  # [5]
            r_qk_top = torch.stack([mean_r_qk[:, tao[i]] for i in range(k)], dim=-1)  # [32, 5]

        # apply the softmax function to r_qk_top along the time sequence dimension, converting them into probabilities
        r_qk_softmax = torch.softmax(r_qk_top, dim=-1)  # [32, 5]

        if not train:
            # duplicate values along the last dimension
            tmp_values = values.repeat(1, 1, 1, 2)  # [32, 8, 64, 32]
        else:
            tmp_values = None

        # create a tensor filled with zeros, having the same shape as values
        delays_agg = torch.zeros_like(values).float()  # [32, 8, 64, 16]

        if not train:
            for i in range(k):
                # get the index of the i-th element
                tao_k = tao[:, i]  # [32]
                tao_k = tao_k.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [32, 1, 1, 1]
                tao_k = tao_k.repeat(1, head, channel, length)  # [32, 8, 64, 16]

                # calculate a temporary index by adding the i-th element of delay to init_index
                tmp_index = init_index + tao_k  # [32, 8, 64, 16] * 3

                # gather elements from tmp_values along the last dimension, according to tmp_index
                roll_v_tao = torch.gather(tmp_values, dim=-1, index=tmp_index)  # [32, 8, 64, 16]

                # get the i-th element of r_qk_softmax
                r_qk_softmax_i = r_qk_softmax[:, i]  # [32]
                r_qk_softmax_i = r_qk_softmax_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [32, 1, 1, 1]
                r_qk_softmax_i = r_qk_softmax_i.repeat(1, head, channel, length)  # [32, 8, 64, 16]

                # update delays_agg by adding the product of rolled value and r_qk_softmax_i
                delays_agg = delays_agg + roll_v_tao * r_qk_softmax_i  # [32, 8, 64, 16] * 3
        else:
            for i in range(k):
                # get the rolled value from the values
                roll_v_tao = torch.roll(values, shifts=-int(tao[i]), dims=-1)  # [32, 8, 64, 16]

                # get the i-th element of r_qk_softmax
                r_qk_softmax_i = r_qk_softmax[:, i]  # [32]
                r_qk_softmax_i = r_qk_softmax_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [32, 1, 1, 1]
                r_qk_softmax_i = r_qk_softmax_i.repeat(1, head, channel, length)  # [32, 8, 64, 16]

                # update delays_agg by adding the product of rolled value and r_qk_softmax_i
                delays_agg = delays_agg + roll_v_tao * r_qk_softmax_i  # [32, 8, 64, 16] * 3

        return delays_agg

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        32 is the batch size
        16 is the input sequence length
        8 is the number of heads
        64 is the dimension for each head
        :param queries: shape: [32, 16, 8, 64]
        :param keys: shape: [32, 16, 8, 64]
        :param values: shape: [32, 16, 8, 64]
        :param attn_mask:
        :param tau: param related to de-stationary attention
        :param delta: param related to de-stationary attention
        :return:
        """

        # B: batch size, 32
        # Q: query length, 16
        # K: keys length, 16
        # V: value length, 16
        # H: heads number, 8

        B, Q, H, _ = queries.shape
        _, K, _, _ = keys.shape
        _, V, _, _ = values.shape

        # if the length of the queries is larger than the values, we need to padding it with zeros
        # else, we only choose the length of the queries
        if Q > V:
            zeros = torch.zeros_like(queries[:, :(Q - V), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :Q, :, :]
            keys = keys[:, :Q, :, :]

        # put the sequence length to the last dimension for fft
        queries = queries.permute(0, 2, 3, 1).contiguous()  # shape: [32, 8, 64, 16]
        keys = keys.permute(0, 2, 3, 1).contiguous()  # shape: [32, 8, 64, 16]
        values = values.permute(0, 2, 3, 1).contiguous()  # shape: [32, 8, 64, 16]

        # apply sequence normalization to the queries, keys, values
        # CHANGE: we make sure the mean value of every sequence is zero, making zero mean nothing in time series data
        # and making the mean values for every head is the same (zero)
        if self.norm is True:
            queries = self.q_norm_layer(queries)  # shape: [32, 8, 64, 16]
            keys = self.k_norm_layer(keys)  # shape: [32, 8, 64, 16]
            values = self.v_norm_layer(values)  # shape: [32, 8, 64, 16]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries, dim=-1)  # shape: [32, 8, 64, 9]
        k_fft = torch.fft.rfft(keys, dim=-1)  # shape: [32, 8, 64, 9]
        res = q_fft * torch.conj(k_fft)  # shape: [32, 8, 64, 9]
        corr = torch.fft.irfft(res, n=Q, dim=-1)  # shape: [32, 8, 64, 16]

        # time delay agg
        # 'speed' is the fastest method, and each method after is slower than before.
        # CHANGE: We use the same top_k for every head.
        if self.agg_mode == 'speed':
            out = self.time_delay_agg_speed(values, corr, self.training)
        elif self.agg_mode == 'same_all':
            out = self.time_delay_agg_same_all(values, corr)
        elif self.agg_mode == 'same_head':
            out = self.time_delay_agg_same_head(values, corr)
        elif self.agg_mode == 'full':
            out = self.time_delay_agg_full(values, corr)
        else:
            raise NotImplementedError

        # restore the original dimension sequence
        out = out.permute(0, 3, 1, 2).contiguous()  # shape: [32, 16, 8, 64]

        if self.output_attention:
            return out, corr.permute(0, 3, 1, 2)
        else:
            return out, None


class AutoCorrelationLayer(AttentionLayer):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__(correlation, d_model, n_heads, d_keys, d_values)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        return super().forward(queries, keys, values, attn_mask, tau, delta)
