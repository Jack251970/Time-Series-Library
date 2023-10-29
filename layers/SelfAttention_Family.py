import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class FullAttention(nn.Module):
    """
    The Attention operation
    Paper: Attention Is All You Need, Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 de_stationary=False):
        """
        mask_flag: whether to use mask on the sequence data
        de_stationary: whether to add de-stationary attention mechanism
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.de_stationary = de_stationary

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
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        if self.de_stationary:
            tau = 1.0 if tau is None else tau.unsqueeze(
                1).unsqueeze(1)  # B x 1 x 1 x 1
            delta = 0.0 if delta is None else delta.unsqueeze(
                1).unsqueeze(1)  # B x 1 x 1 x S

            # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
            scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta
        else:
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [32, 8, 16, 16]

        if self.mask_flag:
            if attn_mask is None:
                # generate a triangular mask to make sure the future data has been masked
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [32, 8, 16, 16]
        V = torch.einsum("bhls,bshd->blhd", A, values)  # [32, 16, 8, 64]

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """
    The Multi-head Self-Attention (MSA) Layer
    Paper: Attention Is All You Need
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, de_stationary=False):
        """

        :param attention:
        :param d_model: the embed size which is the input dimension size
        :param n_heads: the head number of the multi-head attention
        :param d_keys:
        :param d_values:
        """
        super(AttentionLayer, self).__init__()
        self.inner_attention = attention
        self.n_heads = n_heads

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # we use linear layers to get q, k, v
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # we use linear layers to get output
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.de_stationary = de_stationary

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        32 is the batch size
        16 is the input sequence length
        512 is the d_model (from feature size)
        :param queries: shape: [32, 16, 512]
        :param keys: shape: [32, 16, 512]
        :param values: shape: [32, 16, 512]
        :param attn_mask:
        :param tau: param related to de-stationary attention
        :param delta: param related to de-stationary attention
        :return:
        """

        B, Q, _ = queries.shape
        _, K, _ = keys.shape
        _, V, _ = values.shape
        H = self.n_heads

        # B: batch size, 32
        # Q: query length, 16
        # K: keys length, 16
        # V: value length, 16
        # H: heads number, 8

        # apply the q, k, v projections
        queries = self.query_projection(queries)  # [32, 16, 512]
        keys = self.key_projection(keys)  # [32, 16, 512]
        values = self.value_projection(values)  # [32, 16, 512]

        # split into pieces for all heads
        queries = queries.view(B, Q, H, -1)  # [32, 16, 8, 64]
        keys = keys.view(B, K, H, -1)  # [32, 16, 8, 64]
        values = values.view(B, V, H, -1)  # [32, 16, 8, 64]

        if self.de_stationary:
            out, attn = self.inner_attention(
                queries,
                keys,
                values,
                attn_mask,
                tau=tau,
                delta=delta
            )  # [32, 16, 8, 64], unknown
        else:
            out, attn = self.inner_attention(
                queries,
                keys,
                values,
                attn_mask,
            )  # [32, 16, 8, 64], unknown

        # flat the last two dimensions
        out = out.view(B, Q, -1)  # [32, 16, 512]

        # apply the out projection
        out = self.out_projection(out)  # [32, 16, 512]

        return out, attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: default queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    Paper: CROSSFORMER: TRANSFORMER UTILIZING CROSS-DIMENSION DEPENDENCY FOR MULTIVARIATE TIME SERIES FORECASTING
    """

    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router_param = nn.Parameter(torch.randn(seg_num, factor, d_model))  # [2, 2, 512]

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # a multi-layer (two in this paper) feedforward network
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):  # [32, 14, 2, 512]
        batch = x.shape[0]

        # Cross Time Stage
        # Directly apply MSA to each dimension
        time_in = rearrange(x, 'b d seg_num d_model -> (b d) seg_num d_model')  # [448, 2, 512]
        # apply MSA to each dimension, it will encode the information in the second dimension (seg_num)
        time_enc, attn = self.time_attention(time_in, time_in, time_in, attn_mask=None)  # [448, 2, 512], unknown
        # drop out and apply layer norm and MLP
        time_out = time_in + self.dropout(time_enc)
        time_out = self.norm1(time_out)
        time_out = self.MLP1(time_out)
        time_out = time_out + self.dropout(time_out)
        time_out = self.norm2(time_out)  # [448, 2, 512]

        # Cross Dimension Stage
        # use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_in = time_out
        dim_send = rearrange(dim_in, '(b d) seg_num d_model -> (b seg_num) d d_model', b=batch)  # [64, 14, 512]
        batch_router = repeat(self.router_param, 'seg_num factor d_model -> (repeat seg_num) factor d_model',
                              repeat=batch)  # [64, 2, 512], here router use factor as dimension to cut down costs
        # aggregate information from the dimensional input data by router
        dim_agg, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None)  # [64, 2, 512]
        # query dimensional information from the aggregated information and the dimensional input data
        dim_receive, attn = self.dim_receiver(dim_send, dim_agg, dim_agg, attn_mask=None)  # [64, 14, 512]
        # drop out and apply layer norm and MLP
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = self.MLP2(dim_enc)
        dim_enc = dim_enc + self.dropout(dim_enc)
        dim_enc = self.norm4(dim_enc)  # [64, 14, 512]

        final_out = rearrange(dim_enc, '(b seg_num) d d_model -> b d seg_num d_model', b=batch)

        return final_out  # [32, 14, 2, 512]
