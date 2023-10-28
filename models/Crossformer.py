from math import ceil

import torch
import torch.nn as nn
from einops import repeat

from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import DSWEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    Tips:
    The codes of the original paper do not have the inner positional embedding, which means pos_embed equals False.
    If you enable pos_embed, the performance will be improved.
    The codes of the original paper pad in the start of a sequence, which means padding_start equals True.
    """

    def __init__(self, configs, seg_len=12, win_size=2, padding_start=True, pos_embed=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = seg_len
        self.win_size = win_size
        self.padding_start = padding_start

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.in_len_add = self.pad_in_len - configs.seq_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = DSWEmbedding(self.seg_len, configs.d_model, pos_embed=pos_embed)
        # the learnable position embedding for position
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        # the normalization layer for embedding
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if _ is 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if _ is 0 else ceil(self.in_seg_num / self.win_size ** _), configs.factor
                            ) for _ in range(configs.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model,
                                           configs.n_heads, configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    # activation=configs.activation,
                )
                for _ in range(configs.e_layers + 1)
            ],
        )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def _enc_embedding(self, x_enc):
        # padding for input sequence
        if self.in_len_add != 0:
            if self.padding_start:
                # padding for input sequence on the left side
                # This is the codes of the original paper.
                x_padding = torch.cat((x_enc, x_enc[:, 1:, :].expand(-1, self.padding, -1)), dim=1)
            else:
                # padding for input sequence on the right side
                # This is the implementation of the codes of PatchEmbedding in PatchTSE model.
                x_padding = torch.cat((x_enc[:, :1, :].expand(-1, self.padding, -1), x_enc), dim=1)
        else:
            x_padding = x_enc

        # apply value embedding and position embedding
        e_s = self.enc_value_embedding(x_padding)
        e_pos = self.enc_pos_embedding
        h = e_s + e_pos
        h = self.pre_norm(h)
        return h

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc = self._enc_embedding(x_enc)
        # encoder
        enc_out, attns = self.encoder(x_enc)
        # decoder
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        # decoder
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # embedding
        x_enc = self._enc_embedding(x_enc)
        # encoder
        enc_out, attns = self.encoder(x_enc)
        # decoder
        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def anomaly_detection(self, x_enc):
        # embedding
        x_enc = self._enc_embedding(x_enc)
        # encoder
        enc_out, attns = self.encoder(x_enc)
        # decoder
        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        x_enc = self._enc_embedding(x_enc)
        # encoder
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
