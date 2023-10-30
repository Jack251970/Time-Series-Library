import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, LayerNorm, series_decomp
from layers.Embed import DataEmbedding_no_pos


# noinspection DuplicatedCode
class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    Tips:
    The codes of the original paper use speed aggregation mode, which means agg_mode equals 'speed'.
    You can use 'same_all', 'same_head', 'full' for better performance but slower calculation speed.
    """

    def __init__(self, configs, agg_mode='speed'):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Series Decomposition
        kernel_size = configs.moving_avg
        self.series_decomp = series_decomp(kernel_size, series_decomp_mode=configs.series_decomp_mode)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_no_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_no_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout) \
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' else None

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention, agg_mode=agg_mode),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    _moving_avg=configs.moving_avg,
                    series_decomp_mode=configs.series_decomp_mode,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=LayerNorm(configs.d_model)
        )

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False, agg_mode=agg_mode),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False, agg_mode=agg_mode),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        _moving_avg=configs.moving_avg,
                        series_decomp_mode=configs.series_decomp_mode,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for _ in range(configs.d_layers)
                ],
                norm_layer=LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        32 is the batch size, 16 is the sequence length (time steps), and 14 is the feature dimension.
        :param x_enc: shape [32, 16, 14]
        :param x_mark_enc: [32, 16, 5]
        :param x_dec: [32, 32, 14]
        :param x_mark_dec: [32, 32, 5]
        :return:
        """
        # init padding sequence
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)  # mean in every feature
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)

        # init series decomposition
        seasonal_init, trend_init = self.series_decomp(x_enc)

        # decoder input
        # the second 32 is the sequence length (time steps) plus the prediction length.
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)  # shape: [32, 32, 14]
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)  # shape: [32, 32, 14]

        # enc: input the original data
        # 512 is the size of dmodel.
        enc_in = self.enc_embedding(x_enc, x_mark_enc)  # shape: [32, 16, 512]
        enc_out, attentions = self.encoder(enc_in, attn_mask=None)  # [32, 16, 512], Unknown

        # dec: input the data after decomposition
        dec_in = self.dec_embedding(seasonal_init, x_mark_dec)  # shape: [32, 32, 512]
        seasonal_part, trend_part = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # shape: [32, 32, 14], [32, 32, 14]

        # final
        dec_out = trend_part + seasonal_part  # shape: [32, 32, 14]

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attentions
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.output_attention:
                dec_out, attentions = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out, attentions
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
