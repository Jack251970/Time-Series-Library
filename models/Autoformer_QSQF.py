import torch
import torch.nn as nn

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, LayerNorm, series_decomp
from layers.Embed import DataEmbedding_no_pos

from torch.nn.functional import pad


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

    def __init__(self, configs, agg_mode='speed', plan='C'):
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
                                                  configs.dropout)

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

        # QSQF
        self.num_spline = configs.num_spline
        self.sample_times = configs.sample_times
        self.plan = plan
        if self.plan == 'AB':
            # Plan AB:
            # beta_02k:[beta0,beta2,beta_kand1]
            self.beta_02k = nn.Linear(configs.c_out, 2 + 1)
            self.pre_sigma = nn.Linear(configs.c_out, self.num_spline)
            self.pre_gamma = nn.Linear(configs.c_out, self.num_spline)
            # softmax to make sure Σu equals to 1
            self.sigma = nn.Softmax(dim=1)
            # softplus to make sure gamma is positive
            self.gamma = nn.Softplus()
            # softplus to make sure beta2 and beta_kand1 not negative
            self.beta_2k = nn.ReLU()
        else:
            # QSQF - Plan C:
            self.pre_beta_0 = nn.Linear(configs.c_out, 1)
            self.pre_gamma = nn.Linear(configs.c_out, self.num_spline)
            self.beta_0 = nn.Softplus()
            self.gamma = nn.Softplus()

    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        # [256, 96, 14], [256, 96, 5], [256, 32, 14], [256, 32, 14], [256, 32, 5]
        if self.task_name == 'probability_forecast':
            if self.plan == 'AB':
                beta_0, beta_2k_neg, sigma, gamma = self.probability_forecast(x_enc, x_mark_enc, x_dec, y_enc,
                                                                              x_mark_dec, mask)
                return beta_0, beta_2k_neg, sigma, gamma  # [256, 3], [256, 31, 3], [256, 32, 20], [256, 32, 20]
            elif self.plan == 'C':
                beta_0, gamma = self.probability_forecast(x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask)
                return beta_0, gamma
        return None

    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, sample=True, probability_range=0.95):
        if self.task_name == 'probability_forecast':  # [256, 32, 1], [256, 32, 20]
            outputs = self.probability_forecast(x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask)

            if self.plan == 'AB':
                beta_0s, beta_2k_negs, sigmas, gammas = outputs
            elif self.plan == 'C':
                beta_0s, gammas = outputs

            device = x_enc.device
            batch_size = x_enc.shape[0]

            # prediction range
            cdf_high = 1 - (1 - probability_range) / 2
            cdf_low = (1 - probability_range) / 2

            samples_high = torch.zeros(1, batch_size, self.pred_len, device=device, requires_grad=False)  # [1, 256, 16]
            samples_low = torch.zeros(1, batch_size, self.pred_len, device=device, requires_grad=False)  # [1, 256, 16]
            samples = torch.zeros(self.sample_times, batch_size, self.pred_len, device=device,
                                  requires_grad=False)  # [99, 256, 12]

            for j in range(self.sample_times + 2):
                for t in range(self.pred_len):
                    if self.plan == 'AB':
                        beta_0 = beta_0s[:, t, :]  # [256, 1]
                        beta_2k_neg = beta_2k_negs[:, t, :]  # [256, 2]
                        sigma = sigmas[:, t, :]  # [256, 20]
                        gamma = gammas[:, t, :]  # [256, 20]
                    elif self.plan == 'C':
                        # Plan C:
                        beta_0 = beta_0s[:, t, :]  # [256, 1]
                        gamma = gammas[:, t, :]  # [256, 20]

                    # pred_cdf is a uniform distribution
                    if j == 0:  # high
                        pred_cdf = torch.Tensor([cdf_high]).to(device)
                    elif j == 1:  # low
                        pred_cdf = torch.Tensor([cdf_low]).to(device)
                    else:
                        uniform = torch.distributions.uniform.Uniform(
                            torch.tensor([0.0], device=device),
                            torch.tensor([1.0], device=device))
                        pred_cdf = uniform.sample([batch_size])  # [256, 1]

                    if self.plan == 'AB':
                        # Plan AB
                        beta_1 = gamma[:, 0] - 2 * beta_2k_neg[:, 0] * sigma[:, 0]
                        beta_N = pad(torch.unsqueeze(beta_1, dim=1), (1, 0))
                        beta_N[:, 0] = beta_0

                        beta = (gamma - pad(gamma, (1, 0))[:, :-1]) / (2 * sigma)
                        beta[:, 0] = beta_2k_neg[:, 0]
                        beta = beta - pad(beta, (1, 0))[:, :-1]
                        beta[:, -1] = beta_2k_neg[:, 1] - beta[:, :-1].sum(dim=1)

                        ksi = pad(torch.cumsum(sigma, dim=1), (1, 0))[:, :-1]
                        indices = ksi < pred_cdf
                        pred = (beta_N * pad(pred_cdf, (1, 0), value=1)).sum(dim=1)
                        pred = pred + ((pred_cdf - ksi).pow(2) * beta * indices).sum(dim=1)
                    elif self.plan == 'C':
                        # Plan C
                        sigma = torch.full_like(gamma, 1.0 / gamma.shape[1])  # [256, 20]
                        beta = pad(gamma, (1, 0))[:, :-1]
                        beta[:, 0] = beta_0[:, 0]
                        beta = (gamma - beta) / (2 * sigma)
                        beta = beta - pad(beta, (1, 0))[:, :-1]
                        beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)  # [256, 20]

                        ksi = pad(torch.cumsum(sigma, dim=1), (1, 0))[:, :-1]  # [256, 20]
                        indices = ksi < pred_cdf  # [256, 20]
                        pred = (beta_0 * pred_cdf).sum(dim=1)  # [256,]
                        pred = pred + ((pred_cdf - ksi).pow(2) * beta * indices).sum(dim=1)  # [256, 20] # Q(alpha)公式?

                    if j == 0:
                        samples_high[0, :, t] = pred
                    elif j == 1:
                        samples_low[0, :, t] = pred
                    else:
                        samples[j - 2, :, t] = pred

            sample_mu = torch.mean(samples, dim=0).unsqueeze(-1)  # mean or median ? # [256, 12, 1]
            sample_std = samples.std(dim=0).unsqueeze(-1)  # [256, 12, 1]

            return samples, sample_mu, sample_std, samples_high, samples_low
        return None

    def probability_forecast(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        """
        32 is the batch size, 16 is the sequence length (time steps), and 14 is the feature dimension.
        :param x_enc: shape [256, 96, 5]
        :param x_mark_enc: [256, 96, 5]
        :param x_dec: [256, 32, 14]
        :param y_enc: [256, 32, 14]
        :param x_mark_dec: [256, 32, 5]
        :param mask: mask
        :return:
        """
        # init padding sequence
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)  # [256, 16, 5]
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)  # [256, 16, 5]

        # init series decomposition
        seasonal_init, trend_init = self.series_decomp(x_enc)  # [256, 32, 5], [256, 32, 5]

        # decoder input
        # the second 32 is the sequence length (time steps) plus the prediction length.
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)  # [256, 32, 5]
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)  # [256, 32, 5]

        # enc: input the original data
        # 512 is the size of dmodel.
        enc_in = self.enc_embedding(x_enc, x_mark_enc)  # [256, 96, 512]
        enc_out, attentions = self.encoder(enc_in, attn_mask=None)  # [256, 96, 512], Unknown

        # dec: input the data after decomposition
        dec_in = self.dec_embedding(seasonal_init, x_mark_dec)  # [256, 32, 512]
        seasonal_part, trend_part = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # [256, 32, 5], [256, 32, 5]

        # final
        dec_out = trend_part + seasonal_part  # [32, 32, 5]

        if self.plan == 'AB':
            # Plan AB:
            beta_02k = self.beta_02k(dec_out)  # [256, 3]
            beta_0 = beta_02k[:, 0]  # [256]
            pre_sigma = self.pre_sigma(dec_out)  # [256, 20]
            sigma = self.sigma(pre_sigma)  # [256, 20]
            pre_gamma = self.pre_gamma(dec_out)  # [256, 20]
            gamma = self.gamma(pre_gamma)  # [256, 20]
            beta_2k = self.beta_2k(beta_02k[:, 1:])  # [256, 2]
            # to make sure beta_2 is negative
            beta_2k_neg = beta_2k.clone()  # clone this to avoid gradient error!!!
            beta_2k_neg[:, 0] = -beta_2k[:, 0]

            return beta_0, beta_2k_neg, sigma, gamma  # [256], [256, 2], [256, 20], [256, 20]
        elif self.plan == 'C':
            # Plan C:
            pre_beta_0 = self.pre_beta_0(dec_out)  # [256, 32, 1]
            beta_0 = self.beta_0(pre_beta_0)  # [256, 32, 1]
            pre_gamma = self.pre_gamma(dec_out)  # [256, 32, 20]
            gamma = self.gamma(pre_gamma)  # [256, 32, 20]

            return beta_0, gamma  # [256, 32, 1], [256, 32, 20]

        return dec_out
