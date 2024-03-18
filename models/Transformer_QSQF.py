import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from torch.nn.functional import pad


class Model(nn.Module):
    """
    Vanilla Transformer with QSQF (In progress)
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # QSQF - Plan C:
        self.num_spline = 20

        self.pre_beta_0 = nn.Linear(configs.c_out, 1)
        self.pre_gamma = nn.Linear(configs.c_out, self.num_spline)
        self.beta_0 = nn.Softplus()
        self.gamma = nn.Softplus()

        self.sample_times = 99

    def forward(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None):
        # [256, 96, 14], [256, 96, 5], [256, 32, 14], [256, 32, 14], [256, 32, 5]
        if self.task_name == 'probability_forecast':
            beta_0, gamma = self.probability_forecast(x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask)
            return beta_0, gamma
        return None

    def predict(self, x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask=None, probability_range=0.95):
        if self.task_name == 'probability_forecast':  # [256, 32, 1], [256, 32, 20]
            beta_0s, gammas = self.probability_forecast(x_enc, x_mark_enc, x_dec, y_enc, x_mark_dec, mask)  # [256, 32, 1], [256, 32, 20]

            device = x_enc.device
            batch_size = x_enc.shape[0]

            # prediction range
            cdf_high = 1 - (1 - probability_range) / 2
            cdf_low = (1 - probability_range) / 2

            samples_high = torch.zeros(1, batch_size, self.pred_steps, device=device, requires_grad=False)  # [1, 256, 16]
            samples_low = torch.zeros(1, batch_size, self.pred_steps, device=device, requires_grad=False)  # [1, 256, 16]
            samples = torch.zeros(self.sample_times, batch_size, self.pred_steps, device=device, requires_grad=False)  # [99, 256, 12]

            for j in range(self.sample_times + 2):
                for t in range(self.pred_steps):
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
        # [256, 96, 14], [256, 96, 5], [256, 32, 14], [256, 32, 14], [256, 32, 5]
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [256, 96, 512]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # [256, 96, 512], unknown

        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [256, 32, 512]
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)  # [256, 32, 14]

        # Plan C:
        pre_beta_0 = self.pre_beta_0(dec_out)  # [256, 32, 1]
        beta_0 = self.beta_0(pre_beta_0)  # [256, 32, 1]
        pre_gamma = self.pre_gamma(dec_out)  # [256, 32, 20]
        gamma = self.gamma(pre_gamma)  # [256, 32, 20]

        return beta_0, gamma  # [256, 32, 1], [256, 32, 20]


# noinspection DuplicatedCode
def loss_fn(outputs, labels):  # [256, 16, 1], [256, 16, 20], [256, 16, 1]
    loss = torch.zeros(1, device=labels.device, requires_grad=True)  # [,]

    beta_0, gamma = outputs  # [256, 16, 1], [256, 16, 20]

    # get pred_len and reshape
    pred_len = labels.shape[1]
    beta_0 = beta_0.reshape(-1, beta_0.shape[-1])  # [4096, 1]
    gamma = gamma.reshape(-1, gamma.shape[-1])  # [4096, 20]
    labels = labels.reshape(-1, labels.shape[-1])  # [4096, 1]

    crps = get_crps(beta_0, gamma, labels)

    crps = crps * pred_len

    loss = loss + crps

    return loss


def loss_fn_multi_steps(outputs, labels, steps=-1):  # [256, 16, 1], [256, 16, 20], [256, 16, 1]
    loss = torch.zeros(1, device=labels.device, requires_grad=True)  # [,]

    beta_0s, gammas = outputs  # [256, 1], [256, 20], [256, 1]

    steps = labels.shape[1] if steps == -1 else steps

    for step in range(steps):
        crps = get_crps(beta_0s[:, step, :], gammas[:, step, :], labels[:, step, :])

        loss = loss + crps

    return loss


def get_crps(beta_0, gamma, label):
    sigma = torch.full_like(gamma, 1.0 / gamma.shape[1], requires_grad=False)  # [256, 1], [256, 20]

    beta = pad(gamma, (1, 0))[:, :-1]  # [256, 20]
    beta[:, 0] = beta_0[:, 0]
    beta = (gamma - beta) / (2 * sigma)
    beta = beta - pad(beta, (1, 0))[:, :-1]
    beta[:, -1] = gamma[:, -1] - beta[:, :-1].sum(dim=1)

    # calculate the maximum for each segment of the spline
    ksi = torch.cumsum(sigma, dim=1)
    df1 = ksi.expand(sigma.shape[1], sigma.shape[0], sigma.shape[1]).T.clone()
    df2 = ksi.T.unsqueeze(2)
    ksi = pad(ksi, (1, 0))[:, :-1]
    knots = df1 - ksi
    knots[knots < 0] = 0
    knots = (df2 * beta_0).sum(dim=2) + (knots.pow(2) * beta).sum(dim=2)
    knots = pad(knots.T, (1, 0))[:, :-1]  # F(ksi_1~K)=0~max

    diff = label - knots
    alpha_l = diff > 0
    alpha_A = torch.sum(alpha_l * beta, dim=1)
    alpha_B = beta_0[:, 0] - 2 * torch.sum(alpha_l * beta * ksi, dim=1)
    alpha_C = -label.squeeze() + torch.sum(alpha_l * beta * ksi * ksi, dim=1)

    # since A may be zero, roots can be from different methods.
    not_zero = (alpha_A != 0)
    alpha = torch.zeros_like(alpha_A)
    # since there may be numerical calculation error,#0
    idx = (alpha_B ** 2 - 4 * alpha_A * alpha_C) < 0  # 0
    diff = diff.abs()
    index = diff == (diff.min(dim=1)[0].view(-1, 1))
    index[~idx, :] = False
    # index=diff.abs()<1e-4#0,1e-4 is a threshold
    # idx=index.sum(dim=1)>0#0
    alpha[idx] = ksi[index]  # 0
    alpha[~not_zero] = -alpha_C[~not_zero] / alpha_B[~not_zero]
    not_zero = ~(~not_zero | idx)  # 0
    delta = alpha_B[not_zero].pow(2) - 4 * alpha_A[not_zero] * alpha_C[not_zero]
    alpha[not_zero] = (-alpha_B[not_zero] + torch.sqrt(delta)) / (2 * alpha_A[not_zero])

    # formula for CRPS is here!
    crps_1 = label * (2 * alpha - 1)
    crps_2 = beta_0[:, 0] * (1 / 3 - alpha.pow(2))
    crps_3 = torch.sum(beta / 6 * (1 - ksi).pow(4), dim=1)
    crps_4 = torch.sum(alpha_l * 2 / 3 * beta * (alpha.unsqueeze(1) - ksi).pow(3), dim=1)
    crps = crps_1 + crps_2 + crps_3 - crps_4

    crps = torch.mean(crps)

    return crps
