# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    # get the number of the modes
    modes = min(modes, seq_len // 2)

    # generate the modes
    if mode_select_method == 'random':
        # random choose modes, e.g. [4, 3, 0, 2]
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
        index.sort()  # e.g. [0, 2, 3, 4]
    else:
        # e.g. [0, 1, 2]
        index = list(range(0, modes))

    return index


def complex_mul1d(order, x, w):
    """
    Complex multiplication
    """
    x_complex_flag = torch.is_complex(x)
    w_complex_flag = torch.is_complex(w)
    if x_complex_flag is False:
        x = torch.complex(x, torch.zeros_like(x).to(x.device))
    if w_complex_flag is False:
        w = torch.complex(w, torch.zeros_like(w).to(w.device))
    if x_complex_flag is True or w_complex_flag is True:
        return torch.complex(torch.einsum(order, x.real, w.real) - torch.einsum(order, x.imag, w.imag),
                             torch.einsum(order, x.real, w.imag) + torch.einsum(order, x.imag, w.real))
    else:
        return torch.einsum(order, x.real, w.real)


class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random', print_info=False):
        """
        1D Fourier block. It performs representation learning on frequency domain,
        it does FFT, linear transform, and Inverse FFT.
        """
        super(FourierBlock, self).__init__()

        # get modes on frequency domain
        self.index_q = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)

        # get the scaled factor and get the weights
        # nn.Parameter makes the Tensor can be trained
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q),
                                                dtype=torch.float))
        self.weights2 = nn.Parameter(torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q),
                                                dtype=torch.float))

        if print_info:
            print('fourier enhanced block used!')
            print('modes={}, index={}'.format(modes, self.index_q))

    def forward(self, q, k, v, mask):
        """
        We only process the q.
        """
        device = q.device

        B, L, H, E = q.shape  # [B, L, H, E]
        _q = q.permute(0, 2, 3, 1).contiguous()  # [B, H, E, L]

        # Compute Fourier coefficients
        Q = torch.fft.rfft(_q, dim=-1)

        # Perform Fourier neural operations in the selected index
        Y = torch.zeros(B, H, E, L // 2 + 1, device=device, dtype=torch.cfloat)

        # The original codes use Torch.cfloat.
        # for i, _index_q in enumerate(self.index_q):
        #     if _index_q >= Q.shape[3]:
        #         continue
        #     QW = complex_mul1d("bhi,hio->bho", Q[:, :, :, _index_q], self.weights1[:, :, :, i])
        #     Y[:, :, :, i] = QW
        #
        # # multiply the scale factor
        # Y = Y * self.scale

        # By wuhaixu2016
        complex_weights = torch.complex(self.weights1, self.weights2)
        for i, _index_q in enumerate(self.index_q):
            if i >= Y.shape[3] or _index_q >= Q.shape[3]:
                continue
            QW = complex_mul1d("bhi,hio->bho", Q[:, :, :, _index_q], complex_weights[:, :, :, i])
            Y[:, :, :, i] = QW

        # multiply the scale factor
        Y = Y * self.scale * self.scale

        # Return to time domain
        y = torch.fft.irfft(Y, n=_q.size(-1))  # pass the signal length n to make sure the outcome is correct

        # Restore to original dimension sequence
        # The original codes don't restore the original dimension sequence here, it does this operation in moving_avg!
        y = y.permute(0, 3, 1, 2).contiguous()  # [B, L, H, E]

        return y, None


# noinspection DuplicatedCode
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0, num_heads=8, use_weights=True, print_info=False):
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.
        """
        super(FourierCrossAttention, self).__init__()

        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        # get the scaled factor and get the weights
        # nn.Parameter makes the Tensor can be trained
        self.scale = 1 / (in_channels * out_channels)
        self.use_weights = use_weights
        if use_weights:
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads,
                                        len(self.index_q), dtype=torch.float))
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(num_heads, in_channels // num_heads, out_channels // num_heads,
                                        len(self.index_q), dtype=torch.float))

        if print_info:
            print('fourier enhanced cross attention used!')
            print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
            print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

    def forward(self, q, k, v, mask):
        """
        formula: Padding(activation(Selected_Q*Selected_K)*Selected_V)
        """
        device = q.device

        B, L, H, E = q.shape  # [B, L, H, E]
        _q = q.permute(0, 2, 3, 1).contiguous()  # [B, H, E, L]
        _k = k.permute(0, 2, 3, 1).contiguous()
        _v = v.permute(0, 2, 3, 1).contiguous()

        # get selected q
        Q = torch.fft.rfft(_q, dim=-1)
        Selected_Q = torch.zeros(B, H, E, len(self.index_q), device=device, dtype=torch.cfloat)
        for i, _index_q in enumerate(self.index_q):
            if _index_q >= Q.shape[3]:
                continue
            Selected_Q[:, :, :, i] = Q[:, :, :, _index_q]

        # get selected K
        K = torch.fft.rfft(_k, dim=-1)
        Selected_K = torch.zeros(B, H, E, len(self.index_kv), device=device, dtype=torch.cfloat)
        for i, _index_kv in enumerate(self.index_kv):
            if _index_kv >= K.shape[3]:
                continue
            Selected_K[:, :, :, i] = K[:, :, :, _index_kv]

        # get selected V
        V = torch.fft.rfft(_v, dim=-1)
        Selected_V = torch.zeros(B, H, E, len(self.index_kv), device=device, dtype=torch.cfloat)
        for i, _index_kv in enumerate(self.index_kv):
            if _index_kv >= V.shape[3]:
                continue
            Selected_V[:, :, :, i] = V[:, :, :, _index_kv]

        # perform attention mechanism on frequency domain
        Selected_QK = complex_mul1d("bhex,bhey->bhxy", Selected_Q, Selected_K)

        # perform activation on multiplied outcome
        if self.activation == 'tanh':
            Selected_QK = torch.complex(Selected_QK.real.tanh(), Selected_QK.imag.tanh())
        elif self.activation == 'softmax':
            Selected_QK = torch.softmax(abs(Selected_QK), dim=-1)
            Selected_QK = torch.complex(Selected_QK, torch.zeros_like(Selected_QK))
        else:
            raise Exception('{} activation function is not implemented'.format(self.activation))

        # continue to perform attention mechanism on frequency domain
        # The original codes replace the Selected_V with Selected_Q.
        # https://github.com/MAZiqing/FEDformer/issues/19
        # https://github.com/MAZiqing/FEDformer/issues/34
        QKV = complex_mul1d("bhxy,bhey->bhex", Selected_QK, Selected_V)

        if self.use_weights:
            # Perform Fourier neural operations in the selected index
            # get selected qkvw as output
            complex_weight = torch.complex(self.weights1, self.weights2)
            QKVW = complex_mul1d("bhex,heox->bhox", QKV, complex_weight)
            Y = torch.zeros(B, H, E, L // 2 + 1, device=device, dtype=torch.cfloat)
            for i, _index_q in enumerate(self.index_q):
                if i >= QKVW.shape[3] or _index_q > Y.shape[3]:
                    continue
                Y[:, :, :, _index_q] = QKVW[:, :, :, i]

            # original codes: the _index_q maybe larger than L//2 which leads to index-out-of-range error
            # QKVW = torch.einsum("bhex,heox->bhox", QKV, self.weights)
            # Y = torch.zeros(B, H, E, L // 2 + 1, device=device, dtype=torch.cfloat)
            # for i, _index_q in enumerate(self.index_q):
            #     Y[:, :, :, _index_q] = QKVW[:, :, :, i]
        else:
            Y = QKV

            # multiply the scale factor
        Y = Y * self.scale

        # Return to time domain
        y = torch.fft.irfft(Y, n=_q.size(-1))

        # Restore to original dimension sequence
        y = y.permute(0, 3, 1, 2).contiguous()  # [B, L, H, E]

        return y, None
