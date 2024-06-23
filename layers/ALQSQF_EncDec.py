import torch
import torch.nn as nn
import torch.nn.functional as F


class adapt_smoothing(nn.Module):
    """
    Adaptive smoothing block to highlight the trend of time series
    """

    def __init__(self, kernel_size, conv=False):
        super(adapt_smoothing, self).__init__()
        self.kernel_size = kernel_size  # 3

        if not conv:
            self.template = nn.Linear(in_features=kernel_size, out_features=1, bias=False)
        else:
            self.template = nn.Conv1d(in_channels=kernel_size, out_channels=1, kernel_size=1, bias=False)

        # init the weights of the template
        self.template.weight.data.fill_(0.0)

    # noinspection DuplicatedCode
    def forward(self, x):  # [256, 96, 40]
        batch, sequence, feature = x.shape  # 256, 96, 40
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1, 1)  # [256, 4, 40]
        x = torch.cat([front, x], dim=1)  # [256, 100, 40]
        x = x.permute(0, 2, 1)  # [256, 40, 100]
        x = x.unsqueeze(-1)  # [256, 40, 100, 1]
        x = F.unfold(x, (sequence, 1), stride=1)  # [256, 3840, 5]
        x = self.template(x)  # [256, 3840, 1]
        x = x.view(batch, feature, sequence)  # [256, 40, 96]
        x = x.permute(0, 2, 1)  # [256, 96, 40]
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block for AL-QSQF
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.smoothing = adapt_smoothing(kernel_size)
        self.threshold = 1e-3  # use threshold to make sure all data is not on the left point

    def forward(self, x):  # [256, 96, 40]
        trend = self.smoothing(x)  # [256, 96, 40]
        uncertainty = x - trend  # [256, 96, 40]
        return uncertainty + self.threshold, trend - self.threshold
