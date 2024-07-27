import torch
import torch.nn as nn

from torch.nn.functional import pad


def phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type):
    """
    Formula
    2:
    beta_k = (eta_k - gamma) / (2 * alpha_prime_k), k = 1
    let x_k = (eta_k - eta_{k-1}) / (2 * alpha_prime_k), and x_k = 0, eta_0 = gamma
    beta_k = x_k - x_{k-1}, k > 1

    1+2:
    beta_k = (eta_k - gamma_1) / (2 * alpha_prime_k), k = 1
    let x_k = (eta_k - eta_{k-1} - gamma_k) / (2 * alpha_prime_k), and x_k = 0, eta_0 = 0
    beta_k = x_k - x_{k-1}, k > 1

    1:
    none
    """
    # get alpha_k ([0, k])
    alpha_0_k = pad(torch.cumsum(alpha_prime_k, dim=1), pad=(1, 0))[:, :-1]  # [256, 20]

    # get x_k
    if algorithm_type == '2':
        eta_0_1k = pad(eta_k, pad=(1, 0))[:, :-1]  # [256, 20]
        eta_0_1k[:, 0] = gamma[:, 0]  # eta_0 = gamma
        x_k = (eta_k - eta_0_1k) / (2 * alpha_prime_k)
    elif algorithm_type == '1+2':
        eta_0_1k = pad(eta_k, pad=(1, 0))[:, :-1]  # [256, 20]
        x_k = (eta_k - eta_0_1k - gamma) / (2 * alpha_prime_k)
    elif algorithm_type == '1':
        return alpha_0_k, None
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")

    # get beta_k
    beta_k = x_k - pad(x_k, pad=(1, 0))[:, :-1]  # [256, 20]

    return alpha_0_k, beta_k


# noinspection DuplicatedCode
def _get_y_hat(alpha_0_k, _lambda, gamma, beta_k, algorithm_type):
    """
    Formula
    2:
    int{Q(alpha)} = lambda * (max_alpha - min_alpha) + 1/2 * gamma * (max_alpha ^ 2 - min_alpha ^ 2)
    + sum(1/3 * beta_k * (max_alpha - alpha_0_k) ^ 3)
    y_hat = int{Q(alpha)} / (max_alpha - min_alpha)

    1+2:
    int{Q(alpha)} = lambda * (max_alpha - min_alpha) + sum(1/2 * gamma_k * (max_alpha - alpha_0_k) ^ 2)
    + sum(1/3 * beta_k * (max_alpha - alpha_0_k) ^ 3)
    y_hat = int{Q(alpha)} / (max_alpha - min_alpha)

    1:
    int {Q(alpha)} = lambda * (max_alpha - min_alpha) + sum(1/2 * gamma_k * (max_alpha - alpha_0_k) ^ 2)
    y_hat = int{Q(alpha)} / (max_alpha - min_alpha)
    """
    # init min_alpha and max_alpha
    device = gamma.device
    min_alpha = torch.Tensor([0]).to(device)  # [1]
    max_alpha = torch.Tensor([1]).to(device)  # [1]

    # get min pred and max pred
    # indices = alpha_0_k < min_alpha  # [256, 20]
    # min_pred0 = _lambda
    # if algorithm_type == '2':
    #     min_pred1 = (min_alpha * gamma).sum(dim=1)  # [256,]
    #     min_pred2 = ((min_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    #     min_pred = min_pred0 + min_pred1 + min_pred2  # [256,]
    # elif algorithm_type == '1+2':
    #     min_pred1 = ((min_alpha - alpha_0_k) * gamma * indices).sum(dim=1)  # [256,]
    #     min_pred2 = ((min_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    #     min_pred = min_pred0 + min_pred1 + min_pred2  # [256,]
    # elif algorithm_type == '1':
    #     min_pred1 = ((min_alpha - alpha_0_k) * gamma * indices).sum(dim=1)  # [256,]
    #     min_pred = min_pred0 + min_pred1  # [256,]
    # else:
    #     raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    # indices = alpha_0_k < max_alpha  # [256, 20]
    # max_pred0 = _lambda
    # if algorithm_type == '2':
    #     max_pred1 = (max_alpha * gamma).sum(dim=1)  # [256,]
    #     max_pred2 = ((max_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    #     max_pred = max_pred0 + max_pred1 + max_pred2  # [256,]
    # elif algorithm_type == '1+2':
    #     max_pred1 = ((max_alpha - alpha_0_k) * gamma * indices).sum(dim=1)  # [256,]
    #     max_pred2 = ((max_alpha - alpha_0_k).pow(2) * beta_k * indices).sum(dim=1)  # [256,]
    #     max_pred = max_pred0 + max_pred1 + max_pred2  # [256,]
    # elif algorithm_type == '1':
    #     max_pred1 = ((max_alpha - alpha_0_k) * gamma * indices).sum(dim=1)  # [256,]
    #     max_pred = max_pred0 + max_pred1  # [256,]
    # else:
    #     raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    # total_area = ((max_alpha - min_alpha) * (max_pred - min_pred))  # [256,]

    # get int{Q(alpha)}
    if algorithm_type == '2':
        integral0 = _lambda * (max_alpha - min_alpha)
        integral1 = 1 / 2 * gamma.squeeze() * (max_alpha.pow(2) - min_alpha.pow(2))  # [256,]
        integral2 = 1 / 3 * ((max_alpha - alpha_0_k).pow(3) * beta_k).sum(dim=1)  # [256,]
        integral = integral0 + integral1 + integral2  # [256,]
    elif algorithm_type == '1+2':
        integral0 = _lambda * (max_alpha - min_alpha)
        integral1 = 1 / 2 * ((max_alpha - alpha_0_k).pow(2) * gamma).sum(dim=1)  # [256,]
        integral2 = 1 / 3 * ((max_alpha - alpha_0_k).pow(3) * beta_k).sum(dim=1)  # [256,]
        integral = integral0 + integral1 + integral2  # [256,]
    elif algorithm_type == '1':
        integral0 = _lambda * (max_alpha - min_alpha)
        integral1 = 1 / 2 * ((max_alpha - alpha_0_k).pow(2) * gamma).sum(dim=1)  # [256,]
        integral = integral0 + integral1  # [256,]
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    y_hat = integral / (max_alpha - min_alpha)  # [256,]

    return y_hat


# noinspection DuplicatedCode
def sample_pred(alpha_prime_k, alpha, _lambda, gamma, eta_k, algorithm_type):
    """
    Formula
    2:
    Q(alpha) = lambda + gamma * alpha + sum(beta_k * (alpha - alpha_k) ^ 2)

    1+2:
    Q(alpha) = lambda + sum(beta_k * (alpha - alpha_k)) + sum(beta_k * (alpha - alpha_k) ^ 2)

    1:
    Q(alpha) = lambda + sum(beta_k * (alpha - alpha_k))
    """
    # phase parameter
    alpha_0_k, beta_k = phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type)

    if alpha is not None:
        # get Q(alpha)
        indices = alpha_0_k < alpha  # [256, 20]
        pred0 = _lambda
        if algorithm_type == '2':
            pred1 = (gamma * alpha).sum(dim=1)  # [256,]
            pred2 = (beta_k * (alpha - alpha_0_k).pow(2) * indices).sum(dim=1)  # [256,]
            pred = pred0 + pred1 + pred2  # [256,]
        elif algorithm_type == '1+2':
            pred1 = (gamma * (alpha - alpha_0_k) * indices).sum(dim=1)  # [256,]
            pred2 = (beta_k * (alpha - alpha_0_k).pow(2) * indices).sum(dim=1)  # [256,]
            pred = pred0 + pred1 + pred2  # [256,]
        elif algorithm_type == '1':
            pred1 = (gamma * (alpha - alpha_0_k) * indices).sum(dim=1)  # [256,]
            pred = pred0 + pred1  # [256,]
        else:
            raise ValueError("algorithm_type must be '1', '2', or '1+2'")

        return pred
    else:
        # get medium estimated value
        y_hat = sample_pred(alpha_prime_k, 0.5, _lambda, gamma, eta_k, algorithm_type)

        # get mean estimated value
        # y_hat = _get_y_hat(alpha_0_k, _lambda, gamma, beta_k, algorithm_type)

        return y_hat


def loss_fn_crps(tuple_param):
    alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type = tuple_param

    # labels
    labels = labels.unsqueeze(1)  # [256, 1]

    # calculate loss
    crpsLoss = get_crps(alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type)

    return crpsLoss


# noinspection DuplicatedCode
def get_crps(alpha_prime_k, _lambda, gamma, eta_k, y, algorithm_type):
    # [256, 1], [256, 20], [256, 20], [256, 20], [256, 1]
    alpha_0_k, beta_k = phase_gamma_and_eta_k(alpha_prime_k, gamma, eta_k, algorithm_type)
    alpha_1_k1 = pad(alpha_0_k, pad=(0, 1), value=1)[:, 1:]  # [256, 20]

    # calculate the maximum for each segment of the spline and get l
    df1 = alpha_1_k1.expand(alpha_prime_k.shape[1], alpha_prime_k.shape[0],
                            alpha_prime_k.shape[1]).T.clone()  # [20, 256, 20]
    knots = df1 - alpha_0_k  # [20, 256, 20]
    knots[knots < 0] = 0  # [20, 256, 20]
    if algorithm_type == '2':
        df2 = alpha_1_k1.T.unsqueeze(2)
        knots = _lambda + (df2 * gamma).sum(dim=2) + (knots.pow(2) * beta_k).sum(dim=2)  # [20, 256]
    elif algorithm_type == '1+2':
        knots = _lambda + (knots * gamma).sum(dim=2) + (knots.pow(2) * beta_k).sum(dim=2)  # [20, 256]
    elif algorithm_type == '1':
        knots = _lambda + (knots * gamma).sum(dim=2)  # [20, 256]
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")
    knots = pad(knots.T, (1, 0), value=_lambda)[:, :-1]  # F(alpha_{1~K})=0~max  # [256, 20]
    diff = y - knots  # [256, 20]
    alpha_l = diff > 0  # [256, 20]

    # calculate the parameter for quadratic equation
    y = y.squeeze()  # [256,]
    if algorithm_type == '2':
        A = torch.sum(alpha_l * beta_k, dim=1)  # [256,]
        B = gamma[:, 0] - 2 * torch.sum(alpha_l * beta_k * alpha_0_k, dim=1)  # [256,]
        C = _lambda - y + torch.sum(alpha_l * beta_k * alpha_0_k * alpha_0_k, dim=1)  # [256,]
    elif algorithm_type == '1+2':
        A = torch.sum(alpha_l * beta_k, dim=1)  # [256,]
        B = torch.sum(alpha_l * gamma, dim=1) - 2 * torch.sum(alpha_l * beta_k * alpha_0_k, dim=1)  # [256,]
        C = _lambda - y - torch.sum(alpha_l * gamma * alpha_0_k, dim=1) + torch.sum(
            alpha_l * beta_k * alpha_0_k * alpha_0_k, dim=1)  # [256,]
    elif algorithm_type == '1':
        A = torch.zeros_like(y)  # [256,]
        B = torch.sum(alpha_l * gamma, dim=1)
        C = _lambda - y - torch.sum(alpha_l * gamma * alpha_0_k, dim=1)
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")

    # solve the quadratic equation: since A may be zero, roots can be from different methods.
    not_zero = (A != 0)  # [256,]
    alpha_plus = torch.zeros_like(A)  # [256,]
    # since there may be numerical calculation error  #0
    idx = (B ** 2 - 4 * A * C) < 0  # 0  # [256,]
    diff = diff.abs()  # [256,]
    index = diff == (diff.min(dim=1)[0].view(-1, 1))  # [256,]
    index[~idx, :] = False  # [256,]
    # index=diff.abs()<1e-4  # 0,1e-4 is a threshold
    # idx=index.sum(dim=1)>0  # 0
    alpha_plus[idx] = alpha_0_k[index]  # 0  # [256,]
    alpha_plus[~not_zero] = -C[~not_zero] / B[~not_zero]  # [256,]
    not_zero = ~(~not_zero | idx)  # 0  # [256,]
    delta = B[not_zero].pow(2) - 4 * A[not_zero] * C[not_zero]  # [232,]
    alpha_plus[not_zero] = (-B[not_zero] + torch.sqrt(delta)) / (2 * A[not_zero])  # [256,]

    # get CRPS
    crps_1 = (_lambda - y) * (1 - 2 * alpha_plus)  # [256,]
    if algorithm_type == '2':
        crps_2 = gamma[:, 0] * (1 / 3 - alpha_plus.pow(2))
        crps_3 = torch.sum(1 / 6 * beta_k * (1 - alpha_0_k).pow(4), dim=1)
        crps_4 = torch.sum(2 / 3 * alpha_l * beta_k * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(3), dim=1)
        crps = crps_1 + crps_2 + crps_3 - crps_4
    elif algorithm_type == '1+2':
        crps_2 = torch.sum(1 / 3 * gamma * (1 - alpha_0_k).pow(3), dim=1)  # [256,]
        crps_3 = torch.sum(alpha_l * gamma * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(2), dim=1)  # [256,]
        crps_4 = torch.sum(1 / 6 * beta_k * (1 - alpha_0_k).pow(4), dim=1)  # [256,]
        crps_5 = torch.sum(2 / 3 * alpha_l * beta_k * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(3), dim=1)  # [256,]
        crps = crps_1 + crps_2 - crps_3 + crps_4 - crps_5  # [256, 256]
    elif algorithm_type == '1':
        crps_2 = torch.sum(1 / 3 * gamma * (1 - alpha_0_k).pow(3), dim=1)  # [256,]
        crps_3 = torch.sum(alpha_l * gamma * (alpha_plus.unsqueeze(1) - alpha_0_k).pow(2), dim=1)  # [256,]
        crps = crps_1 + crps_2 - crps_3  # [256,]
    else:
        raise ValueError("algorithm_type must be '1', '2', or '1+2'")

    crps = torch.mean(crps)  # [256,]
    return crps


def loss_fn_mse(tuple_param):
    alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type = tuple_param

    # get y_hat
    y_hat = sample_pred(alpha_prime_k, None, _lambda, gamma, eta_k, algorithm_type)  # [256,]

    # calculate loss
    loss = nn.MSELoss()
    mseLoss = loss(y_hat, labels)

    return mseLoss


def loss_fn_mae(tuple_param):
    alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type = tuple_param

    # get y_hat
    y_hat = sample_pred(alpha_prime_k, None, _lambda, gamma, eta_k, algorithm_type)  # [256,]

    # calculate loss
    loss = nn.L1Loss()
    mseLoss = loss(y_hat, labels)

    return mseLoss


_quantiles_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def loss_fn_quantiles(tuple_param):
    alpha_prime_k, _lambda, gamma, eta_k, labels, algorithm_type = tuple_param

    # labels
    labels = labels.unsqueeze(1)  # [256, 1]

    # sample quantiles
    global _quantiles_list
    quantiles_number = len(_quantiles_list)
    batch_size = labels.shape[0]
    device = labels.device
    quantiles = torch.Tensor(_quantiles_list).unsqueeze(0).expand(batch_size, -1).to(device)  # [256, 9]
    quantiles_y_pred = torch.zeros(batch_size, quantiles_number, device=device)  # [256, 9]
    for i in range(quantiles_number):
        quantile = torch.Tensor([_quantiles_list[i]]).unsqueeze(0).expand(batch_size, -1).to(device)  # [256, 1]
        samples = sample_pred(alpha_prime_k, quantile, _lambda, gamma, eta_k, algorithm_type)
        quantiles_y_pred[:, i] = samples  # [256,]

    # calculate loss
    residual = quantiles_y_pred - labels  # [256, 9]
    quantilesLoss = torch.max((quantiles - 1) * residual, quantiles * residual).mean()

    return quantilesLoss
