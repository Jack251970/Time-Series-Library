import torch
import properscoring as ps

def init_metrics(pred_len, device):
    metrics = {'num': torch.zeros(1, device=device), 'CRPS': torch.zeros(pred_len, device=device),
               'mre': torch.zeros((19, pred_len), device=device), 'naps': torch.zeros(pred_len, device=device),
               'picp': torch.zeros(pred_len, device=device), 'pinaw': torch.zeros(pred_len, device=device),
               'cwc': torch.zeros(pred_len, device=device)}
    for i in range(pred_len):
        metrics[f'CRPS_{i}'] = torch.zeros(1, device=device)
    for i in range(pred_len):
        metrics[f'naps_{i}'] = torch.zeros(1, device=device)
    for i in range(pred_len):
        metrics[f'picp_{i}'] = torch.zeros(1, device=device)
    for i in range(1, 10, 1):
        alpha = i / 20
        metrics[f'picp_alpha{alpha}'] = torch.zeros(pred_len, device=device)
    for i in range(pred_len):
        metrics[f'pinaw_{i}'] = torch.zeros(1, device=device)
    for i in range(pred_len):
        metrics[f'cwc_{i}'] = torch.zeros(1, device=device)
    return metrics


def update_metrics(metrics, samples, labels, filter_nan=False):  # [99, 256, 96], [256, 96]
    # filter out nan
    if filter_nan:
        for i in range(samples.shape[0]):
            if torch.isnan(samples[i]).sum() > 0:
                nan_mask = torch.isnan(samples[i])
                samples[i][nan_mask] = labels[nan_mask]

    # record metrics
    batch_size = samples.shape[1]
    metrics['num'] = metrics['num'] + batch_size
    metrics['CRPS'] = metrics['CRPS'] + accuracy_CRPS(samples, labels)
    metrics['mre'] = metrics['mre'] + accuracy_MRE(samples, labels)
    metrics['naps'] = metrics['naps'] + accuracy_NAPS(samples)
    metrics['picp'] = metrics['picp'] + accuracy_PICP(samples)
    for i in range(1, 10, 1):
        alpha = i / 20
        metrics[f'picp_alpha{alpha}'] = metrics[f'picp_alpha{alpha}'] + accuracy_PICP(samples, alpha)
    metrics['pinaw'] = metrics['pinaw'] + accuracy_PINAW(samples)
    return metrics


def final_metrics(metrics, seq_len):
    summary = {}
    # crps
    summary['CRPS'] = metrics['CRPS'] / metrics['num']
    for i in range(seq_len):
        summary[f'CRPS_{i}'] = summary['CRPS'][i]
    summary['CRPS'] = summary['CRPS'].mean()
    # mre
    summary['mre'] = metrics['mre'] / metrics['num']
    summary['mre'] = summary['mre'].T - torch.arange(0.05, 1, 0.05, device=metrics['mre'].device)
    # naps
    summary['naps'] = metrics['naps'] / metrics['num']
    for i in range(seq_len):
        summary[f'naps_{i}'] = summary['naps'][i]
    summary['naps'] = summary['naps'].mean()
    # picp & pinaw & cwc
    summary['picp'] = metrics['picp'] / metrics['num']
    summary['pinaw'] = metrics['pinaw'] / metrics['num']
    summary['cwc'] = accuracy_CWC(summary['picp'], summary['pinaw'])
    for i in range(seq_len):
        summary[f'picp_{i}'] = summary['picp'][i]
    summary['picp'] = summary['picp'].mean()
    for i in range(seq_len):
        summary[f'pinaw_{i}'] = summary['pinaw'][i]
    summary['pinaw'] = summary['pinaw'].mean()
    for i in range(seq_len):
        summary[f'cwc_{i}'] = summary['cwc'][i]
    summary['cwc'] = summary['cwc'].mean()
    # picp alpha
    for i in range(1, 10, 1):
        alpha = i / 20
        summary[f'picp_alpha{alpha}'] = (metrics[f'picp_alpha{alpha}'] / metrics['num']).mean()
    return summary


def accuracy_CRPS(samples: torch.Tensor, labels: torch.Tensor):  # [99, 256, 96], [256, 96]
    samples_permute = samples.permute(1, 2, 0)
    crps = ps.crps_ensemble(labels.cpu().detach().numpy(),
                            samples_permute.cpu().detach().numpy()).sum(axis=0)
    return torch.Tensor(crps).to(samples.device)


def accuracy_MRE(samples: torch.Tensor, labels: torch.Tensor):  # [99, 256, 96], [256, 96]
    samples_sorted = samples.sort(dim=0).values
    df1 = torch.sum(samples_sorted > labels, dim=1)
    mre = df1[[i - 1 for i in range(5, 100, 5)], :]
    return mre


def accuracy_NAPS(samples: torch.Tensor):  # [99, 256, 96]
    out = torch.zeros(samples.shape[2], device=samples.device)  # [96]
    for i in range(1, 10, 1):
        q_n1 = samples.quantile(1 - i / 20, dim=0)  # [256, 96]
        q_n2 = samples.quantile(i / 20, dim=0)  # [256, 96]
        out = out + torch.sum((q_n1 - q_n2) / (1 - i / 10), dim=0)  # [96]
    out = out / 9
    return out


def accuracy_PICP(samples: torch.Tensor, alpha: float = 0.05):  # [99, 256, 96]
    n = samples.shape[0]
    q_n1 = samples.quantile(1 - alpha, dim=0)  # [256, 96]
    q_n1 = q_n1.unsqueeze(0).expand(n, -1, -1)  # [99, 256, 96]
    q_n2 = samples.quantile(alpha, dim=0)  # [256, 96]
    q_n2 = q_n2.unsqueeze(0).expand(n, -1, -1)  # [99, 256, 96]
    out = torch.sum((q_n1 >= samples) & (samples >= q_n2), dim=0) / n # [256, 96]
    out = out.sum(dim=0)  # [96]
    return out


def accuracy_PINAW(samples: torch.Tensor):  # [99, 256, 96]
    out = torch.zeros(samples.shape[2], device=samples.device)  # [96]
    for i in range(1, 10, 1):
        q_n1 = samples.quantile(1 - i / 20, dim=0)  # [256, 96]
        q_n2 = samples.quantile(i / 20, dim=0)  # [256, 96]
        min = torch.min(samples, dim=0).values  # [256, 96]
        max = torch.max(samples, dim=0).values  # [256, 96]
        out = out + torch.sum((q_n1 - q_n2) / (max - min), dim=0)  # [96]
    out = out / 9
    return out


def accuracy_CWC(picp: torch.Tensor, pinaw: torch.Tensor):
    out = picp * (1 - pinaw)
    return out
