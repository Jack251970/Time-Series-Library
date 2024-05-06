import torch
import matplotlib
import properscoring as ps

matplotlib.use('Agg')


def init_metrics(pred_len, device):
    metrics = {'num': torch.zeros(1, device=device), 'CRPS': torch.zeros(pred_len, device=device),
               'mre': torch.zeros((19, pred_len), device=device), 'pinaw': torch.zeros(pred_len, device=device)}
    for i in range(pred_len):
        metrics[f'CRPS_{i}'] = torch.zeros(1, device=device)
    for i in range(pred_len):
        metrics[f'pinaw_{i}'] = torch.zeros(1, device=device)
    return metrics


def update_metrics(metrics, samples, labels, pred_len):
    df = labels[:, -pred_len:]
    batch_size = samples.shape[1]
    metrics['num'] = metrics['num'] + batch_size
    metrics['CRPS'] = metrics['CRPS'] + accuracy_CRPS(samples, df)  # [99, 256, 96], [256, 96]
    for i in range(pred_len):
        metrics[f'CRPS_{i}'] = metrics[f'CRPS_{i}'] + accuracy_CRPS(samples[:, :, i].unsqueeze(-1), df[:, i].unsqueeze(-1))
    metrics['mre'] = metrics['mre'] + accuracy_MRE(samples, df)
    metrics['pinaw'] = metrics['pinaw'] + accuracy_PINAW(samples)
    for i in range(pred_len):
        metrics[f'pinaw_{i}'] = metrics[f'pinaw_{i}'] + accuracy_PINAW(samples[:, :, i].unsqueeze(-1))
    return metrics


def final_metrics(metrics, seq_len):
    summary = {'CRPS': metrics['CRPS'] / metrics['num'], 'pinaw': (metrics['pinaw'] / metrics['num']).mean()}
    for i in range(seq_len):
        summary[f'CRPS_{i}'] = metrics[f'CRPS_{i}'] / metrics['num']
    summary['mre'] = metrics['mre'] / metrics['num']
    summary['mre'] = summary['mre'].T - torch.arange(0.05, 1, 0.05, device=metrics['mre'].device)
    for i in range(seq_len):
        summary[f'pinaw_{i}'] = (metrics[f'pinaw_{i}'] / metrics['num']).mean()
    return summary


def accuracy_CRPS(samples: torch.Tensor, labels: torch.Tensor):
    samples_permute = samples.permute(1, 2, 0)
    crps = ps.crps_ensemble(labels.cpu().detach().numpy(),
                            samples_permute.cpu().detach().numpy()).sum(axis=0)
    _ = torch.Tensor(crps, device="cpu")
    return _.to(samples.device)


def accuracy_MRE(samples: torch.Tensor, labels: torch.Tensor):
    samples_sorted = samples.sort(dim=0).values
    df1 = torch.sum(samples_sorted > labels, dim=1)
    mre = df1[[i - 1 for i in range(5, 100, 5)], :]
    return mre


def accuracy_PINAW(samples: torch.Tensor):
    out = torch.zeros(samples.shape[2], device=samples.device)
    for i in range(10, 100, 10):
        q_n1 = samples.quantile(1 - i / 200, dim=0)
        q_n2 = samples.quantile(i / 200, dim=0)
        out = out + torch.sum((q_n1 - q_n2) / (1 - i / 100), dim=0)
    out = out / 9
    return out
