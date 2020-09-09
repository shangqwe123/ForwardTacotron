import torch


def attention_score(att, x_lens, mel_lens):
    device = att.device
    b, t_max, c_max = att.size()
    mel_range = torch.arange(1, t_max, device=device)
    mel_mask = (mel_range[None, :] < mel_lens[:, None]).float()
    x_range = torch.arange(0, c_max, device=device)
    x_mask = (x_range[None, :] < x_lens[:, None]).float()
    mel_max = torch.argmax(att, dim=2)
    mel_max_diff = torch.abs(mel_max[:, 1:] - mel_max[:, :-1])
    mel_max_diff = (mel_max_diff < 2).float()
    mel_max_diff = mel_max_diff * mel_mask
    mel_max_sum = torch.sum(mel_max_diff, dim=1)
    x_max = torch.argmax(att, dim=1).long()
    x_max = x_max * x_mask
    x_coverage = [torch.unique(x_max[i]).size(0) for i in range(b)]
    x_coverage = torch.tensor(x_coverage, device=device, dtype=torch.float)
    corr_score = mel_max_sum / (mel_lens - 1)
    cov_score = x_coverage / x_lens
    score = corr_score * cov_score
    return score