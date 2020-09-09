import torch


def attention_score(att, x_lens, mel_lens):
    device = att.device
    b, t_max, c_max = att.size()

    # create mel padding mask
    mel_range = torch.arange(0, t_max, device=device)
    mask = (mel_range[None, :] < mel_lens[:, None]).float()

    # score for how adjacent the attention loc is
    max_loc = torch.argmax(att, dim=2)
    max_loc_diff = torch.abs(max_loc[:, 1:] - max_loc[:, :-1])
    loc_score = (max_loc_diff < 2).float()
    loc_score = torch.sum(loc_score * mask[:, 1:], dim=1)
    loc_score = loc_score / (mel_lens - 1)

    # score for coverage of input phonemes
    max_loc_masked = max_loc * mask
    # account for x padding with another mask
    x_mask = (max_loc_masked < x_lens[:, None]).float()
    max_loc_masked = max_loc_masked * x_mask
    x_coverage = [torch.unique(max_loc_masked[i]).size(0) for i in range(b)]
    x_coverage = torch.tensor(x_coverage, device=device, dtype=torch.float)
    cov_score = x_coverage / x_lens
    score = loc_score * cov_score

    return score