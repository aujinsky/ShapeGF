import torch
import random
import numpy as np
from torch import optim


def get_opt(params, cfgopt):
    if cfgopt.type == 'adam':
        optimizer = optim.Adam(params, lr=float(cfgopt.lr),
                               betas=(cfgopt.beta1, cfgopt.beta2),
                               weight_decay=cfgopt.weight_decay)
    elif cfgopt.type == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=float(cfgopt.lr), momentum=cfgopt.momentum)
    else:
        assert 0, "Optimizer type should be either 'adam' or 'sgd'"

    scheduler = None
    scheduler_type = getattr(cfgopt, "scheduler", None)
    if scheduler_type is not None:
        if scheduler_type == 'exponential':
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
        elif scheduler_type == 'step':
            step_size = int(getattr(cfgopt, "step_epoch", 500))
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay)
        elif scheduler_type == 'linear':
            step_size = int(getattr(cfgopt, "step_epoch", 2000))
            final_ratio = float(getattr(cfgopt, "final_ratio", 0.01))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.5))
            duration_ratio = float(getattr(cfgopt, "duration_ratio", 0.45))

            def lambda_rule(ep):
                lr_l = 1.0 - min(1, max(0, ep - start_ratio * step_size) / float(duration_ratio * step_size)) * (1 - final_ratio)
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif scheduler_type == 'cosine_anneal_nocycle':
            final_lr_ratio = float(getattr(cfgopt, "final_lr_ratio", 0.01))
            eta_min = float(cfgopt.lr) * final_lr_ratio
            eta_max = float(cfgopt.lr)

            total_epoch = int(getattr(cfgopt, "step_epoch", 2000))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.2))
            T_max = total_epoch * (1 - start_ratio)

            def lambda_rule(ep):
                curr_ep = max(0., ep - start_ratio * total_epoch)
                lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * curr_ep / T_max))
                lr_l = lr / eta_max
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear' or 'step'"
    return optimizer, scheduler


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ground_truth_field(prior_points, tr_pts, sigma):

    # 3.2
    # x_i = tr_pts
    # x = prior_points
    bs, num_pts = tr_pts.size(0), tr_pts.size(1)
    smp_pts = prior_points.size(1)
    prior_points = prior_points.view(bs, smp_pts, 1, -1)
    # bs, smp_pts, 1, dim
    tr_pts = tr_pts.view(bs, 1, num_pts, -1)
    # bs, 1, num_pts, dim
    dist = (prior_points - tr_pts).norm(dim=3, keepdim=True) ** 2.
    a = - dist / sigma ** 2.
    # argmin x_i x_i -x = argmax x_i x- x_i
    max_a, _ = torch.max(a, dim=2, keepdim=True)
    diff = torch.exp(a - max_a)
    w_i = diff / diff.sum(dim=2, keepdim=True)
    # w_i = (bs, smp_pts, num_pts, 1)
    # tr_pts = (bs, 1, num_pts, dim)
    # (bs, #pts-prior, 1, dim)
    trg_pts = (w_i * tr_pts).sum(dim=2, keepdim=True)
    y = - ((prior_points - trg_pts) / sigma ** 2.).view(bs, smp_pts, -1)
    return y
    """
    d2 = (prior_points-tr_pts).norm(dim=3, keepdim=True) ** 2.
    ww_i = -1/(sigma**2)*d2 # difference between code and paper. we use coeff in repo.
    w_i = ww_i / (ww_i.sum(dim=2, keepdim=True))
    grad = 1/ (sigma**2) * (-prior_points + (w_i * tr_pts).sum(dim=2, keepdim=True))
    grad = grad.view(bs, smp_pts, -1)
    return grad
    #doesn't works
    """



def ground_truth_reconstruct_multi(inp, cfg):
    # 3.4
    with torch.no_grad():
        assert hasattr(cfg, "inference")
        step_size_ratio = float(getattr(cfg.inference, "step_size_ratio", 1))
        num_steps = int(getattr(cfg.inference, "num_steps", 5))
        num_points = int(getattr(cfg.inference, "num_points", inp.size(1)))
        weight = float(getattr(cfg.inference, "weight", 1))
        sigma_begin = float(cfg.trainer.sigma_begin)
        sigma_end = float(cfg.trainer.sigma_end)
        num_classes = int(cfg.trainer.sigma_num) # k
        sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                                        num_classes))
        x = get_prior(inp.size(0), num_points, cfg.models.scorenet.dim).cuda()
        # reused argument getting part, sigma generating part, prior generating part
        # grad_ascent
        x_list = []
        x_list.append(x.clone())

        for sigma in sigmas:
            sigma = torch.full((1,), sigma).cuda()
            step_size = 2 * sigma ** 2 * step_size_ratio
            for t in range(num_steps):
                eps = torch.randn_like(x) * weight # in paper weight is always 1
                x = x + step_size/2* ground_truth_field(x, inp, sigma) + torch.sqrt(step_size)*eps
            x_list.append(x.clone())
    return x, x_list




def get_prior(batch_size, num_points, inp_dim):
    # -1 to 1, uniform
    return (torch.rand(batch_size, num_points, inp_dim) * 2 - 1.) * 1.5


