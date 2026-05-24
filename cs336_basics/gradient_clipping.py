import torch
from collections.abc import Iterable


def GradientClipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    params = [p for p in parameters if p.grad is not None]
    grad_norm = sum(p.grad.data.pow(2).sum() for p in params).sqrt()
    if grad_norm <= max_l2_norm:
        return
    clip_coef = max_l2_norm / (grad_norm + eps)
    for p in params:
        p.grad.data.mul_(clip_coef)
