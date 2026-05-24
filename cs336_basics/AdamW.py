from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, weight_decay, betas1, betas2, eps = (
                group["lr"],
                group["weight_decay"],
                group["betas"][0],
                group["betas"][1],
                group["eps"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]

                if "t" not in state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                m = betas1 * m + (1 - betas1) * g
                v = betas2 * v + (1 - betas2) * (g**2)

                m_hat = m / (1 - betas1**t)
                v_hat = v / (1 - betas2**t)

                p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                p.data -= lr * weight_decay * p.data

                state["m"] = m
                state["v"] = v
        return loss
