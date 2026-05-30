from typing import Any

import torch
from cs336_basics.transformer import Transformer
from cs336_basics.config import ModelConfig


class GPTModel(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.trans = Transformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.n_head,
            d_ff=config.d_ff,
            context_length=config.context_length,
            num_layers=config.n_layer,
            rope_theta=config.rope_theta,
            token_positions=torch.arange(0, config.context_length, dtype=torch.int),
        )

    def forward(self, x: torch.Tensor):
        return self.trans(x)
