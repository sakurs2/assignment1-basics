from typing import Any

import torch

from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,
        token_positions: torch.Tensor,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    token_positions,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.o = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.norm(x)
        x = self.o(x)
        return x
