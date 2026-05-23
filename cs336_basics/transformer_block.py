import torch

from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiheadSelfAttention
from .swiglu import SwiGlu


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, theta, max_seq_len, token_positions
        )
        self.rmsnorm2 = RMSNorm(d_model)
        self.swiglu = SwiGlu(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rmsnorm1(x)
        h = self.attn(h)
        x = x + h

        h = self.rmsnorm2(x)
        return x + self.swiglu(h)
