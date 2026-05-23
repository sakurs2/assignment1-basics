import torch
import math
from einops import einsum
from .scaled_dot_product_attention import ScaledDotProductAttention
from .rope import Rope
from .linear import Linear


class MultiheadSelfAttention(torch.nn.Module):
    __slots__ = ("d_model", "num_heads", "head_dim")

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        if (
            theta is not None
            and token_positions is not None
            and max_seq_len is not None
        ):
            self.rope = Rope(theta, self.head_dim, max_seq_len)
            self.token_positions = token_positions
        else:
            self.rope = None
            self.token_positions = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        assert D == self.d_model, "d_model must equal to D"

        # shape (B, L, D)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # shape (B, num_heads, L, head_dim)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rope
        if self.rope is not None:
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)

        # attn mask
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
        # shape (B, num_heads, L, head_dim)
        out = ScaledDotProductAttention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)
