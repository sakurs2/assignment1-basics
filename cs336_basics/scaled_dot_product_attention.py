import torch
import math
from einops import einsum
from .softmax import Softmax


def ScaledDotProductAttention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    dim = q.shape[-1]
    qk = einsum(q, k, "... seq_len1 dim,... seq_len2 dim->... seq_len1 seq_len2")
    qk = qk / math.sqrt(dim)
    if mask != None:
        qk = qk.masked_fill(~mask, float("-inf"))
    qk = Softmax(qk, -1)
    res = einsum(qk, v, "... seq_len1 seq_len2,... seq_len2 dim_v->... seq_len1 dim_v")
    return res


if __name__ == "__main__":
    q = torch.Tensor([[1, 2], [3, 4]])
    k = torch.Tensor([[5, 6], [7, 8]])
    ScaledDotProductAttention(q, k, q)
