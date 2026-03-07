import math
import torch
from einops import einsum


class Linear(torch.nn.Module):
    __slots__ = "w"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        w = torch.empty(out_features, in_features, dtype=dtype, device=device)
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(w, mean=0, std=std, a=-3 * std, b=3 * std)
        self.w = torch.nn.Parameter(w, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... k, j k -> ... j")


if __name__ == "__main__":
    a = torch.randn(2, 3, 4)
    l = Linear(4, 5)
    out = l.forward(a)
    print(l.state_dict())
