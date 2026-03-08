import math
import torch
from einops import einsum


class SwiGlu(torch.nn.Module):
    __slots__ = ("w1", "w2", "w3")

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # w1
        w1 = torch.empty(d_ff, d_model, device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1)
        self.w1 = torch.nn.Parameter(w1, requires_grad=True)

        # w3
        w3 = torch.empty(d_ff, d_model, device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w3)
        self.w3 = torch.nn.Parameter(w3, requires_grad=True)

        # w2
        w2 = torch.empty(d_model, d_ff, device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2)
        self.w2 = torch.nn.Parameter(w2, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w1*x
        x1 = einsum(x, self.w1, "... d_model,d_ff d_model->... d_ff")
        # SiLU(W1x)
        x1 = x1 * torch.sigmoid(x1)
        # w3*x
        x3 = einsum(x, self.w3, "... d_model,d_ff d_model->... d_ff")

        y = x1 * x3
        return einsum(self.w2, y, "d_model d_ff,... d_ff -> ... d_model")


if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    g = SwiGlu(4, 16)
    out = g.forward(a)
    print("out: ", out, ", shape: ", out.shape)
