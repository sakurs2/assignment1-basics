import math
import torch
from einops import einsum, reduce


class RMSNorm(torch.nn.Module):
    __slots__ = ("eps", "g")

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        g = torch.empty(d_model, device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(g)
        self.g = torch.nn.Parameter(g, requires_grad=True)

    # input shape: (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = einsum(x, x, "... d_model,... d_model -> ... d_model")
        result = reduce(result, "... d_model -> ... 1", "mean") + self.eps
        result = x / torch.sqrt(result)
        return einsum(result, self.g, "... d_model,d_model -> ... d_model")


if __name__ == "__main__":
    input = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    rms = RMSNorm(4)
    output = rms.forward(input)
    print(output)
