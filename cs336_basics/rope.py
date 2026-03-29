import torch
import math


class Rope(torch.nn.Module):
    __slots__ = "freqs_cis"

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(pos, freqs)
        freqs_cis = torch.polar(torch.ones_like(angles, device=device), angles)
        self.register_buffer("freqs_cis", freqs_cis, False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        freqs_cis = self.freqs_cis[token_positions]
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * freqs_cis
        return torch.view_as_real(x_rotated).flatten(-2)


if __name__ == "__main__":
    rope = Rope(2, 4, 6)
    input = torch.Tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]])
    pos = torch.tensor([[0, 1], [0, 1]])
    print(rope.forward(input, pos))
