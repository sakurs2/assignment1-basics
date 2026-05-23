import torch


def Softmax(input: torch.Tensor, dim_pos: int) -> torch.Tensor:
    c: float = 1e-5
    x_max = input.max(dim=dim_pos, keepdim=True).values
    x_stable = input - x_max
    exp_x = torch.exp(x_stable)
    return exp_x / exp_x.sum(dim=dim_pos, keepdim=True)


def LogSoftmax(input: torch.Tensor, dim_pos: int) -> torch.Tensor:
    c: float = 1e-5
    x_max = input.max(dim=dim_pos, keepdim=True).values
    x_stable = input - x_max
    exp_x = torch.exp(x_stable)
    return x_stable - torch.log(exp_x.sum(dim=dim_pos, keepdim=True))


if __name__ == "__main__":
    input = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    res = Softmax(input, 0)
    print(f"res: {res}")
