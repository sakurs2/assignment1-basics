import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .softmax import LogSoftmax


# ..., batch, seq_len, vocab_size
def CrossEntropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
):
    B, V = inputs.shape
    x = LogSoftmax(inputs, -1)
    result = x.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return -result.mean(dim=-1).mean(dim=-1)
