import math
import torch
from einops import einsum
from .utils import one_hot


class Embedding(torch.nn.Module):
    __slots__ = ("vocab_size", "embedding")

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = num_embeddings
        embedding = torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device=device,
        )
        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        torch.nn.init.trunc_normal_(embedding, mean=0, std=std, a=-3 * std, b=3 * std)
        self.embedding = torch.nn.Parameter(embedding, requires_grad=True)

    # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = one_hot(token_ids, self.embedding.size()[0], self.embedding.dtype)
        return einsum(
            x,
            self.embedding,
            "... seq_len vocab_size, vocab_size d_model -> ... seq_len d_model",
        )


if __name__ == "__main__":
    emb = Embedding(5, 4)
    print("embedding ", emb.embedding)
    input = torch.tensor([[0, 4], [1, 2]])
    output = emb.forward(input)
    print("result: ", output)
