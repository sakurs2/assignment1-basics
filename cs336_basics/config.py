from dataclasses import dataclass


@dataclass
class ModelConfig:
    context_length: int = 1024
    vocab_size: int = 10240
    n_embd: int = 128
    d_model: int = 128
    d_ff: int = 128 * 4
    n_layer: int = 12
    n_head: int = 16
    rope_theta: float = 10000.0
    # dropout: float = 0.0
    # bias: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_iters: int = 10000
    warmup_iters: int = 1000
    cosine_cycle_iters: int = 3000
    max_lr: float = 0.001
    min_lr: float = 0.0001
