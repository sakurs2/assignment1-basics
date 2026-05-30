import torch
from torch import dtype
from pathlib import Path
import numpy as np

from cs336_basics.utils import get_device
from cs336_basics.AdamW import AdamW
from cs336_basics.learning_rate_schedule import LearningRateSchedule
from cs336_basics.config import ModelConfig, TrainConfig
from cs336_basics.model import GPTModel
from cs336_basics.pretokenization import train_bpe
from cs336_basics.dataloader import get_batch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.cross_entropy import CrossEntropy
from cs336_basics.gradient_clipping import GradientClipping

input_train = Path(
    "/Users/bytedance/Desktop/github/assignment1-basics/data/tinyshakespeare.txt"
)


# TODO:
# 1. 修复bpe产物vocab保存报错
# 2. get_batch从memmap获取
# 3. 支持温度参数、top_k采样
# 4. 细节待打磨
def train(trainConfig: TrainConfig, modelConfig: ModelConfig):
    device = get_device()

    vocab, merges = train_bpe(
        input_train, modelConfig.vocab_size, special_tokens=["<|endoftext|>"]
    )
    token = Tokenizer(vocab, merges)
    data = input_train.read_text(encoding="utf-8")
    ids = token.encode(data)
    np_ids = np.array(ids, dtype=np.uint16)
    print(f"=========> data ready!")

    model = GPTModel(modelConfig)
    model.to(device)

    scaler = torch.amp.grad_scaler.GradScaler(enabled=(dtype == "float16"))
    optimizer = AdamW(model.parameters())
    iter: int = 0

    while iter < trainConfig.max_iters:
        lr = LearningRateSchedule(
            iter,
            trainConfig.max_lr,
            trainConfig.min_lr,
            trainConfig.warmup_iters,
            trainConfig.cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(
            np_ids, trainConfig.batch_size, modelConfig.context_length, device.type
        )
        logits = model(x)
        loss = CrossEntropy(logits, y)
        scaler.scale(loss).backward()

        # scaler.unscale_(optimizer)
        # GradientClipping(model.parameters(), max)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        iter += 1
        if iter % 100 == 0:
            print(f"=========> iter {iter} finish")


if __name__ == "__main__":
    modelConfig = ModelConfig()
    trainConfig = TrainConfig()
    train(trainConfig, modelConfig)
