import torch
import numpy as np
import numpy.typing as npt
from pathlib import Path

# FILE_PATH = "/Users/bytedance/Desktop/github/assignment1-basics/temp/data.npy"


# class DataLoader:
#     def __init__(
#         self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str
#     ) -> None:
#         self.batch_size = batch_size
#         self.context_length = context_length
#         self.device = device
#         self.shape = dataset.shape
#         self.dtype = dataset.dtype
#         np.save(FILE_PATH, dataset)

#     def __iter__(self):
#         self.data = np.memmap(FILE_PATH, self.dtype, mode="r", shape=self.shape)
#         idx = 0
#         mx_len = len(self.data)
#         while True:
#             pass


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    ix = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
            for i in ix
        ]
    )
    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y
