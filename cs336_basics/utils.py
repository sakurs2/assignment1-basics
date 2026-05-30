from __future__ import annotations
from typing import Optional
import torch

KV = tuple[bytes, int]


def _lower_bound(items: list[KV], key: bytes) -> int:
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid][0] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_find(items: list[KV], key: bytes) -> Optional[KV]:
    i = _lower_bound(items, key)
    if i < len(items) and items[i][0] == key:
        return items[i]
    return None


def binary_upsert(items: list[KV], key: bytes, value: int) -> int:
    i = _lower_bound(items, key)
    if i < len(items) and items[i][0] == key:
        items[i] = (key, value)  # 更新
    else:
        items.insert(i, (key, value))  # 插入（保持有序）
    return i


def binary_remove(items: list[KV], key: bytes):
    i = _lower_bound(items, key)
    if i < len(items) and items[i][0] == key:
        del items[i]


def one_hot(
    indices: torch.Tensor, num_classes: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    scatter_ 的核心作用: 按照索引张量指定的位置，将值 "分散" 写入到目标张量中。
    语义可以用一句话概括：沿着 dim 维度，把 src 的值写到 self 中 index 指定的位置。
    对于一个 3D 张量，scatter_(dim, index, src) 的等价伪代码为：

    for i in range(index.size(0)):
        for j in range(index.size(1)):
            for k in range(index.size(2)):
                if dim == 0:
                    self[index[i][j][k], j, k] = src[i][j][k]
                elif dim == 1:
                    self[i, index[i][j][k], k] = src[i][j][k]
                elif dim == 2:
                    self[i, j, index[i][j][k]] = src[i][j][k]
    """
    return torch.zeros(
        *indices.shape, num_classes, dtype=dtype, device=indices.device
    ).scatter_(-1, indices.unsqueeze(-1), 1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def to_jsonable(obj):
    """把 dict 里的 bytes 转成 str，递归处理"""
    if isinstance(obj, bytes):
        return obj.decode("utf-8")  # 纯二进制改用 base64，见下
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj
