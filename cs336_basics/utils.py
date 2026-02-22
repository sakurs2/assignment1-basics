from __future__ import annotations
from typing import Optional

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
