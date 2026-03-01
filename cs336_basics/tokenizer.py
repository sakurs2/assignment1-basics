import sys
import json
import time
from pathlib import Path
import regex as re
import numpy as np
from datetime import datetime
from typing import Self, Dict, List, Tuple, Iterable, Iterator
from dataclasses import dataclass, field
from .pretokenization import PAT


CHILD_SIZE = 256


@dataclass(slots=True)
class DictTreeNode:
    childs: np.ndarray = field(
        default_factory=lambda: np.zeros(CHILD_SIZE, dtype=np.uint32)
    )  # value means nodes position

    # means pos in vocab
    # value - 1 == vocab key
    value: int = field(default=0)


class Tokenizer:
    __slots__ = (
        "_nodes",
        "_base_node",
        "_vocab",
        "_merges",
        "_merges_index",
        "_special_tokens",
    )

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        if special_tokens:
            for token in special_tokens:
                vocab[len(vocab)] = token.encode()
            self._special_tokens = split_tokens = "|".join(
                re.escape(token) for token in special_tokens
            )
        else:
            self._special_tokens = None

        self._vocab = vocab
        self._base_node = DictTreeNode()
        self._nodes: list[DictTreeNode] = [self._base_node]

        self._merges = merges
        self._merges_index: Dict[bytes, int] = {}
        for i in range(len(merges)):
            if merges[i][0] not in self._merges_index.keys():
                self._merges_index[merges[i][0]] = i

        # Build dict tree
        for key, value in vocab.items():
            encodes = list(value)
            base_node = self._base_node
            for byte in encodes:
                if base_node.childs[byte] == 0:
                    self._nodes.append(DictTreeNode())
                    base_node.childs[byte] = len(self._nodes) - 1

                base_node = self._nodes[base_node.childs[byte]]
            base_node.value = key + 1

    @classmethod
    def from_files(
        cls, vocab_filepath, merges_filepath, special_tokens=None
    ) -> "Tokenizer|None":
        if vocab_filepath == "" or merges_filepath == "":
            return None

        vocab: dict[int, bytes] = json.loads(Path(vocab_filepath).read_text())
        merges: list[tuple[bytes, bytes]] = json.loads(
            Path(merges_filepath).read_text()
        )
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens: list[str] = self._get_byte_tokens(text)

        res: list[int] = []
        for token in tokens:
            encodes = token.encode()

            # # base_node: DictTreeNode = self._base_node

            # idx = 0
            # while idx < len(encodes):
            #     i = idx
            #     last = -1
            #     base_node: DictTreeNode = self._base_node
            #     while i < len(encodes):
            #         if base_node.childs[encodes[i]] != 0:
            #             base_node = self._nodes[base_node.childs[encodes[i]]]
            #             if base_node.value != 0:
            #                 last = base_node.value - 1
            #                 idx = i + 1
            #         else:
            #             idx = i
            #             break
            #         i += 1

            #     if last != -1:
            #         res.append(last)
            # for byte in encodes:
            #     if base_node.childs[byte] != 0:
            #         base_node = self._nodes[base_node.childs[byte]]
            #     else:
            #         res.append(base_node.value - 1)
            #         base_node = self._nodes[self._base_node.childs[byte]]
            # if base_node.value != 0:
            #     res.append(base_node.value - 1)
        return res

    def _get_byte_tokens(self, text: str) -> list[str]:
        if self._special_tokens and self._special_tokens != "":
            sub_texts = re.split(self._special_tokens, text)
        else:
            sub_texts = [text]

        res: list[str] = []
        for sub_text in sub_texts:
            for token in re.finditer(PAT, sub_text):
                res.append(token.group())
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens: list[str] = self._get_byte_tokens(text)
            for token in tokens:
                encodes = token.encode()
                base_node: DictTreeNode = self._base_node
                for byte in encodes:
                    if base_node.childs[byte] != 0:
                        base_node = self._nodes[base_node.childs[byte]]
                    else:
                        yield base_node.value - 1
                        base_node = self._nodes[self._base_node.childs[byte]]
                if base_node.value != 0:
                    yield base_node.value - 1

    def decode(self, ids: list[int]) -> str:
        output: bytes = b""
        for id in ids:
            if id in self._vocab.keys():
                output += self._vocab[id]
            else:
                output += b"\xef\xbf\xbd"  # "\ufffd".encode()
        return output.decode()
