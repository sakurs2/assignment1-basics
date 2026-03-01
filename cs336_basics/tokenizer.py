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
        "_reverse_vocab",
        "_merges",
        "_merges_index",
        "_special_tokens",
        "_split_tokens",
    )

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            self._split_tokens = (
                "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
            )
        self._special_tokens = special_tokens

        self._vocab = vocab
        self._reverse_vocab: Dict[bytes, int] = {}
        for key, value in self._vocab.items():
            self._reverse_vocab[value] = key
        if special_tokens:
            new_add: list[bytes] = []
            for token in special_tokens:
                if token.encode() not in self._reverse_vocab.keys():
                    new_add.append(token.encode())
            for token in new_add:
                self._reverse_vocab[token] = len(self._vocab)
                self._vocab[len(self._vocab)] = token

        self._base_node = DictTreeNode()
        self._nodes: list[DictTreeNode] = [self._base_node]

        self._merges = merges
        self._merges_index: Dict[bytes, list[bytes]] = {}
        for i in range(len(merges)):
            # if (
            #     merges[i][0] == "H".encode()
            #     or merges[i][0] == "He".encode()
            #     or merges[i][0] == "Hel".encode()
            #     or merges[i][0] == "Hell".encode()
            # ):
            #     print(f"\n merges: {merges[i]}")

            if merges[i][0] not in self._merges_index.keys():
                self._merges_index[merges[i][0]] = [merges[i][1]]
            else:
                self._merges_index[merges[i][0]].append(merges[i][1])

        for key in self._merges_index.keys():
            self._merges_index[key] = sorted(
                self._merges_index[key], key=lambda x: len(x), reverse=True
            )

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
            if encodes in self._reverse_vocab.keys():
                res.append(self._reverse_vocab[encodes])
                continue
            idx = 1
            cur = bytes([encodes[0]])
            while idx < len(encodes):
                find_merge = False
                if cur in self._merges_index.keys():
                    for val in self._merges_index[cur]:
                        if (
                            idx + len(val) - 1 <= len(encodes)
                            and encodes[idx : (idx + len(val))] == val
                        ):
                            cur += val
                            idx += len(val)
                            find_merge = True
                            break

                if find_merge:
                    continue
                # no match
                res.append(self._reverse_vocab[cur])
                cur = bytes([encodes[idx]])
                idx += 1
            res.append(self._reverse_vocab[cur])
            if "Leland" in token:
                print(
                    f"\n ==========> res: {res}, token: {token}, mp: {self._merges_index["L".encode()]},1004: {self._vocab[1004]}, 1044: {self._vocab[1044]}, 406: {self._vocab[406]}, 8822: {self._vocab[8822]}"
                )

        return res

    def _get_byte_tokens(self, text: str) -> list[str]:
        if self._special_tokens and self._special_tokens != "":
            sub_texts = re.split(self._split_tokens, text)
        else:
            sub_texts = [text]

        res: list[str] = []
        for sub_text in sub_texts:
            if self._special_tokens and sub_text in self._special_tokens:
                res.append(sub_text)
                continue
            for token in re.finditer(PAT, sub_text):
                res.append(token.group())
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens: list[str] = self._get_byte_tokens(text)
            for token in tokens:
                encodes = token.encode()
                if encodes in self._reverse_vocab.keys():
                    yield self._reverse_vocab[encodes]
                    continue
                idx = 1
                cur = bytes([encodes[0]])
                while idx < len(encodes):
                    find_merge = False
                    if cur in self._merges_index.keys():
                        for val in self._merges_index[cur]:
                            if (
                                idx + len(val) - 1 <= len(encodes)
                                and encodes[idx : (idx + len(val))] == val
                            ):
                                cur += val
                                idx += len(val)
                                find_merge = True
                                break

                    if find_merge:
                        continue
                    # no match
                    yield self._reverse_vocab[cur]
                    cur = bytes([encodes[idx]])
                    idx += 1
                yield self._reverse_vocab[cur]

    def decode(self, ids: list[int]) -> str:
        output: bytes = b""
        for id in ids:
            if id in self._vocab.keys():
                output += self._vocab[id]
            else:
                output += b"\xef\xbf\xbd"  # "\ufffd".encode()
        return output.decode(errors="replace")
