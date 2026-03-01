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


class Tokenizer:
    __slots__ = (
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

        self._merges = merges
        self._merges_index: Dict[bytes, list[bytes]] = {}
        for i in range(len(merges)):
            if merges[i][0] not in self._merges_index.keys():
                self._merges_index[merges[i][0]] = [merges[i][1]]
            else:
                self._merges_index[merges[i][0]].append(merges[i][1])

        for key in self._merges_index.keys():
            self._merges_index[key] = sorted(
                self._merges_index[key], key=lambda x: len(x), reverse=True
            )

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
            res.extend(self._encode_token(token))

        return res

    def _encode_token(self, text: str) -> list[int]:
        encodes = text.encode()
        if encodes in self._reverse_vocab.keys():
            return [self._reverse_vocab[encodes]]

        tokens: list[bytes] = [bytes([b]) for b in encodes]

        while (len(tokens)) > 2:
            best_pair = None
            best_rank = len(self._vocab)

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merges and (
                    best_pair is None
                    or self._reverse_vocab[tokens[i] + tokens[i + 1]] < best_rank
                ):
                    best_pair = pair
                    best_rank = self._reverse_vocab[tokens[i] + tokens[i + 1]]

            if best_pair is None:
                break

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self._reverse_vocab[t] for t in tokens]

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
                res = self._encode_token(token)
                for item in res:
                    yield item

    def decode(self, ids: list[int]) -> str:
        output: bytes = b""
        for id in ids:
            if id in self._vocab.keys():
                output += self._vocab[id]
            else:
                output += b"\xef\xbf\xbd"  # "\ufffd".encode()
        return output.decode(errors="replace")
