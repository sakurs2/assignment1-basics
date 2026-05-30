import sys
import json
import pickle
import numpy as np
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import train_bpe
from cs336_basics.config import ModelConfig
from cs336_basics.utils import to_jsonable

input_train = Path(
    "/Users/bytedance/Desktop/github/assignment1-basics/data/tinyshakespeare.txt"
)

vocab_save_path = Path(
    "/Users/bytedance/Desktop/github/assignment1-basics/temp/vocab.pkl"
)
merge_save_path = Path(
    "/Users/bytedance/Desktop/github/assignment1-basics/temp/merge.pkl"
)

config = ModelConfig()

print(f"========== begin train ppe ==========")

vocab, merges = train_bpe(
    input_train, config.vocab_size, special_tokens=["<|endoftext|>"]
)

print(f"========== train ppe finish ==========")

vocab_save_path.write_text(
    json.dumps(
        to_jsonable(vocab),
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

merge_save_path.write_text(
    json.dumps(
        to_jsonable(merges),
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(f"========== Tokenizer init ==========")

token = Tokenizer.from_files(vocab_save_path, merge_save_path)
if token is None:
    print(f"========== Tokenizer init failed ==========")
    sys.exit(0)

print(f"========== Tokenizer init finish ==========")

text = "hello world!!!!"
ids = token.encode(text)
decode = token.decode(ids)
print(f"decode text: {decode}")
