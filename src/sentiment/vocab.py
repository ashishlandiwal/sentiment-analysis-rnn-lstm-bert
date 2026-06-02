"""Word-level vocabulary and encoding for the RNN/LSTM models."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9']+")
PAD = "<pad>"
UNK = "<unk>"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2, max_size: int = 20000) -> Vocab:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        tokens = [w for w, f in counter.most_common() if f >= min_freq][:max_size]
        return cls([PAD, UNK] + tokens)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokenize(text)][:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids
