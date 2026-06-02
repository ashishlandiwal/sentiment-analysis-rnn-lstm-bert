"""Dataset loading.

`load_rotten_tomatoes` pulls the real Cornell Rotten Tomatoes sentiment dataset via
HuggingFace `datasets` (8,530 train / 1,066 test, binary). `tiny_dataset` is a small,
bundled, linearly-separable corpus used by the unit tests so they run fast and offline
with no downloads.
"""
from __future__ import annotations

import random

DATASET_ID = "cornell-movie-review-data/rotten_tomatoes"


def load_rotten_tomatoes(
    train_subset: int | None = None,
    test_subset: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    train_texts = list(ds["train"]["text"])
    train_labels = [int(x) for x in ds["train"]["label"]]
    test_texts = list(ds["test"]["text"])
    test_labels = [int(x) for x in ds["test"]["label"]]

    if train_subset:
        train_texts, train_labels = _subsample(train_texts, train_labels, train_subset, seed)
    if test_subset:
        test_texts, test_labels = _subsample(test_texts, test_labels, test_subset, seed)
    return train_texts, train_labels, test_texts, test_labels


def _subsample(texts, labels, k, seed):
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    idx = idx[:k]
    return [texts[i] for i in idx], [labels[i] for i in idx]


# ----- tiny, offline corpus for tests -------------------------------------------------
# Controlled vocabulary: the sentiment-bearing words recur across train and test so the
# held-out examples are in-vocabulary (a fair smoke test of learning, not of OOV handling).
_POS = [
    "i love this great movie",
    "a wonderful and excellent film",
    "great fantastic and wonderful story",
    "i love it so good and great",
    "an excellent and fantastic movie",
    "really good and wonderful film",
    "love this excellent great film",
    "a good and fantastic wonderful movie",
]
_NEG = [
    "i hate this terrible movie",
    "a boring and awful film",
    "terrible bad and boring story",
    "i hate it so poor and bad",
    "an awful and terrible movie",
    "really bad and boring film",
    "hate this awful terrible film",
    "a poor and bad boring movie",
]


def tiny_dataset() -> tuple[list[str], list[int], list[str], list[int]]:
    """Small separable corpus: first 6 of each class for train, last 2 for test."""
    train_texts = _POS[:6] + _NEG[:6]
    train_labels = [1] * 6 + [0] * 6
    test_texts = _POS[6:] + _NEG[6:]
    test_labels = [1] * 2 + [0] * 2
    return train_texts, train_labels, test_texts, test_labels
