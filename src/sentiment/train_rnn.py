"""Train the RNN / LSTM sequence classifiers."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import evaluate
from .models import make_model
from .vocab import Vocab


def _encode(texts, vocab: Vocab, max_len: int) -> torch.Tensor:
    return torch.tensor([vocab.encode(t, max_len) for t in texts], dtype=torch.long)


def _predict(model, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(model(X[i : i + batch_size]).argmax(1))
    return torch.cat(preds).cpu().numpy()


def train_sequence_model(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    *,
    cell: str = "lstm",
    max_len: int = 40,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    embed_dim: int = 64,
    hidden_dim: int = 64,
    min_freq: int = 2,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab = Vocab.build(train_texts, min_freq=min_freq)
    X_train = _encode(train_texts, vocab, max_len)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = _encode(test_texts, vocab, max_len)

    model = make_model(len(vocab), cell=cell, embed_dim=embed_dim, hidden_dim=hidden_dim, pad_id=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            # Gradient clipping stabilises recurrent training (esp. the vanilla RNN).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    preds = _predict(model, X_test)
    return model, vocab, evaluate.classification_metrics(test_labels, preds)
