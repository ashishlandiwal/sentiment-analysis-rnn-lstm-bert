"""PyTorch sequence classifiers: a vanilla RNN and an LSTM, sharing one architecture."""
from __future__ import annotations

import torch
from torch import nn


class SequenceClassifier(nn.Module):
    """Embedding -> (RNN | LSTM) -> last hidden state -> linear classifier."""

    def __init__(
        self,
        vocab_size: int,
        cell: str = "lstm",
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if cell not in ("rnn", "lstm"):
            raise ValueError("cell must be 'rnn' or 'lstm'")
        self.cell = cell
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        if self.cell == "lstm":
            hidden = hidden[0]  # (h_n, c_n) -> h_n
        last = hidden[-1]  # final layer's hidden state: (batch, hidden_dim)
        return self.fc(self.dropout(last))


def make_model(vocab_size: int, cell: str = "lstm", **kwargs) -> SequenceClassifier:
    return SequenceClassifier(vocab_size, cell=cell, **kwargs)
