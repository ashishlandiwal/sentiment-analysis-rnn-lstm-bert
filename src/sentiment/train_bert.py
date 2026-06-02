"""Fine-tune DistilBERT for sentiment classification (manual PyTorch loop).

A manual loop is used (rather than the HF Trainer) to stay robust across transformers
versions and to keep the training logic explicit and readable.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import evaluate


def train_bert(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    *,
    model_name: str = "distilbert-base-uncased",
    max_len: int = 64,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 42,
):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def encode(texts):
        return tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )

    enc_train = encode(train_texts)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    dataset = TensorDataset(enc_train["input_ids"], enc_train["attention_mask"], y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for input_ids, attention_mask, yb in loader:
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=yb)
            out.loss.backward()
            optimizer.step()

    enc_test = encode(test_texts)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            logits = model(
                input_ids=enc_test["input_ids"][i : i + batch_size],
                attention_mask=enc_test["attention_mask"][i : i + batch_size],
            ).logits
            preds.append(logits.argmax(1))
    preds = torch.cat(preds).numpy()
    return model, tokenizer, evaluate.classification_metrics(test_labels, preds)
