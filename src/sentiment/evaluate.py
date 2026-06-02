"""Metrics and the model-comparison plot."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_comparison(results: dict[str, dict], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results)
    accs = [results[m]["accuracy"] for m in models]
    f1s = [results[m]["f1"] for m in models]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, accs, width, label="accuracy")
    ax.bar(x + width / 2, f1s, width, label="macro-F1")
    ax.set_xticks(x, labels=[m.upper() for m in models])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Sentiment models on Rotten Tomatoes (test)")
    for i, (a, f) in enumerate(zip(accs, f1s, strict=True)):
        ax.text(i - width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
