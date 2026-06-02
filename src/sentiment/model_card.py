"""Generate a Markdown model card from a comparison payload."""
from __future__ import annotations


def generate(payload: dict) -> str:
    results = payload["results"]
    lines = [
        "# Model Card — Sentiment Analysis (RNN vs LSTM vs DistilBERT)",
        "",
        "## Overview",
        "- **Task:** binary sentiment classification of movie reviews.",
        f"- **Dataset:** {payload['dataset']} "
        f"({payload['train_size']} train / {payload['test_size']} test).",
        "- **Models:** a from-scratch RNN and LSTM (word embeddings) and a fine-tuned "
        "DistilBERT transformer.",
        "",
        "## Results (test set)",
        "",
        "| Model | Accuracy | Precision | Recall | Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name.upper()} | {m['accuracy'] * 100:.1f}% | {m['precision'] * 100:.1f}% | "
            f"{m['recall'] * 100:.1f}% | {m['f1'] * 100:.1f}% |"
        )

    bert = payload.get("bert")
    if bert:
        lines += [
            "",
            "## DistilBERT fine-tuning",
            f"- Base model: `{bert['model']}`",
            f"- Fine-tuned on {bert['train_examples']} examples for {bert['epochs']} epoch(s).",
            f"- _{bert['note']}._",
        ]

    lines += [
        "",
        "## Intended use & limitations",
        "- Educational comparison of recurrent vs transformer architectures for text classification.",
        "- Trained on short movie-review snippets; will not transfer cleanly to other domains "
        "(tweets, product reviews) without retraining.",
        "- The RNN/LSTM use a vocabulary built only from the training set; out-of-vocabulary words "
        "map to `<unk>`.",
        "",
        "_This card was generated automatically by `sentiment.model_card`._",
    ]
    return "\n".join(lines)
