"""Train RNN, LSTM, and DistilBERT on the same split and compare them."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import evaluate
from .data import load_rotten_tomatoes
from .train_rnn import train_sequence_model


def run(
    *,
    max_len: int = 40,
    rnn_epochs: int = 5,
    bert_subset: int = 2000,
    bert_epochs: int = 2,
    skip_bert: bool = False,
    train_subset: int | None = None,
    output: str | Path = "reports",
    seed: int = 42,
):
    train_texts, train_labels, test_texts, test_labels = load_rotten_tomatoes(
        train_subset=train_subset, seed=seed
    )

    results: dict[str, dict] = {}
    _, _, results["rnn"] = train_sequence_model(
        train_texts, train_labels, test_texts, test_labels,
        cell="rnn", max_len=max_len, epochs=rnn_epochs, seed=seed,
    )
    _, _, results["lstm"] = train_sequence_model(
        train_texts, train_labels, test_texts, test_labels,
        cell="lstm", max_len=max_len, epochs=rnn_epochs, seed=seed,
    )

    bert_info = None
    if not skip_bert:
        from .train_bert import train_bert

        rng = random.Random(seed)
        idx = list(range(len(train_texts)))
        rng.shuffle(idx)
        idx = idx[:bert_subset]
        sub_texts = [train_texts[i] for i in idx]
        sub_labels = [train_labels[i] for i in idx]
        _, _, results["bert"] = train_bert(
            sub_texts, sub_labels, test_texts, test_labels, epochs=bert_epochs, seed=seed
        )
        bert_info = {
            "model": "distilbert-base-uncased",
            "train_examples": len(sub_texts),
            "epochs": bert_epochs,
            "note": "fine-tuned on a CPU-friendly subset; scale up on GPU for the headline number",
        }

    payload = {
        "dataset": "rotten_tomatoes",
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "results": results,
        "bert": bert_info,
    }

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    evaluate.plot_comparison(results, out / "comparison.png")
    from .model_card import generate

    (out / "model_card.md").write_text(generate(payload), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--rnn-epochs", type=int, default=5)
    ap.add_argument("--bert-subset", type=int, default=2000)
    ap.add_argument("--bert-epochs", type=int, default=2)
    ap.add_argument("--skip-bert", action="store_true")
    ap.add_argument("--train-subset", type=int, default=None)
    ap.add_argument("--output", default="reports")
    args = ap.parse_args()

    payload = run(
        max_len=args.max_len,
        rnn_epochs=args.rnn_epochs,
        bert_subset=args.bert_subset,
        bert_epochs=args.bert_epochs,
        skip_bert=args.skip_bert,
        train_subset=args.train_subset,
        output=args.output,
    )
    print(f"\nDataset: {payload['dataset']}  train={payload['train_size']}  test={payload['test_size']}")
    for name, metrics in payload["results"].items():
        print(f"  {name.upper():5s}  acc={metrics['accuracy']:.3f}  macro-F1={metrics['f1']:.3f}")
    print(f"Artifacts -> {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
