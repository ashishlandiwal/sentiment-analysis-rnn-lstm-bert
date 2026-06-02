# Model Card — Sentiment Analysis (RNN vs LSTM vs DistilBERT)

## Overview
- **Task:** binary sentiment classification of movie reviews.
- **Dataset:** rotten_tomatoes (8530 train / 1066 test).
- **Models:** a from-scratch RNN and LSTM (word embeddings) and a fine-tuned DistilBERT transformer.

## Results (test set)

| Model | Accuracy | Precision | Recall | Macro-F1 |
|---|---|---|---|---|
| RNN | 49.5% | 49.5% | 49.5% | 49.5% |
| LSTM | 71.7% | 72.9% | 71.7% | 71.3% |
| BERT | 82.6% | 83.0% | 82.6% | 82.5% |

## DistilBERT fine-tuning
- Base model: `distilbert-base-uncased`
- Fine-tuned on 2000 examples for 2 epoch(s).
- _fine-tuned on a CPU-friendly subset; scale up on GPU for the headline number._

## Intended use & limitations
- Educational comparison of recurrent vs transformer architectures for text classification.
- Trained on short movie-review snippets; will not transfer cleanly to other domains (tweets, product reviews) without retraining.
- The RNN/LSTM use a vocabulary built only from the training set; out-of-vocabulary words map to `<unk>`.

_This card was generated automatically by `sentiment.model_card`._