"""Sentiment analysis comparing RNN, LSTM, and fine-tuned DistilBERT.

Public API:
    load_rotten_tomatoes(...) -> (train_texts, train_labels, test_texts, test_labels)
    Vocab.build(texts)        -> word->id vocabulary for the RNN/LSTM path
    make_model(vocab_size, cell="lstm") -> torch sequence classifier
"""
from .data import load_rotten_tomatoes, tiny_dataset
from .models import make_model
from .vocab import Vocab

__all__ = ["load_rotten_tomatoes", "tiny_dataset", "Vocab", "make_model"]
__version__ = "0.1.0"
