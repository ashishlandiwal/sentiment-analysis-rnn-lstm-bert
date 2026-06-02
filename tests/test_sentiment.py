import torch

from sentiment.data import tiny_dataset
from sentiment.models import make_model
from sentiment.train_rnn import train_sequence_model
from sentiment.vocab import Vocab


def test_vocab_encode_pads_and_handles_unknown():
    vocab = Vocab.build(["good great film", "bad awful movie"], min_freq=1)
    ids = vocab.encode("good unknownword", max_len=5)
    assert len(ids) == 5
    assert ids[0] == vocab.stoi["good"]
    assert ids[1] == vocab.unk_id           # OOV -> <unk>
    assert ids[2:] == [vocab.pad_id] * 3    # padded


def test_model_forward_shapes():
    for cell in ("rnn", "lstm"):
        model = make_model(vocab_size=50, cell=cell, embed_dim=16, hidden_dim=16)
        logits = model(torch.randint(0, 50, (4, 12)))
        assert logits.shape == (4, 2)


def test_tiny_training_learns_separable_data():
    tr_t, tr_y, te_t, te_y = tiny_dataset()
    _, _, metrics = train_sequence_model(
        tr_t, tr_y, te_t, te_y, cell="lstm", max_len=12, epochs=20, min_freq=1, seed=0
    )
    # the toy corpus is linearly separable; the model should clearly beat chance
    assert metrics["accuracy"] >= 0.75
