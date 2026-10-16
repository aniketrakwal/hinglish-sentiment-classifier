"""Tests for the BiLSTM pieces (src/lstm_data.py, src/lstm_model.py, src/train_lstm.py)."""

import os
import sys

import torch
from torch import nn

# Make `src/` importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lstm_data import MAX_LEN, PAD_ID, UNK_ID, TweetDataset, build_vocab, encode, tokenize  # noqa: E402
from lstm_model import BiLSTMClassifier  # noqa: E402
from train_lstm import set_seed, train_step  # noqa: E402

TRAIN_TOKENS = [["bahut", "accha", "hai"], ["bahut", "bura", "hai"], ["film", "accha", "hai"]]


def test_tokenize_uses_clean_text_and_splits_on_spaces():
    assert tokenize("RT @ Nehr _ who SOOOO good https // t . co / AbC123xyz") == ["@user", "soo", "good", "http"]
    assert tokenize("") == []


def test_vocab_reserves_pad_and_unk_and_drops_rare_words():
    vocab = build_vocab(TRAIN_TOKENS, min_freq=2)
    assert vocab["<pad>"] == 0 and vocab["<unk>"] == 1
    assert "bahut" in vocab and "accha" in vocab and "hai" in vocab    # each appears 2+ times
    assert "bura" not in vocab and "film" not in vocab                 # appear once only


def test_unknown_words_map_to_1_and_padding_is_0():
    vocab = build_vocab(TRAIN_TOKENS, min_freq=2)
    ids, length = encode(["bahut", "neverseen", "hai"], vocab, max_len=6)
    assert ids[1] == UNK_ID == 1
    assert ids[3:] == [PAD_ID] * 3 and PAD_ID == 0
    assert length == 3


def test_encoded_length_is_always_40():
    vocab = build_vocab(TRAIN_TOKENS, min_freq=2)
    for tokens in [[], ["hai"], ["hai"] * 40, ["hai"] * 100]:
        ids, length = encode(tokens, vocab)
        assert len(ids) == MAX_LEN == 40
        assert 1 <= length <= 40
    dataset = TweetDataset(TRAIN_TOKENS, ["negative", "neutral", "positive"], vocab)
    assert dataset.ids.shape == (3, 40)


def test_model_output_shape_is_batch_by_3():
    ids = torch.randint(1, 50, (5, 40))
    lengths = torch.tensor([40, 10, 3, 1, 25])
    for bidirectional in (True, False):
        model = BiLSTMClassifier(vocab_size=50, bidirectional=bidirectional)
        assert model(ids, lengths).shape == (5, 3)


def test_padding_does_not_change_the_output():
    set_seed(0)
    model = BiLSTMClassifier(vocab_size=50).eval()
    short = torch.tensor([[5, 6, 7] + [0] * 37])
    changed_padding = short.clone()
    changed_padding[0, 10:] = 9                      # junk after the real length
    lengths = torch.tensor([3])
    assert torch.allclose(model(short, lengths), model(changed_padding, lengths), atol=1e-6)


def test_one_training_step_reduces_the_loss():
    set_seed(0)
    model = BiLSTMClassifier(vocab_size=50)
    ids = torch.randint(2, 50, (8, 40))
    lengths = torch.randint(5, 41, (8,))
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.eval()                                      # measure the loss without dropout noise
    with torch.no_grad():
        before = loss_fn(model(ids, lengths), labels).item()
    train_step(model, (ids, lengths, labels), loss_fn, optimizer, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        after = loss_fn(model(ids, lengths), labels).item()
    assert after < before
