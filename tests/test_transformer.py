"""Tests for src/train_transformer.py: dataset output, label mapping, truncation.

Uses a tiny tokenizer built in memory, so no model has to be downloaded to run the tests.
"""

import os
import sys

import pytest
from transformers import BertTokenizerFast

# Make `src/` importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_transformer import ID2LABEL, LABEL2ID, LABELS, TweetDataset, compute_metrics  # noqa: E402

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "nahi", "bahut", "accha", "hai", "film", "ok"]


@pytest.fixture
def tokenizer(tmp_path):
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB), encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(vocab_file))


def test_label_mapping_is_negative_neutral_positive():
    assert LABELS == ["negative", "neutral", "positive"]
    assert LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}
    assert ID2LABEL == {0: "negative", 1: "neutral", 2: "positive"}


def test_dataset_returns_input_ids_and_labels(tokenizer):
    dataset = TweetDataset(["bahut accha hai", "nahi ok"], ["positive", "negative"], tokenizer, max_length=48)
    assert len(dataset) == 2
    item = dataset[0]
    assert "input_ids" in item and "labels" in item and "attention_mask" in item
    assert item["labels"] == 2 and dataset[1]["labels"] == 0
    assert item["input_ids"][0] == tokenizer.cls_token_id and item["input_ids"][-1] == tokenizer.sep_token_id


def test_unpadded_items_have_their_own_length(tokenizer):
    dataset = TweetDataset(["ok", "bahut accha hai film"], ["neutral", "neutral"], tokenizer, max_length=48)
    assert len(dataset[0]["input_ids"]) == 3      # [CLS] ok [SEP]  (padding is added later, per batch)
    assert len(dataset[1]["input_ids"]) == 6


def test_truncation_respects_max_length(tokenizer):
    long_text = " ".join(["bahut"] * 100)
    dataset = TweetDataset([long_text, "ok"], ["neutral", "neutral"], tokenizer, max_length=10)
    assert len(dataset[0]["input_ids"]) == 10
    assert dataset[0]["input_ids"][-1] == tokenizer.sep_token_id      # [SEP] is kept after truncation
    assert len(dataset[1]["input_ids"]) == 3


def test_compute_metrics_perfect_and_wrong_predictions():
    import numpy as np
    labels = np.array([0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[labels]                          # logits that point at the right class
    metrics = compute_metrics((perfect, labels))
    assert metrics["accuracy"] == 1.0 and metrics["macro_f1"] == 1.0
    rotated = np.eye(3)[(labels + 1) % 3]                # every prediction wrong
    assert compute_metrics((rotated, labels))["accuracy"] == 0.0


# ---------------------------------------------------------------- helpers of compare_transformers.py
def test_predictions_from_errors_fills_in_correct_predictions(tmp_path):
    import pandas as pd
    from compare_transformers import predictions_from_errors
    val = pd.DataFrame({"id": ["1", "2", "3"], "label": ["negative", "neutral", "positive"]})
    errors = tmp_path / "errors.csv"
    errors.write_text("id,predicted_label\n2,positive\n", encoding="utf-8")
    # only tweet 2 is a mistake; tweets 1 and 3 were predicted correctly
    assert predictions_from_errors(val, str(errors)) == ["negative", "positive", "positive"]


def test_best_epoch_is_the_epoch_with_highest_val_macro_f1():
    from compare_transformers import best_epoch
    summary = {"epochs": [{"epoch": 1, "val_macro_f1": 0.5}, {"epoch": 2, "val_macro_f1": 0.7},
                          {"epoch": 3, "val_macro_f1": 0.6}]}
    assert best_epoch(summary) == 2


def test_mean_std_formats_mean_and_sample_std():
    from compare_transformers import mean_std
    assert mean_std([0.5, 0.6, 0.7]) == "0.6000 ± 0.1000"


def test_run_transformers_skips_only_runs_made_with_the_same_max_length(tmp_path, monkeypatch):
    import json
    import run_transformers as rt
    monkeypatch.setattr(rt, "RESULTS_DIR", str(tmp_path))
    model = "google/muril-base-cased"
    path = tmp_path / "muril-base-cased_seed42_val_summary.json"
    path.write_text(json.dumps({"model": model, "max_length": 48}), encoding="utf-8")
    assert rt.already_done(model, 42) is False          # an old 48-token result must be re-run
    path.write_text(json.dumps({"model": model, "max_length": 64}), encoding="utf-8")
    assert rt.already_done(model, 42) is True
    assert rt.already_done(model, 43) is False          # no file at all
