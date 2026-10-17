"""Tests for the LoRA part of src/train_transformer.py and src/run_lora.py.

Uses a tiny randomly initialised RoBERTa (no download), with the same shape of problem as the real
model: a big word-embedding table, a few layers, a small 3-class head.
"""

import json
import os
import sys

import pytest
from transformers import RobertaConfig, RobertaForSequenceClassification

# Make `src/` importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_transformer import apply_lora, weights_size_mb  # noqa: E402


@pytest.fixture
def tiny_model():
    config = RobertaConfig(vocab_size=10000, hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
                           intermediate_size=512, num_labels=3)
    return RobertaForSequenceClassification(config)


def test_lora_trains_less_than_2_percent_and_includes_the_classifier(tiny_model):
    model = apply_lora(tiny_model, r=8)
    total = sum(p.numel() for p in model.parameters())
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    assert trainable / total < 0.02
    assert any("classifier" in name for name in trainable_names)      # the new head is trained
    assert any("lora_A" in name for name in trainable_names) and any("lora_B" in name for name in trainable_names)


def test_lora_only_touches_query_and_value_and_freezes_the_rest(tiny_model):
    model = apply_lora(tiny_model, r=8)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert "classifier" in name or ("lora_" in name and (".query." in name or ".value." in name))
    embedding = model.get_input_embeddings()
    assert all(not p.requires_grad for p in embedding.parameters())


def test_lora_alpha_is_twice_the_rank(tiny_model):
    model = apply_lora(tiny_model, r=16)
    config = model.peft_config["default"]
    assert config.r == 16 and config.lora_alpha == 32 and config.lora_dropout == 0.1


def test_larger_rank_has_more_trainable_parameters(tiny_model):
    small = apply_lora(tiny_model, r=4)
    n_small = sum(p.numel() for p in small.parameters() if p.requires_grad)
    other = RobertaForSequenceClassification(tiny_model.config)
    n_big = sum(p.numel() for p in apply_lora(other, r=16).parameters() if p.requires_grad)
    assert n_big > n_small


def test_weights_size_counts_only_safetensors_files(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1024 * 1024)
    (tmp_path / "tokenizer.json").write_bytes(b"y" * 5 * 1024 * 1024)
    assert weights_size_mb(str(tmp_path)) == pytest.approx(1.0)


def test_run_lora_skips_only_runs_with_the_same_settings(tmp_path, monkeypatch):
    import run_lora as rl
    monkeypatch.setattr(rl, "RESULTS_DIR", str(tmp_path))
    path = tmp_path / "hing-roberta_lora_r8_lr0.0002_seed42_val_summary.json"
    good = {"lora": True, "r": 8, "learning_rate": 2e-4, "max_length": 64, "num_epochs": 6}
    path.write_text(json.dumps(good), encoding="utf-8")
    assert rl.already_done(8, 2e-4, 42) is True
    assert rl.already_done(8, 2e-4, 43) is False                  # no file at all
    path.write_text(json.dumps({**good, "num_epochs": 4}), encoding="utf-8")
    assert rl.already_done(8, 2e-4, 42) is False                  # made with other settings: must be re-run


def test_run_name_matches_the_name_used_by_train_transformer():
    import run_lora as rl
    assert rl.run_name(8, 2e-4, 42) == "hing-roberta_lora_r8_lr0.0002_seed42"
    assert rl.run_name(16, 5e-4, 44) == "hing-roberta_lora_r16_lr0.0005_seed44"
