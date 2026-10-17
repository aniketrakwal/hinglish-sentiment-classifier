"""Save the predicted probabilities of every trained model, so later analysis needs no GPU.

    python src/collect_predictions.py

Trains nothing and chooses nothing. It only loads the saved models from earlier steps and writes
results/predictions/<model>_seed<seed>_<split>.csv.
  - VALIDATION: every model (TF-IDF, BiLSTM, LSTM, MuRIL, XLM-R, HingRoBERTa full / LoRA r=8 / LoRA r=16).
  - TEST: only the models that were already evaluated on test in earlier steps
    (TF-IDF, BiLSTM, HingRoBERTa full, HingRoBERTa LoRA r=16).
At the end it recomputes the macro-F1 from the saved files and compares it with results.md.
Files that already exist are skipped (use --force to recompute them).
"""

import os

# The models are already on disk (and in the Hugging Face cache): do not wait for the internet
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from evaluate_lora import predict as predict_lora          # adapter + head on top of HingRoBERTa
from evaluate_winner import predict as predict_full        # a fully fine-tuned transformer
from lstm_data import TweetDataset as LstmDataset, make_loader, tokenize
from lstm_model import BiLSTMClassifier
from prediction_files import (LABELS, MODELS, PRED_DIR, SEEDS, TEST_MODELS, label_ids, load_model_runs,
                              make_prediction_table, prediction_path, probabilities)
from train_lstm import evaluate as lstm_evaluate
from train_transformer import load_split

sys.stdout.reconfigure(encoding="utf-8")

MAX_LENGTH = 64                    # the transformers were trained with this max_length
TRANSFORMER_DIRS = {              # model key -> (run folder pattern in models/)
    "muril": "muril-base-cased_seed{seed}",
    "xlmr": "xlm-roberta-base_seed{seed}",
    "hingroberta_full": "hing-roberta_seed{seed}",
}
LORA_DIRS = {
    "hingroberta_lora_r8": "hing-roberta_lora_r8_lr0.0002_seed{seed}",
    "hingroberta_lora_r16": "hing-roberta_lora_r16_lr0.0002_seed{seed}",
}

# Mean macro-F1 as printed in results.md, to check that the saved probabilities reproduce it
EXPECTED_F1 = {
    ("val", "tfidf"): 0.6226, ("val", "bilstm"): 0.5870, ("val", "lstm_uni"): 0.5850,
    ("val", "muril"): 0.6447, ("val", "xlmr"): 0.6411, ("val", "hingroberta_full"): 0.6796,
    ("val", "hingroberta_lora_r8"): 0.6759, ("val", "hingroberta_lora_r16"): 0.6774,
    ("test", "tfidf"): 0.6849, ("test", "bilstm"): 0.6318, ("test", "hingroberta_full"): 0.7436,
    ("test", "hingroberta_lora_r16"): 0.7515,
}
TOLERANCE = 0.0006                 # results.md shows 4 decimals; allow a little fp16 / rounding noise


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)       # subtract the row maximum for stability
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def save(model, seed, split, df, probs):
    table = make_prediction_table(df["id"], df["label"], probs)
    table.to_csv(prediction_path(model, seed, split), index=False, float_format="%.8f")


def needs_work(model, seed, split, force):
    return force or not os.path.exists(prediction_path(model, seed, split))


# ---------------------------------------------------------------------------------------------
# One function per model family. Each returns the [n, 3] probabilities for one split.
# ---------------------------------------------------------------------------------------------

def tfidf_probabilities(pipeline, df):
    # The pipeline was fitted on cleaned text, and its classes are alphabetical = our label order
    assert list(pipeline.classes_) == LABELS
    return pipeline.predict_proba(df["cleaned_text"])


def lstm_probabilities(model_key, seed, df, word2idx):
    """Probabilities of a saved BiLSTM / LSTM checkpoint (small models: the CPU is fast enough)."""
    config = "bilstm" if model_key == "bilstm" else "lstm"          # names used when the checkpoints were saved
    model = BiLSTMClassifier(vocab_size=len(word2idx), bidirectional=(config == "bilstm"))
    model.load_state_dict(torch.load(os.path.join("models", f"lstm_{config}_seed{seed}.pt")))
    tokens = df["text"].apply(tokenize)                              # clean_text() + split, as in training
    loader = make_loader(LstmDataset(tokens, df["label"], word2idx), shuffle=False)
    _, _, probs, _ = lstm_evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    return probs


def main():
    parser = argparse.ArgumentParser(description="Save predicted probabilities of all models.")
    parser.add_argument("--force", action="store_true", help="recompute files that already exist")
    args = parser.parse_args()
    os.makedirs(PRED_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    splits = {"val": load_split("data/processed/val.csv"),            # adds the cleaned_text column
              "test": load_split("data/processed/test.csv")}
    jobs = [("val", key, seed) for key, (_, seeds) in MODELS.items() for seed in seeds]
    jobs += [("test", key, seed) for key in TEST_MODELS for seed in MODELS[key][1]]

    pipeline = joblib.load(os.path.join("models", "tfidf_logreg.joblib"))
    with open(os.path.join("models", "lstm_vocab.json"), encoding="utf-8") as f:
        word2idx = json.load(f)

    for split, key, seed in jobs:
        if not needs_work(key, seed, split, args.force):
            print(f"{split:4s} {key} seed {seed}: file exists, skipping")
            continue
        df = splits[split]
        if key == "tfidf":
            probs = tfidf_probabilities(pipeline, df)
        elif key in ("bilstm", "lstm_uni"):
            probs = lstm_probabilities(key, seed, df, word2idx)
        elif key in TRANSFORMER_DIRS:
            logits, _ = predict_full(os.path.join("models", TRANSFORMER_DIRS[key].format(seed=seed)), df, MAX_LENGTH, device)
            probs = softmax(logits)
        else:
            logits, _ = predict_lora(os.path.join("models", LORA_DIRS[key].format(seed=seed)), df, device)
            probs = softmax(logits)
        save(key, seed, split, df, probs)
        print(f"{split:4s} {key} seed {seed}: saved {len(df)} rows", flush=True)

    # ---- Check: recompute the macro-F1 from the saved files and compare with results.md
    print("\nmacro-F1 recomputed from the saved files vs results.md (mean over seeds)")
    all_ok = True
    for (split, key), expected in EXPECTED_F1.items():
        values = []
        for table in load_model_runs(key, split):
            values.append(f1_score(label_ids(table["true_label"]), label_ids(table["predicted_label"]),
                                   labels=[0, 1, 2], average="macro"))
        mean = float(np.mean(values))
        ok = abs(mean - expected) <= TOLERANCE
        all_ok = all_ok and ok
        print(f"  {split:4s} {MODELS[key][0]:24s} recomputed {mean:.4f}   results.md {expected:.4f}   "
              f"{'OK' if ok else 'MISMATCH'}")
    # The probabilities must also sum to 1 and give the saved predicted label
    for table in load_model_runs("hingroberta_full", "val"):
        assert np.allclose(probabilities(table).sum(axis=1), 1.0, atol=1e-5)
    print("\nALL MATCH results.md" if all_ok else "\nSOME VALUES DO NOT MATCH results.md - look above")


if __name__ == "__main__":
    main()
