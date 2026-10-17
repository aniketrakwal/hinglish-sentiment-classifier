"""Where the saved predictions live and how to read them (pandas + numpy only, no torch).

src/collect_predictions.py WRITES these files; bootstrap.py, calibration.py and val_test_gap.py READ them.
One CSV per (model, seed, split), columns: id, true_label, prob_negative, prob_neutral, prob_positive,
predicted_label. The rows are in the same order as data/processed/val.csv or test.csv.
"""

import os

import numpy as np
import pandas as pd

PRED_DIR = os.path.join("results", "predictions")
LABELS = ["negative", "neutral", "positive"]          # class ids 0, 1, 2 (the same order everywhere)
PROB_COLUMNS = ["prob_" + label for label in LABELS]
SEEDS = [42, 43, 44]

# key -> (display name, seeds). TF-IDF is deterministic, so it has no seed (None).
MODELS = {
    "tfidf": ("TF-IDF + LogReg", [None]),
    "bilstm": ("BiLSTM", SEEDS),
    "lstm_uni": ("LSTM (uni)", SEEDS),
    "muril": ("MuRIL", SEEDS),
    "xlmr": ("XLM-R", SEEDS),
    "hingroberta_full": ("HingRoBERTa (full)", SEEDS),
    "hingroberta_lora_r8": ("HingRoBERTa LoRA r=8", SEEDS),
    "hingroberta_lora_r16": ("HingRoBERTa LoRA r=16", SEEDS),
}
# Models that were already evaluated on the test set in earlier steps (and ONLY these may have test files)
TEST_MODELS = ["tfidf", "bilstm", "hingroberta_full", "hingroberta_lora_r16"]


def prediction_path(model, seed, split):
    seed_part = "" if seed is None else f"_seed{seed}"
    return os.path.join(PRED_DIR, f"{model}{seed_part}_{split}.csv")


def make_prediction_table(ids, true_labels, probs):
    """Build the table that is saved: id, true label, the 3 probabilities, predicted label."""
    table = pd.DataFrame({"id": list(ids), "true_label": list(true_labels)})
    for i, column in enumerate(PROB_COLUMNS):
        table[column] = probs[:, i]
    table["predicted_label"] = [LABELS[i] for i in probs.argmax(axis=1)]
    return table


def load_run(model, seed, split):
    """One saved prediction table (a pandas DataFrame)."""
    return pd.read_csv(prediction_path(model, seed, split), dtype={"id": str})


def load_model_runs(model, split):
    """The tables of all seeds of a model, as a list (one table for TF-IDF)."""
    return [load_run(model, seed, split) for seed in MODELS[model][1]]


def label_ids(labels):
    """Label strings -> ids 0/1/2."""
    return np.array([LABELS.index(label) for label in labels])


def probabilities(table):
    """The [n, 3] probability matrix of a prediction table."""
    return table[PROB_COLUMNS].to_numpy()
