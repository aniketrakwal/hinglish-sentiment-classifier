"""Evaluate the WINNING transformer (3 seed checkpoints) on the TEST set, once.

    python src/evaluate_winner.py

It refuses to run unless results/transformer_choice.txt exists, i.e. unless the winner was chosen on
validation first (src/compare_transformers.py). Nothing else is evaluated on test here.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from train_transformer import LABELS, TweetDataset, compute_metrics, load_split

sys.stdout.reconfigure(encoding="utf-8")

SEEDS = [42, 43, 44]
RESULTS = "results"
# Test macro-F1 of the earlier models (from results.md), for comparison only
TFIDF_TEST_F1, BILSTM_TEST_F1 = 0.6849, 0.6318


def read_winner():
    path = os.path.join(RESULTS, "transformer_choice.txt")
    if not os.path.exists(path):
        sys.exit("results/transformer_choice.txt not found. Run src/compare_transformers.py first: "
                 "the winner must be chosen on validation BEFORE looking at test.")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("WINNER:"):
                return line.split(":", 1)[1].strip()
    sys.exit("no 'WINNER:' line in results/transformer_choice.txt")


def predict(model_dir, test_df, max_length, device):
    """Load a saved checkpoint and return its predicted label ids for the test tweets."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    dataset = TweetDataset(test_df["cleaned_text"], test_df["label"], tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
    all_logits = []
    with torch.no_grad():                                        # no learning, just predicting
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast("cuda", dtype=torch.float16):    # half precision, like in training
                all_logits.append(model(**batch).logits.float().cpu())
    del model
    torch.cuda.empty_cache()                                     # free the GPU before the next checkpoint
    return torch.cat(all_logits).numpy(), np.array(dataset.labels)


def main():
    winner = read_winner()
    name = winner.split("/")[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Winner chosen on validation: {winner}. Evaluating its {len(SEEDS)} checkpoints on TEST (once each).")

    test_df = load_split("data/processed/test.csv")               # the first time test is opened in this step
    rows = []
    for seed in SEEDS:
        run = f"{name}_seed{seed}"
        with open(os.path.join(RESULTS, f"{run}_val_summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        logits, labels = predict(os.path.join("models", run), test_df, summary["max_length"], device)
        metrics = compute_metrics((logits, labels))
        rows.append({"model": name, "seed": seed, "val_macro_f1": summary["best_val"]["macro_f1"],
                     **{"test_" + k: float(v) for k, v in metrics.items()}})
        print(f"  seed {seed}: test accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "transformer_test.csv"), index=False, float_format="%.4f")
    print("\n" + table.round(4).to_string(index=False))
    f1 = table["test_macro_f1"]
    print(f"\nTEST macro-F1 {f1.mean():.4f} ± {f1.std(ddof=1):.4f}   "
          f"accuracy {table['test_accuracy'].mean():.4f} ± {table['test_accuracy'].std(ddof=1):.4f}")
    print(f"per-class test F1: negative {table['test_f1_negative'].mean():.4f}, "
          f"neutral {table['test_f1_neutral'].mean():.4f}, positive {table['test_f1_positive'].mean():.4f}")
    print(f"vs TF-IDF test {TFIDF_TEST_F1}: {f1.mean() - TFIDF_TEST_F1:+.4f}   vs BiLSTM test {BILSTM_TEST_F1}: "
          f"{f1.mean() - BILSTM_TEST_F1:+.4f}")
    print(f"mean val macro-F1 of these checkpoints: {table['val_macro_f1'].mean():.4f} "
          f"(test minus val: {f1.mean() - table['val_macro_f1'].mean():+.4f})")


if __name__ == "__main__":
    main()
