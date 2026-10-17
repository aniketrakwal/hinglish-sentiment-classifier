"""Evaluate the CHOSEN LoRA setup (3 seed checkpoints) on the TEST set, once.

    python src/evaluate_lora.py

It refuses to run unless results/lora_choice.txt exists, i.e. unless the setup was chosen on
validation first (src/compare_lora.py). Nothing else is evaluated on test here.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from train_transformer import ID2LABEL, LABEL2ID, TweetDataset, compute_metrics, load_split

sys.stdout.reconfigure(encoding="utf-8")

BASE_MODEL = "l3cube-pune/hing-roberta"
SEEDS = [42, 43, 44]
RESULTS = "results"
MAX_LENGTH = 64
FULL_TEST_F1 = 0.7436          # full fine-tune of HingRoBERTa on test (results.md), for comparison only


def read_choice():
    path = os.path.join(RESULTS, "lora_choice.txt")
    if not os.path.exists(path):
        sys.exit("results/lora_choice.txt not found. Run src/compare_lora.py first: "
                 "the LoRA setup must be chosen on validation BEFORE looking at test.")
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(("R:", "LEARNING_RATE:")):
                key, value = line.split(":", 1)
                values[key] = value.strip()
    return int(values["R"]), float(values["LEARNING_RATE"])


def predict(adapter_dir, test_df, device):
    """Pretrained HingRoBERTa + the saved adapter and classification head -> logits for the test tweets."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()    # puts the saved A, B and head in
    dataset = TweetDataset(test_df["cleaned_text"], test_df["label"], tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))
    all_logits = []
    with torch.no_grad():                                        # no learning, just predicting
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast("cuda", dtype=torch.float16):    # half precision, like in training
                all_logits.append(model(**batch).logits.float().cpu())
    del model, base
    torch.cuda.empty_cache()                                     # free the GPU before the next checkpoint
    return torch.cat(all_logits).numpy(), np.array(dataset.labels)


def main():
    r, lr = read_choice()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LoRA setup chosen on validation: r={r}, lr {lr:g}. Evaluating its {len(SEEDS)} checkpoints on TEST (once each).")

    test_df = load_split("data/processed/test.csv")               # the first time test is opened in this step
    rows = []
    for seed in SEEDS:
        run = f"hing-roberta_lora_r{r}_lr{lr:g}_seed{seed}"
        logits, labels = predict(os.path.join("models", run), test_df, device)
        metrics = compute_metrics((logits, labels))
        rows.append({"model": f"hing-roberta LoRA r={r}", "seed": seed,
                     **{"test_" + k: float(v) for k, v in metrics.items()}})
        print(f"  seed {seed}: test accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "lora_test.csv"), index=False, float_format="%.4f")
    print("\n" + table.round(4).to_string(index=False))
    f1 = table["test_macro_f1"]
    print(f"\nTEST macro-F1 {f1.mean():.4f} ± {f1.std(ddof=1):.4f}   "
          f"accuracy {table['test_accuracy'].mean():.4f} ± {table['test_accuracy'].std(ddof=1):.4f}")
    print(f"per-class test F1: negative {table['test_f1_negative'].mean():.4f}, "
          f"neutral {table['test_f1_neutral'].mean():.4f}, positive {table['test_f1_positive'].mean():.4f}")
    print(f"vs full fine-tune test {FULL_TEST_F1}: {f1.mean() - FULL_TEST_F1:+.4f}")


if __name__ == "__main__":
    main()
