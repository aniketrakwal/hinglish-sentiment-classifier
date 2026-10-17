"""Compare LoRA (r=8, r=16) with the full fine-tune of HingRoBERTa, VALIDATION only.

    python src/compare_lora.py

Reads the summary JSON files of the 3 seeds of every setup, writes
  results/lora_comparison.csv   one row per setup (mean +- std over seeds, parameters, time, memory, size)
  results/lora_per_seed.csv     LoRA minus full fine-tune, seed by seed
  results/lora_choice.txt       the best LoRA setup (by mean val macro-F1), written BEFORE test is used
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from train_transformer import weights_size_mb

sys.stdout.reconfigure(encoding="utf-8")

RESULTS = "results"
MODELS = "models"
SEEDS = [42, 43, 44]
RANKS = [8, 16]
FULL_NAME = "hing-roberta"


def read_lr():
    """The learning rate chosen in the check of src/run_lora.py (validation, r=8, seed 42)."""
    with open(os.path.join(RESULTS, "lora_lr_choice.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("LEARNING_RATE:"):
                return float(line.split(":", 1)[1])
    sys.exit("no LEARNING_RATE line in results/lora_lr_choice.txt")


def load_summaries(run_names):
    summaries = []
    for name in run_names:
        with open(os.path.join(RESULTS, f"{name}_val_summary.json"), encoding="utf-8") as f:
            summaries.append(json.load(f))
    return summaries


def mean_std(values):
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"       # sample std over the 3 seeds


def describe(label, summaries, run_names):
    """One row of the comparison table."""
    f1 = [s["best_val"]["macro_f1"] for s in summaries]
    epoch_seconds = [np.mean([e["epoch_time_s"] for e in s["epochs"]]) for s in summaries]
    sizes = [weights_size_mb(os.path.join(MODELS, name)) for name in run_names]
    total, trainable = summaries[0]["total_params"], summaries[0]["trainable_params"]
    return {
        "setup": label,
        "val_macro_f1": mean_std(f1),
        "val_macro_f1_mean": round(float(np.mean(f1)), 4),
        "f1_negative": round(float(np.mean([s["best_val"]["f1_negative"] for s in summaries])), 4),
        "f1_neutral": round(float(np.mean([s["best_val"]["f1_neutral"] for s in summaries])), 4),
        "f1_positive": round(float(np.mean([s["best_val"]["f1_positive"] for s in summaries])), 4),
        "total_params": total, "trainable_params": trainable,
        "trainable_percent": round(100 * trainable / total, 2),
        "sec_per_epoch": round(float(np.mean(epoch_seconds)), 1),
        "peak_gpu_gib": round(float(np.mean([s["peak_gpu_allocated_gib"] for s in summaries])), 2),
        "saved_weights_mb": round(float(np.mean(sizes)), 1),
    }, f1


def main():
    lr = read_lr()
    full_runs = [f"{FULL_NAME}_seed{seed}" for seed in SEEDS]
    setups = {"full fine-tune": full_runs}
    for r in RANKS:
        setups[f"LoRA r={r}"] = [f"{FULL_NAME}_lora_r{r}_lr{lr:g}_seed{seed}" for seed in SEEDS]

    rows, f1_by_setup = [], {}
    for label, run_names in setups.items():
        row, f1 = describe(label, load_summaries(run_names), run_names)
        rows.append(row)
        f1_by_setup[label] = f1
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "lora_comparison.csv"), index=False)
    print(f"LoRA learning rate: {lr:g}, {6} epochs (full fine-tune: 2e-5, 4 epochs)\n")
    print(table.drop(columns="val_macro_f1_mean").to_string(index=False))

    # ---- Seed by seed: LoRA minus full fine-tune (same seed = same data order and classifier start)
    diffs = pd.DataFrame({"seed": SEEDS, "full_f1": f1_by_setup["full fine-tune"]})
    for r in RANKS:
        diffs[f"lora_r{r}_f1"] = f1_by_setup[f"LoRA r={r}"]
        diffs[f"lora_r{r}_minus_full"] = np.array(f1_by_setup[f"LoRA r={r}"]) - np.array(diffs["full_f1"])
    diffs.to_csv(os.path.join(RESULTS, "lora_per_seed.csv"), index=False, float_format="%.4f")
    print("\nper seed (macro-F1):")
    print(diffs.round(4).to_string(index=False))
    for r in RANKS:
        d = diffs[f"lora_r{r}_minus_full"]
        print(f"LoRA r={r} minus full: mean {d.mean():+.4f}, per seed {[round(float(x), 4) for x in d]}")
    d16 = np.array(f1_by_setup["LoRA r=16"]) - np.array(f1_by_setup["LoRA r=8"])
    print(f"LoRA r=16 minus r=8: mean {d16.mean():+.4f}, per seed {[round(float(x), 4) for x in d16]}")
    print(f"seed noise (std over seeds): full {np.std(f1_by_setup['full fine-tune'], ddof=1):.4f}, "
          f"r=8 {np.std(f1_by_setup['LoRA r=8'], ddof=1):.4f}, r=16 {np.std(f1_by_setup['LoRA r=16'], ddof=1):.4f}")

    # ---- Choose the best LoRA setup on VALIDATION and write it down before test is touched
    lora_rows = table[table["setup"].str.startswith("LoRA")]
    best = lora_rows.loc[lora_rows["val_macro_f1_mean"].idxmax()]
    best_r = int(best["setup"].split("=")[1])
    with open(os.path.join(RESULTS, "lora_choice.txt"), "w", encoding="utf-8") as f:
        f.write("Best LoRA setup, chosen by mean validation macro-F1 (test not used yet):\n")
        for _, row in lora_rows.iterrows():
            f.write(f"  {row['setup']}: val macro-F1 {row['val_macro_f1']}\n")
        f.write(f"R: {best_r}\n")
        f.write(f"LEARNING_RATE: {lr:g}\n")
    print(f"\nchosen LoRA setup: r={best_r}, lr {lr:g}  (written to results/lora_choice.txt)")


if __name__ == "__main__":
    main()
