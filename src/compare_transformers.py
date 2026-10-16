"""Compare the three transformers on VALIDATION, choose the winner, and analyse the errors.

    python src/compare_transformers.py

Reads the files made by run_transformers.py. It never touches the test set. The decision is written to
results/transformer_choice.txt; only AFTER that file exists may src/evaluate_winner.py look at test.

Saved: results/transformer_comparison.csv  (one row per model),
       results/transformer_per_seed.csv    (per-seed macro-F1 and the HingRoBERTa - XLM-R difference),
       results/transformer_emoji.csv       (macro-F1 on tweets with / without emoji),
       results/transformer_choice.txt      (the decision).
"""

import json
import os
import sys
from datetime import datetime

import emoji
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from preprocess import clean_text

sys.stdout.reconfigure(encoding="utf-8")   # print Devanagari / emoji safely on Windows

MODELS = ["google/muril-base-cased", "l3cube-pune/hing-roberta", "xlm-roberta-base"]
SHORT = {m: m.split("/")[-1] for m in MODELS}
SEEDS = [42, 43, 44]
LABELS = ["negative", "neutral", "positive"]
MAX_LENGTH = 64
TFIDF_VAL_F1 = 0.6226            # from results.md (TF-IDF + LogReg, validation macro-F1)
RESULTS = "results"


def load_summaries():
    """{(model, seed): summary dict} for every finished run made with the current max_length."""
    summaries = {}
    for model in MODELS:
        for seed in SEEDS:
            path = os.path.join(RESULTS, f"{SHORT[model]}_seed{seed}_val_summary.json")
            if not os.path.exists(path):
                print(f"WARNING: missing {path}")
                continue
            with open(path, encoding="utf-8") as f:
                summary = json.load(f)
            if summary["max_length"] != MAX_LENGTH:
                print(f"WARNING: {path} was made with max_length {summary['max_length']}, ignoring it")
                continue
            summaries[(model, seed)] = summary
    return summaries


def mean_std(values, digits=4):
    values = list(values)
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1) if len(values) > 1 else 0:.{digits}f}"


def best_epoch(summary):
    """The epoch with the highest validation macro-F1 (the one load_best_model_at_end keeps)."""
    return max(summary["epochs"], key=lambda e: e["val_macro_f1"])["epoch"]


def predictions_from_errors(val_df, errors_path):
    """Rebuild the prediction for EVERY val tweet from an errors file (which lists only the mistakes):
    a tweet that is not in the file was predicted correctly, so its prediction equals its true label."""
    errors = pd.read_csv(errors_path, dtype=str, keep_default_na=False)
    wrong = dict(zip(errors["id"], errors["predicted_label"]))
    return [wrong.get(tweet_id, label) for tweet_id, label in zip(val_df["id"], val_df["label"])]


def macro_f1(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels, labels=LABELS, average="macro")


def main():
    summaries = load_summaries()
    seeds_for = {m: [s for s in SEEDS if (m, s) in summaries] for m in MODELS}
    models = [m for m in MODELS if seeds_for[m]]

    # ------------------------------------------------------------------ 1. per-model table
    rows = []
    for m in models:
        runs = [summaries[(m, s)] for s in seeds_for[m]]
        epoch_times = [e["epoch_time_s"] for r in runs for e in r["epochs"]]
        rows.append({
            "model": SHORT[m],
            "seeds": len(runs),
            "val_accuracy": mean_std(r["best_val"]["accuracy"] for r in runs),
            "val_macro_f1": mean_std(r["best_val"]["macro_f1"] for r in runs),
            "f1_negative": mean_std(r["best_val"]["f1_negative"] for r in runs),
            "f1_neutral": mean_std(r["best_val"]["f1_neutral"] for r in runs),
            "f1_positive": mean_std(r["best_val"]["f1_positive"] for r in runs),
            "best_epoch_per_seed": "/".join(str(best_epoch(r)) for r in runs),
            "sec_per_epoch": round(float(np.mean(epoch_times)), 1),
            "peak_gpu_gib": round(max(r["peak_gpu_allocated_gib"] for r in runs), 2),
            "total_params_M": round(runs[0]["total_params"] / 1e6, 1),
            "trainable_params_M": round(runs[0]["trainable_params"] / 1e6, 1),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(os.path.join(RESULTS, "transformer_comparison.csv"), index=False)
    print("=== 1. Per model, validation (mean ± std over seeds; std uses n-1) ===")
    print(comparison.to_string(index=False))

    # ------------------------------------------------------------------ 2. per-seed table
    per_seed = pd.DataFrame(
        {SHORT[m]: {s: summaries[(m, s)]["best_val"]["macro_f1"] for s in seeds_for[m]} for m in models})
    per_seed.index.name = "seed"
    print("\n=== 2. Validation macro-F1 per seed ===")
    print(per_seed.round(4).to_string())
    print(f"\nBeats TF-IDF ({TFIDF_VAL_F1}) on every seed?")
    for m in models:
        wins = [bool(per_seed[SHORT[m]][s] > TFIDF_VAL_F1) for s in seeds_for[m]]
        print(f"  {SHORT[m]:26s} {'YES' if all(wins) else 'NO '}  ({sum(wins)} of {len(wins)} seeds)")

    # ------------------------------------------------------------------ 3. HingRoBERTa - XLM-R
    hing, xlmr = "l3cube-pune/hing-roberta", "xlm-roberta-base"
    if hing in models and xlmr in models:
        common = [s for s in SEEDS if s in seeds_for[hing] and s in seeds_for[xlmr]]
        diff = pd.Series({s: summaries[(hing, s)]["best_val"]["macro_f1"] - summaries[(xlmr, s)]["best_val"]["macro_f1"]
                          for s in common})
        per_seed["hing_minus_xlmr"] = diff
        print("\n=== 3. HingRoBERTa minus XLM-R (value of Hinglish pre-training), macro-F1 ===")
        print(diff.round(4).to_string())
        print(f"mean difference {diff.mean():+.4f}   (positive on {int((diff > 0).sum())} of {len(diff)} seeds)")
    per_seed.round(4).to_csv(os.path.join(RESULTS, "transformer_per_seed.csv"))

    # ------------------------------------------------------------------ 4. emoji / no emoji
    val = pd.read_csv("data/processed/val.csv", dtype=str, keep_default_na=False)
    val["cleaned"] = val["text"].apply(clean_text)
    has_emoji = val["cleaned"].apply(lambda t: len(emoji.emoji_list(t)) > 0).to_numpy()
    print(f"\n=== 4. Macro-F1 on val tweets with / without emoji "
          f"({has_emoji.sum()} with, {(~has_emoji).sum()} without; mean over seeds) ===")
    true = val["label"].to_numpy()
    emoji_rows = []
    systems = {SHORT[m]: [os.path.join(RESULTS, f"{SHORT[m]}_seed{s}_val_errors.csv") for s in seeds_for[m]] for m in models}
    systems["TF-IDF + LogReg"] = [os.path.join(RESULTS, "ml_val_errors.csv")]
    systems["BiLSTM (best seed)"] = [os.path.join(RESULTS, "lstm_val_errors.csv")]
    for name, paths in systems.items():
        with_emoji, without_emoji = [], []
        for path in paths:
            pred = np.array(predictions_from_errors(val, path))
            with_emoji.append(macro_f1(true[has_emoji], pred[has_emoji]))
            without_emoji.append(macro_f1(true[~has_emoji], pred[~has_emoji]))
        emoji_rows.append({"system": name, "with_emoji_f1": round(np.mean(with_emoji), 4),
                           "without_emoji_f1": round(np.mean(without_emoji), 4),
                           "gap (with - without)": round(np.mean(with_emoji) - np.mean(without_emoji), 4),
                           "n_with": int(has_emoji.sum()), "n_without": int((~has_emoji).sum())})
    emoji_table = pd.DataFrame(emoji_rows)
    emoji_table.to_csv(os.path.join(RESULTS, "transformer_emoji.csv"), index=False)
    print(emoji_table.to_string(index=False))

    # ------------------------------------------------------------------ 5. choose the winner (BEFORE test)
    means = {m: np.mean([summaries[(m, s)]["best_val"]["macro_f1"] for s in seeds_for[m]]) for m in models}
    ranking = sorted(models, key=lambda m: means[m], reverse=True)
    winner, runner_up = ranking[0], ranking[1]
    stds = {m: np.std([summaries[(m, s)]["best_val"]["macro_f1"] for s in seeds_for[m]], ddof=1) for m in models}
    margin = means[winner] - means[runner_up]
    lines = [f"Transformer choice, made on VALIDATION only ({datetime.now():%Y-%m-%d %H:%M}).",
             "No test-set evaluation had been done when this file was written.", ""]
    for m in ranking:
        lines.append(f"  {SHORT[m]:26s} mean val macro-F1 {means[m]:.4f} ± {stds[m]:.4f}  ({len(seeds_for[m])} seeds)")
    lines += ["", f"WINNER: {winner}",
              f"Margin over runner-up ({SHORT[runner_up]}): {margin:+.4f} "
              f"= {margin / max(stds[winner], stds[runner_up], 1e-9):.1f} x the larger seed std."]
    if margin < max(stds[winner], stds[runner_up]):
        lines.append("NOTE: the margin is smaller than the seed-to-seed std, so the ranking is not clear-cut.")
    choice_text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS, "transformer_choice.txt"), "w", encoding="utf-8") as f:
        f.write(choice_text)
    print("\n=== 5. Decision (saved to results/transformer_choice.txt) ===")
    print(choice_text)

    # ------------------------------------------------------------------ 6. error analysis (validation)
    print("=== 6. Error analysis (validation; best seed of each model) ===")
    best_seed = {m: max(seeds_for[m], key=lambda s: summaries[(m, s)]["best_val"]["macro_f1"]) for m in models}
    error_files = {m: os.path.join(RESULTS, f"{SHORT[m]}_seed{best_seed[m]}_val_errors.csv") for m in models}
    errors = {m: pd.read_csv(error_files[m], dtype=str, keep_default_na=False) for m in models}
    ids = {m: set(errors[m]["id"]) for m in models}
    for m in models:
        print(f"  {SHORT[m]} (seed {best_seed[m]}): {len(ids[m])} errors")
    all_transformers = set.intersection(*ids.values())
    tfidf_ids = set(pd.read_csv(os.path.join(RESULTS, "ml_val_errors.csv"), dtype=str, keep_default_na=False)["id"])
    lstm_ids = set(pd.read_csv(os.path.join(RESULTS, "lstm_val_errors.csv"), dtype=str, keep_default_na=False)["id"])
    all_five = all_transformers & tfidf_ids & lstm_ids
    print(f"  wrong for ALL {len(models)} transformers: {len(all_transformers)} of {len(val)} "
          f"({100 * len(all_transformers) / len(val):.1f}%)")
    print(f"  wrong for ALL models so far (TF-IDF, BiLSTM, {len(models)} transformers): {len(all_five)} "
          f"({100 * len(all_five) / len(val):.1f}%)")
    labels_of_all_five = val[val["id"].isin(all_five)]["label"].value_counts().to_dict()
    print(f"  true labels of those {len(all_five)}: {labels_of_all_five}")

    others = [m for m in models if m != winner]
    only_winner_right = set.intersection(*[ids[m] for m in others]) - ids[winner]
    print(f"\n  tweets the winner ({SHORT[winner]}) gets right but BOTH other transformers get wrong: {len(only_winner_right)}")
    confidence = {m: dict(zip(errors[m]["id"], errors[m]["confidence"].astype(float))) for m in others}
    predicted = {m: dict(zip(errors[m]["id"], errors[m]["predicted_label"])) for m in others}
    ranked = sorted(only_winner_right, key=lambda i: -np.mean([confidence[m][i] for m in others]))
    by_id = val.set_index("id")
    for tweet_id in ranked[:3]:
        print(f"\n  id {tweet_id}  true: {by_id.loc[tweet_id, 'label']}  (winner correct)")
        print(f"     {by_id.loc[tweet_id, 'cleaned']}")
        print("     others said: " + ", ".join(f"{SHORT[m]} -> {predicted[m][tweet_id]} ({confidence[m][tweet_id]:.2f})" for m in others))


if __name__ == "__main__":
    main()
