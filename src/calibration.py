"""Calibration of HingRoBERTa (full fine-tune) and, for comparison, TF-IDF + LogReg.

    python src/calibration.py

Reads the saved predictions (results/predictions/); trains nothing.

A model is "calibrated" if, among the tweets where it says "80% sure", it is right about 80% of the time.
  confidence = the probability the model gives to its predicted class
  ECE (expected calibration error) = sort the tweets into 10 confidence bins, and in each bin compare the
      average confidence with the real accuracy; ECE is the average of these gaps, weighted by bin size.
      0 = perfectly calibrated.
Temperature scaling: divide the model's scores (logits) by one number T before the softmax.
  T > 1 makes the probabilities less extreme (fixes over-confidence), T < 1 makes them sharper.
  It never changes which class has the highest probability, so accuracy and F1 stay exactly the same.
  T is fitted on VALIDATION only (minimise the validation log-loss) and then applied to test.
(The saved files hold probabilities, not logits. That is enough: log(probabilities) differ from the
logits only by a constant per tweet, and softmax ignores such a constant, so softmax(log(p) / T) is
exactly the same as softmax(logits / T).)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")                      # draw into files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prediction_files import MODELS, SEEDS, label_ids, load_run, probabilities

sys.stdout.reconfigure(encoding="utf-8")

N_BINS = 10
RESULTS = "results"
FIG_DIR = os.path.join(RESULTS, "figures")
TEMPERATURE_GRID = np.arange(0.05, 10.0, 0.01)       # candidate values of T for the (simple) grid search


def apply_temperature(probs, temperature):
    """softmax(log(p) / T): the probabilities after temperature scaling."""
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)        # for numerical stability
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def log_loss(probs, y_true):
    """Average negative log of the probability given to the true class (lower = better)."""
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0))))


def fit_temperature(probs, y_true):
    """The T (from a grid) with the lowest log-loss. The loss has a single valley in T, so a grid is enough."""
    losses = [log_loss(apply_temperature(probs, t), y_true) for t in TEMPERATURE_GRID]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def reliability_bins(probs, y_true, n_bins=N_BINS):
    """Per confidence bin: number of tweets, mean confidence, accuracy. Bins are 0-0.1, 0.1-0.2, ..., 0.9-1.0."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    # bin number 0..9 of every tweet (a confidence of exactly 1.0 goes to the last bin)
    bin_of = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    count = np.zeros(n_bins)
    mean_conf = np.zeros(n_bins)
    accuracy = np.zeros(n_bins)
    for b in range(n_bins):
        in_bin = bin_of == b
        count[b] = in_bin.sum()
        if count[b] > 0:
            mean_conf[b] = confidence[in_bin].mean()
            accuracy[b] = correct[in_bin].mean()
    return count, mean_conf, accuracy


def expected_calibration_error(probs, y_true, n_bins=N_BINS):
    count, mean_conf, accuracy = reliability_bins(probs, y_true, n_bins)
    return float(np.sum(count / count.sum() * np.abs(accuracy - mean_conf)))


def confidence_split(probs, y_true):
    """Average confidence on correct and on wrong predictions."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    return float(confidence[correct].mean()), float(confidence[~correct].mean())


def plot_reliability(ax, probs, y_true, title):
    count, mean_conf, accuracy = reliability_bins(probs, y_true)
    centers = (np.arange(N_BINS) + 0.5) / N_BINS
    filled = count > 0
    ax.bar(centers[filled], accuracy[filled], width=1 / N_BINS, edgecolor="black", color="#3b6ea5",
           alpha=0.8, label="accuracy in bin")
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE {expected_calibration_error(probs, y_true):.3f}", fontsize=10)


def calibrate_run(model, seed, make_plot):
    """Fit T on val, evaluate before / after on val and test. Returns a result dict."""
    val, test = load_run(model, seed, "val"), load_run(model, seed, "test")
    y_val, y_test = label_ids(val["true_label"]), label_ids(test["true_label"])
    p_val, p_test = probabilities(val), probabilities(test)

    temperature = fit_temperature(p_val, y_val)                  # VALIDATION only
    q_val, q_test = apply_temperature(p_val, temperature), apply_temperature(p_test, temperature)

    # Scaling must never change a prediction
    assert (q_val.argmax(axis=1) == p_val.argmax(axis=1)).all()
    assert (q_test.argmax(axis=1) == p_test.argmax(axis=1)).all()

    row = {"model": MODELS[model][0], "seed": seed if seed is not None else "", "temperature": temperature,
           "argmax_unchanged": True}
    for split, p, q, y in [("val", p_val, q_val, y_val), ("test", p_test, q_test, y_test)]:
        correct_before, wrong_before = confidence_split(p, y)
        correct_after, wrong_after = confidence_split(q, y)
        row.update({
            f"{split}_ece_before": expected_calibration_error(p, y), f"{split}_ece_after": expected_calibration_error(q, y),
            f"{split}_logloss_before": log_loss(p, y), f"{split}_logloss_after": log_loss(q, y),
            f"{split}_conf_correct_before": correct_before, f"{split}_conf_wrong_before": wrong_before,
            f"{split}_conf_correct_after": correct_after, f"{split}_conf_wrong_after": wrong_after,
            f"{split}_accuracy": float((p.argmax(axis=1) == y).mean()),
        })
    if make_plot:
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        plot_reliability(axes[0, 0], p_val, y_val, "VALIDATION, before temperature scaling")
        plot_reliability(axes[0, 1], q_val, y_val, f"VALIDATION, after (T = {temperature:.2f}, fitted here)")
        plot_reliability(axes[1, 0], p_test, y_test, "TEST, before temperature scaling")
        plot_reliability(axes[1, 1], q_test, y_test, f"TEST, after (T = {temperature:.2f} from validation)")
        axes[0, 0].legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Reliability diagrams: {MODELS[model][0]}" + (f", seed {seed}" if seed is not None else ""))
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, f"calibration_{model}.png"), dpi=130)
        plt.close(fig)
    return row


def main():
    rows = []
    for seed in SEEDS:                                        # HingRoBERTa full: every seed gets its own T
        rows.append(calibrate_run("hingroberta_full", seed, make_plot=(seed == SEEDS[0])))
    rows.append(calibrate_run("tfidf", None, make_plot=True))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "calibration.csv"), index=False, float_format="%.4f")

    print("Temperature T (fitted on validation only) and ECE before -> after")
    print("(the diagrams in results/figures/ show HingRoBERTa seed 42 and TF-IDF)\n")
    for _, r in table.iterrows():
        name = f"{r['model']}" + (f" seed {r['seed']}" if r["seed"] != "" else "")
        print(f"{name}: T = {r['temperature']:.2f}   predictions unchanged: {r['argmax_unchanged']}")
        for split in ["val", "test"]:
            print(f"   {split:4s} ECE {r[f'{split}_ece_before']:.4f} -> {r[f'{split}_ece_after']:.4f}   "
                  f"log-loss {r[f'{split}_logloss_before']:.4f} -> {r[f'{split}_logloss_after']:.4f}   "
                  f"accuracy {r[f'{split}_accuracy']:.4f}")
            print(f"        mean confidence, correct predictions: {r[f'{split}_conf_correct_before']:.3f} -> "
                  f"{r[f'{split}_conf_correct_after']:.3f};  wrong predictions: "
                  f"{r[f'{split}_conf_wrong_before']:.3f} -> {r[f'{split}_conf_wrong_after']:.3f}")

    hing = table[table["model"] == MODELS["hingroberta_full"][0]]
    print("\nHingRoBERTa, mean over the 3 seeds:")
    print(f"   T {hing['temperature'].mean():.2f}")
    for split in ["val", "test"]:
        print(f"   {split} ECE {hing[f'{split}_ece_before'].mean():.4f} -> {hing[f'{split}_ece_after'].mean():.4f}")


if __name__ == "__main__":
    main()
