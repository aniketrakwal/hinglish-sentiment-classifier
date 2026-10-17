"""Bootstrap confidence intervals for macro-F1, and paired bootstrap tests for model differences.

    python src/bootstrap.py

Reads the saved predictions (results/predictions/, made by src/collect_predictions.py); trains nothing.

Bootstrap in one paragraph: the 3000 tweets of a split are only one sample of all possible tweets.
We draw 3000 tweets WITH replacement (a "resample"), recompute the macro-F1 on it, and repeat 1000
times. The middle 95% of those 1000 scores (2.5th to 97.5th percentile) is the 95% confidence interval.
It shows how much the score moves just because of WHICH tweets are in the split. (It does not include
the randomness of training; the seed-to-seed spread is a separate thing.)
For models with 3 seeds, every resample gives one F1 per seed and we use their mean.

Paired bootstrap: to compare model A with model B, both are scored on the SAME resample and we keep
the difference A - B. Using the same tweets for both removes most of the noise, because a hard tweet is
hard for both models. If the 95% interval of the difference does not contain 0, the difference is
statistically clear (it keeps the same sign in at least 97.5% of the resamples).
"""

import os
import sys

import numpy as np
import pandas as pd

from prediction_files import MODELS, TEST_MODELS, label_ids, load_model_runs

sys.stdout.reconfigure(encoding="utf-8")

N_RESAMPLES = 1000
RANDOM_SEED = 2024                 # fixed, so the intervals are the same on every run
RESULTS = "results"

# (model A, model B): the differences asked for. They are computed on every split where BOTH models have predictions.
PAIRS = [
    ("hingroberta_full", "tfidf"),
    ("hingroberta_full", "muril"),
    ("hingroberta_full", "xlmr"),
    ("hingroberta_full", "hingroberta_lora_r16"),
    ("muril", "xlmr"),
]
CI_MODELS_ORDER = list(MODELS)


def macro_f1(y_true, y_pred):
    """Macro-F1 for class ids 0/1/2. Same maths as sklearn's f1_score(average="macro"), but fast:
    a 3x3 confusion matrix (rows = true class, columns = predicted class) from one bincount."""
    confusion = np.bincount(y_true * 3 + y_pred, minlength=9).reshape(3, 3)
    tp = np.diag(confusion)                           # correctly predicted, per class
    fp = confusion.sum(axis=0) - tp                   # predicted as the class, but wrong
    fn = confusion.sum(axis=1) - tp                   # belong to the class, but missed
    denominator = 2 * tp + fp + fn
    f1 = np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.0)   # 0 if a class never occurs
    return f1.mean()


def resample_indices(n_rows, n_resamples=N_RESAMPLES, seed=RANDOM_SEED):
    """[n_resamples, n_rows] matrix: each row = tweet numbers drawn with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_rows, size=(n_resamples, n_rows))


def bootstrap_scores(y_true, predictions, indices):
    """Macro-F1 on every resample. `predictions` = list of predicted-id arrays (one per seed);
    the score of a resample is the mean over the seeds. Returns an array of length n_resamples."""
    scores = np.zeros(len(indices))
    for b, idx in enumerate(indices):
        scores[b] = np.mean([macro_f1(y_true[idx], pred[idx]) for pred in predictions])
    return scores


def interval(values):
    """The middle 95% of the values: (2.5th percentile, 97.5th percentile)."""
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def load_split_predictions(split, keys):
    """For each model key: the true ids and the list of predicted-id arrays (one per seed)."""
    y_true, preds = None, {}
    for key in keys:
        tables = load_model_runs(key, split)
        for table in tables:
            ids = label_ids(table["true_label"])
            if y_true is None:
                y_true, first_ids = ids, table["id"].tolist()
            # Every file must describe the same tweets in the same order, or the pairing would be wrong
            assert table["id"].tolist() == first_ids and (ids == y_true).all(), f"{key}: rows do not line up"
        preds[key] = [label_ids(table["predicted_label"]) for table in tables]
    return y_true, preds


def run_split(split):
    """All intervals and paired differences for one split. Returns a list of result rows."""
    keys = [k for k in CI_MODELS_ORDER if k in TEST_MODELS] if split == "test" else CI_MODELS_ORDER
    y_true, preds = load_split_predictions(split, keys)
    indices = resample_indices(len(y_true))                # the SAME resamples for every model
    scores = {key: bootstrap_scores(y_true, preds[key], indices) for key in keys}

    rows = []
    for key in keys:
        point = float(np.mean([macro_f1(y_true, p) for p in preds[key]]))     # score on the real split
        low, high = interval(scores[key])
        rows.append({"kind": "ci", "split": split, "model": MODELS[key][0], "model_b": "",
                     "point_estimate": point, "mean_bootstrap": float(scores[key].mean()),
                     "ci_low": low, "ci_high": high, "excludes_zero": ""})
    for a, b in PAIRS:
        if a not in scores or b not in scores:
            continue                                        # e.g. MuRIL has no test predictions
        diff = scores[a] - scores[b]                        # paired: same resample for both models
        point = (float(np.mean([macro_f1(y_true, p) for p in preds[a]]))
                 - float(np.mean([macro_f1(y_true, p) for p in preds[b]])))
        low, high = interval(diff)
        rows.append({"kind": "diff", "split": split, "model": MODELS[a][0], "model_b": MODELS[b][0],
                     "point_estimate": point, "mean_bootstrap": float(diff.mean()),
                     "ci_low": low, "ci_high": high, "excludes_zero": bool(low > 0 or high < 0)})
    return rows


def main():
    rows = run_split("val") + run_split("test")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "bootstrap.csv"), index=False, float_format="%.4f")

    for split in ["val", "test"]:
        part = table[(table["split"] == split) & (table["kind"] == "ci")]
        print(f"\n95% bootstrap CI of macro-F1, {split.upper()} ({N_RESAMPLES} resamples)")
        for _, r in part.iterrows():
            print(f"  {r['model']:24s} {r['point_estimate']:.4f}   [{r['ci_low']:.4f}, {r['ci_high']:.4f}]   "
                  f"(width {r['ci_high'] - r['ci_low']:.4f})")
        part = table[(table["split"] == split) & (table["kind"] == "diff")]
        print(f"\nPaired difference A - B in macro-F1, {split.upper()}")
        for _, r in part.iterrows():
            verdict = "CI excludes 0 (clear)" if r["excludes_zero"] else "CI contains 0 (not clear)"
            print(f"  {r['model']} - {r['model_b']}: {r['point_estimate']:+.4f}   "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]   {verdict}")


if __name__ == "__main__":
    main()
