"""Classical ML baseline: TF-IDF + Logistic Regression.

Run from the project root:
    python src/train_ml.py

What it does (all tuning uses the VALIDATION set only; test is used exactly once at the end):
  1. dummy floor (always predicts the most frequent class)
  2. three feature setups (word n-grams, char n-grams, both) x five values of C
  3. picks the best setup by validation macro-F1, evaluates it once on test
  4. saves the model, confusion matrices, error list and top features
"""

import argparse
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from preprocess import clean_text

RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]
C_VALUES = [0.25, 0.5, 1, 2, 4]
SETUPS = ["word (1,2)", "char_wb (2,5)", "word + char"]


# ---------------------------------------------------------------------------------------------
# Building blocks (small functions so the tests can reuse them)
# ---------------------------------------------------------------------------------------------

def load_split(path):
    """Read one CSV and add a 'clean' column made with the shared clean_text()."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["clean"] = df["text"].apply(clean_text)
    return df


def make_word_vectorizer():
    """TF-IDF over word 1- and 2-grams."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        # The default token_pattern (r"(?u)\b\w\w+\b") only keeps words of 2+ letters/digits.
        # That would throw away single characters like "h" (a very common short form of "hai"),
        # all emoji, and all punctuation like "!" or "?". r"\S+" means "any run of non-space
        # characters is a token", and clean_text() has already put spaces around emoji.
        token_pattern=r"\S+",
        lowercase=False,     # clean_text() already lowercased; keep ONE source of preprocessing
        sublinear_tf=True,   # use 1 + log(count) so a word repeated 10 times is not 10x as important
        min_df=2,            # ignore n-grams that occur in only one tweet
    )


def make_char_vectorizer():
    """TF-IDF over character 2- to 5-grams taken inside word boundaries (handles spelling variants)."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
    )


def make_features(setup):
    """Return the (unfitted) feature extractor for a setup name."""
    if setup == "word (1,2)":
        return make_word_vectorizer()
    if setup == "char_wb (2,5)":
        return make_char_vectorizer()
    if setup == "word + char":
        # FeatureUnion runs both vectorizers and glues their columns side by side
        return FeatureUnion([("word", make_word_vectorizer()), ("char", make_char_vectorizer())])
    raise ValueError("unknown setup: " + setup)


def make_classifier(C):
    """Logistic regression; class_weight='balanced' gives smaller classes a bigger say."""
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)


def make_pipeline(setup, C):
    """Features + classifier in one object, so raw cleaned text goes in and labels come out."""
    return Pipeline([("features", make_features(setup)), ("clf", make_classifier(C))])


def scores(y_true, y_pred):
    """Accuracy, macro-F1 and the F1 of each class."""
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None)
    return {
        "val_accuracy": accuracy_score(y_true, y_pred),
        "val_macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro"),
        "f1_negative": per_class[0],
        "f1_neutral": per_class[1],
        "f1_positive": per_class[2],
    }


def plot_confusion(y_true, y_pred, title, path):
    """Confusion matrix as a heatmap: colour = share of the true class, text = count and share."""
    counts = confusion_matrix(y_true, y_pred, labels=LABELS)
    shares = counts / counts.sum(axis=1, keepdims=True)
    text = np.array([[f"{counts[i, j]}\n{100 * shares[i, j]:.0f}%" for j in range(3)] for i in range(3)])
    plt.figure(figsize=(5, 4))
    sns.heatmap(shares, annot=text, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def top_features(model, n=20):
    """The n highest-weighted features of each class from the fitted logistic regression."""
    names = model.named_steps["features"].get_feature_names_out()
    clf = model.named_steps["clf"]
    lines = []
    for row, label in enumerate(clf.classes_):
        best = np.argsort(clf.coef_[row])[::-1][:n]          # indexes of the largest weights
        lines.append(f"=== {label} (top {n} features) ===")
        for k in best:
            lines.append(f"{clf.coef_[row][k]:8.3f}  {names[k]!r}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_path", default="models/tfidf_logreg.joblib")
    args = parser.parse_args()

    fig_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # ---- 1. Load train and validation (test is NOT loaded until step 4)
    train = load_split(os.path.join(args.data_dir, "train.csv"))
    val = load_split(os.path.join(args.data_dir, "val.csv"))
    print(f"train {train.shape}, val {val.shape}")

    # ---- 2. Dummy floor: always predicts the most frequent training class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train["clean"], train["label"])
    dummy_pred = dummy.predict(val["clean"])
    dummy_scores = scores(val["label"], dummy_pred)
    print(f"\nDummy (most_frequent): val accuracy {dummy_scores['val_accuracy']:.4f}, "
          f"val macro-F1 {dummy_scores['val_macro_f1']:.4f}")

    # ---- 3. Experiments: for each setup fit the features once, then loop over C
    rows = []
    for setup in SETUPS:
        t0 = time.time()
        features = make_features(setup)
        X_train = features.fit_transform(train["clean"])     # learn vocabulary from TRAIN only
        X_val = features.transform(val["clean"])
        feature_time = time.time() - t0
        n_features = X_train.shape[1]

        for C in C_VALUES:
            t1 = time.time()
            clf = make_classifier(C).fit(X_train, train["label"])
            train_time = feature_time + (time.time() - t1)    # feature extraction + fitting
            row = {"setup": setup, "C": C}
            row.update(scores(val["label"], clf.predict(X_val)))
            row["n_features"] = n_features
            row["train_time_s"] = round(train_time, 1)
            rows.append(row)
            print(f"  {setup:14s} C={C:<5} val acc {row['val_accuracy']:.4f}  "
                  f"macro-F1 {row['val_macro_f1']:.4f}  ({row['train_time_s']}s)")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.results_dir, "ml_experiments.csv"), index=False, float_format="%.4f")
    print("\n" + table.round(4).to_string(index=False))

    # ---- 4. Best setup by validation macro-F1, refit on train, evaluate on test ONCE
    best = table.loc[table["val_macro_f1"].idxmax()]
    best_setup, best_C = best["setup"], float(best["C"])
    print(f"\nBest by validation macro-F1: {best_setup}, C={best_C} (val macro-F1 {best['val_macro_f1']:.4f})")

    model = make_pipeline(best_setup, best_C)
    model.fit(train["clean"], train["label"])
    val_pred = model.predict(val["clean"])

    test = load_split(os.path.join(args.data_dir, "test.csv"))    # first time test is touched
    test_pred = model.predict(test["clean"])
    dummy_test_scores = scores(test["label"], dummy.predict(test["clean"]))
    for name, df, pred in [("VALIDATION", val, val_pred), ("TEST", test, test_pred)]:
        print(f"\n===== {name} =====")
        print(f"accuracy {accuracy_score(df['label'], pred):.4f}   "
              f"macro-F1 {f1_score(df['label'], pred, labels=LABELS, average='macro'):.4f}")
        print(classification_report(df["label"], pred, labels=LABELS, digits=4))
    print(f"Dummy on test (for the results table): accuracy {dummy_test_scores['val_accuracy']:.4f}, "
          f"macro-F1 {dummy_test_scores['val_macro_f1']:.4f}")

    # ---- 5. Save everything
    joblib.dump(model, args.model_path)
    plot_confusion(val["label"], val_pred, f"Validation ({best_setup}, C={best_C})",
                   os.path.join(fig_dir, "ml_confusion_val.png"))
    plot_confusion(test["label"], test_pred, f"Test ({best_setup}, C={best_C})",
                   os.path.join(fig_dir, "ml_confusion_test.png"))

    # Misclassified validation tweets, most confident mistakes first
    probs = model.predict_proba(val["clean"])
    confidence = probs.max(axis=1)
    wrong = val_pred != val["label"].to_numpy()
    errors = pd.DataFrame({
        "id": val["id"], "text": val["text"], "cleaned_text": val["clean"],
        "true_label": val["label"], "predicted_label": val_pred, "confidence": confidence.round(4),
    })[wrong].sort_values("confidence", ascending=False)
    errors.to_csv(os.path.join(args.results_dir, "ml_val_errors.csv"), index=False)
    print(f"\nSaved {len(errors)} misclassified validation tweets")

    with open(os.path.join(args.results_dir, "ml_top_features.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_setup}, C={best_C}\n\n")
        f.write(top_features(model))
    print("Saved model, confusion matrices, errors and top features.")


if __name__ == "__main__":
    main()
