"""Why do all models score about 6 points higher on test than on validation?

    python src/val_test_gap.py

Looks at the data and at the saved predictions (results/predictions/); trains no model for the task
(the only fitted model is the small "val or test?" classifier of the adversarial validation).
Prints facts; the reading of them is written down in results.md.
"""

import re
import sys

import emoji
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocess import clean_text
from prediction_files import LABELS, MODELS, label_ids, load_model_runs, load_run
from train_ml import make_features

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "data/processed"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
URL_WORD = re.compile(r"\bhttps?\b", re.IGNORECASE)         # in this data a URL is split into tokens: "https // t . co / ..."

# Rough word lists (lower-case tokens of the cleaned text). They are only a rough hint, not a labelled lexicon.
# PROFANITY comes from the top negative character n-grams in results/ml_top_features.txt
# ('bc', 'fuck', 'shame', 'kut', 'chu', 'ale') plus the common spellings around them.
PROFANITY = {"bc", "mc", "bkl", "bsdk", "fuck", "fucking", "fucked", "shame", "kutta", "kutte", "kamina", "kameena",
             "chutiya", "chutiye", "chu", "sale", "saale", "saala", "harami", "gandu", "madarchod", "bhosdike"}
GREETING = {"birthday", "bday", "janamdin", "happy", "congratulations", "congrats", "wish", "wishes", "wishing",
            "mubarak", "badhai", "anniversary", "blessed", "blessings", "greetings"}


# ---------------------------------------------------------------------------------------------
# Near-duplicates (same definition as notebooks/01_eda.ipynb, section 9): same text after lowercasing and
# removing URLs. In this data a URL is split into tokens, e.g. "https // t . co / AbC123xyz".
# ---------------------------------------------------------------------------------------------
URL_PIECES = {"//", "t", "co", "tco", ".", "/", ":"}


def find_urls(tokens):
    """Return (start, end) of every URL in a token list (copy of the EDA helper, without the 'shape')."""
    found, i = [], 0
    while i < len(tokens):
        if tokens[i].lower() in ("http", "https"):
            j = i + 1
            while j < len(tokens) and tokens[j] in URL_PIECES:
                j += 1
            if j < len(tokens) and tokens[j - 1] == "/" and re.fullmatch(r"[A-Za-z0-9]+", tokens[j]):
                j += 1                                           # the random id after the last "/"
            elif j < len(tokens) and tokens[j].startswith("�"):
                j += 1                                           # a cut-off URL ends in a replacement character
            found.append((i, j))
            i = j
        else:
            i += 1
    return found


def near_key(text):
    tokens = text.split()
    skip = set()
    for start, end in find_urls(tokens):
        skip.update(range(start, end))
    return " ".join(t.lower() for k, t in enumerate(tokens) if k not in skip)


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

def share(mask):
    return 100 * float(np.mean(mask))


def composition(df):
    """Per-tweet facts about one split (booleans and counts)."""
    cleaned = df["text"].apply(clean_text)
    tokens = cleaned.str.split()
    return pd.DataFrame({
        "n_tokens": tokens.str.len(),
        "emoji": df["text"].apply(lambda t: len(emoji.emoji_list(t)) > 0),
        "url": df["text"].apply(lambda t: bool(URL_WORD.search(t))),
        "mention": df["text"].str.contains("@"),
        "devanagari": df["text"].apply(lambda t: bool(DEVANAGARI.search(t))),
        "profanity": tokens.apply(lambda ts: any(t in PROFANITY for t in ts)),
        "greeting": tokens.apply(lambda ts: any(t in GREETING for t in ts)),
    })


def per_class_f1(table):
    y, p = label_ids(table["true_label"]), label_ids(table["predicted_label"])
    return np.append(f1_score(y, p, labels=[0, 1, 2], average=None), f1_score(y, p, labels=[0, 1, 2], average="macro"))


def mean_over_seeds(model, split):
    """Per-class F1 + macro-F1 (4 numbers), averaged over the seeds of a model."""
    return np.mean([per_class_f1(t) for t in load_model_runs(model, split)], axis=0)


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", dtype=str, keep_default_na=False)
    val = pd.read_csv(f"{DATA_DIR}/val.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(f"{DATA_DIR}/test.csv", dtype=str, keep_default_na=False)

    # ---- 1. Label distribution
    print("1. LABEL DISTRIBUTION (% of tweets)")
    dist = pd.DataFrame({name: df["label"].value_counts(normalize=True).reindex(LABELS) * 100
                         for name, df in [("train", train), ("val", val), ("test", test)]})
    print(dist.round(1).to_string())

    # ---- 2. Composition
    print("\n2. COMPOSITION (val vs test)")
    comp_val, comp_test = composition(val), composition(test)
    print(f"   tokens per tweet after clean_text: val mean {comp_val['n_tokens'].mean():.1f} / median "
          f"{comp_val['n_tokens'].median():.0f}   test mean {comp_test['n_tokens'].mean():.1f} / median "
          f"{comp_test['n_tokens'].median():.0f}")
    for column in ["emoji", "url", "mention", "devanagari", "profanity", "greeting"]:
        print(f"   % tweets with {column:11s}: val {share(comp_val[column]):5.1f}   test {share(comp_test[column]):5.1f}")

    # ---- 3. Where is the gap? Per-class F1, val vs test
    print("\n3. PER-CLASS F1, val vs test (mean over seeds)")
    for model in ["hingroberta_full", "hingroberta_lora_r16", "bilstm", "tfidf"]:
        v, t = mean_over_seeds(model, "val"), mean_over_seeds(model, "test")
        print(f"   {MODELS[model][0]:24s} " + "  ".join(
            f"{name} {v[i]:.3f}->{t[i]:.3f} ({t[i] - v[i]:+.3f})" for i, name in enumerate(["neg", "neu", "pos", "MACRO"])))

    # ---- 4. Near-duplicates of train tweets
    print("\n4. NEAR-DUPLICATES OF TRAIN TWEETS (EDA definition: lower-case, URLs removed)")
    train_keys = train.groupby(train["text"].apply(near_key))["label"].agg(set)      # key -> set of train labels
    near = {}
    for name, df in [("val", val), ("test", test)]:
        keys = df["text"].apply(near_key)
        has_twin = keys.isin(train_keys.index).to_numpy()
        same_label = np.array([has and label in train_keys[key] for key, label, has in zip(keys, df["label"], has_twin)])
        near[name] = has_twin
        print(f"   {name}: {share(has_twin):.1f}% of tweets ({int(has_twin.sum())}) have a near-twin in train; "
              f"for {share(same_label[has_twin]):.0f}% of those the label also appears in train for that text")
    for model in ["hingroberta_full", "tfidf"]:
        for name in ["val", "test"]:
            acc_twin = np.mean([accuracy_score(t["true_label"][near[name]], t["predicted_label"][near[name]])
                                for t in load_model_runs(model, name)])
            acc_rest = np.mean([accuracy_score(t["true_label"][~near[name]], t["predicted_label"][~near[name]])
                                for t in load_model_runs(model, name)])
            print(f"   {MODELS[model][0]:20s} {name} accuracy: tweets WITH a twin {acc_twin:.3f}, without {acc_rest:.3f}")

    # ---- 5. Easy / hard tweets: how many are right (or wrong) for all four systems?
    print("\n5. EASY AND HARD TWEETS (TF-IDF, BiLSTM s42, HingRoBERTa s42, LoRA r=16 s42)")
    for split in ["val", "test"]:
        correct = []
        for model, seed in [("tfidf", None), ("bilstm", 42), ("hingroberta_full", 42), ("hingroberta_lora_r16", 42)]:
            t = load_run(model, seed, split)
            correct.append((t["true_label"] == t["predicted_label"]).to_numpy())
        correct = np.array(correct)
        hing = load_run("hingroberta_full", 42, split)
        print(f"   {split}: all four right {share(correct.all(axis=0)):.1f}%   all four wrong "
              f"{share(~correct.any(axis=0)):.1f}%   HingRoBERTa mean confidence "
              f"{hing[['prob_negative', 'prob_neutral', 'prob_positive']].max(axis=1).mean():.3f}")

    # ---- 6. Adversarial validation: can a classifier tell val tweets from test tweets?
    print("\n6. ADVERSARIAL VALIDATION (TF-IDF word + char n-grams + logistic regression: val (0) vs test (1))")
    texts = pd.concat([val["text"], test["text"]]).apply(clean_text).to_numpy()
    domain = np.array([0] * len(val) + [1] * len(test))
    model = Pipeline([("features", make_features("word + char")),
                      ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fit_idx, eval_idx in folds.split(texts, domain):
        model.fit(texts[fit_idx], domain[fit_idx])
        aucs.append(roc_auc_score(domain[eval_idx], model.predict_proba(texts[eval_idx])[:, 1]))
    print(f"   5-fold ROC-AUC {np.mean(aucs):.3f} +- {np.std(aucs, ddof=1):.3f}   (0.5 = cannot tell them apart, 1.0 = easy)")
    # Control: the same thing with the domain labels shuffled must give about 0.5
    shuffled = np.random.default_rng(0).permutation(domain)
    control = []
    for fit_idx, eval_idx in folds.split(texts, shuffled):
        model.fit(texts[fit_idx], shuffled[fit_idx])
        control.append(roc_auc_score(shuffled[eval_idx], model.predict_proba(texts[eval_idx])[:, 1]))
    print(f"   control with shuffled val/test labels: ROC-AUC {np.mean(control):.3f}")


if __name__ == "__main__":
    main()
