"""Show what clean_text does. Run from the project root:

    python src/preprocess_report.py

C. 15 random before/after pairs from train (some with URLs, split mentions, hashtags, emoji).
D. Tweet length in tokens BEFORE and AFTER cleaning, per split.
"""

import emoji
import numpy as np
import pandas as pd

from preprocess import clean_text

SPLITS = ["train", "val", "test"]
data = {s: pd.read_csv(f"data/processed/{s}.csv", dtype=str, keep_default_na=False) for s in SPLITS}


def has_url(t):
    return "https" in t.split() or "http" in t.split()


def has_underscore_mention(t):
    tokens = t.split()
    return any(tok == "@" and i + 3 < len(tokens) and tokens[i + 2] == "_" for i, tok in enumerate(tokens))


def has_hashtag(t):
    return "#" in t.split()


def has_emoji(t):
    return len(emoji.emoji_list(t)) > 0


# ---------------------------------------------------------------- C. examples
print("C. 15 random before/after pairs (train, seed 7)\n")
rng = np.random.RandomState(7)
train = data["train"]
picked = []
for name, test_fn, how_many in [("URL", has_url, 4), ("multi-part mention", has_underscore_mention, 4),
                                ("hashtag", has_hashtag, 3), ("emoji", has_emoji, 3)]:
    pool = [i for i in range(len(train)) if test_fn(train.loc[i, "text"]) and i not in [p[1] for p in picked]]
    for i in rng.choice(pool, size=how_many, replace=False):
        picked.append((name, int(i)))
# one plain random tweet to finish
picked.append(("random", int(rng.choice([i for i in range(len(train)) if i not in [p[1] for p in picked]]))))

for k, (name, i) in enumerate(picked, 1):
    raw = train.loc[i, "text"]
    print(f"{k:2d}. [{name}] id {train.loc[i, 'id']}")
    print("    before:", raw)
    print("    after: ", clean_text(raw), "\n")

# ---------------------------------------------------------------- D. lengths
print("D. Tweet length in tokens, before vs after cleaning\n")
rows = []
for s in SPLITS:
    for stage in ["before", "after"]:
        texts = data[s]["text"] if stage == "before" else data[s]["text"].apply(clean_text)
        n = texts.str.split().str.len()
        rows.append({"split": s, "stage": stage, "mean": round(n.mean(), 1), "median": n.median(),
                     "p95": n.quantile(0.95), "p99": n.quantile(0.99), "max": n.max(), "min": n.min()})
table = pd.DataFrame(rows).set_index(["split", "stage"])
print(table.to_string())
