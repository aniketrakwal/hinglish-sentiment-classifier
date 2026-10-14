"""Check that the mojibake fix worked. Run from the project root:

    python src/check_mojibake.py

It prints: tokens changed per split, tweets still containing marker characters,
5 before/after examples, and whether token count == lang_tags count in every row.
"""

import re

import pandas as pd

from load_data import MOJIBAKE_MARKERS, fix_mojibake

RAW_FILES = {
    "train": "data/raw/train_14k_split_conll.txt",
    "val": "data/raw/dev_3k_split_conll.txt",
    "test": "data/raw/Hindi_test_unalbelled_conll_updated.txt",
}


def has_marker(text):
    """True if the text contains any mojibake marker character."""
    for ch in text:
        if ch in MOJIBAKE_MARKERS:
            return True
    return False


# 1 + 3: go through the RAW token lines and count / collect the ones the fix changes
print("1. Tokens changed per split")
examples = []
for split, path in RAW_FILES.items():
    changed = 0
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            # Same splitting as the parser: tabs/spaces only, so U+00A0 stays inside the token
            parts = [p for p in re.split(r"[ \t]+", line.strip(" \t\r\n")) if p]
            if not parts or parts[0] == "meta":
                continue
            fixed = fix_mojibake(parts[0])
            if fixed != parts[0]:
                changed += 1
                if len(examples) < 5:
                    examples.append((parts[0], fixed))
    print(f"   {split}: {changed}")

# 2 + 4: check the saved CSVs
print("\n2. Tweets still containing marker characters (CSV)")
alignment_ok = {}
for split in RAW_FILES:
    df = pd.read_csv(f"data/processed/{split}.csv", dtype=str, keep_default_na=False)
    print(f"   {split}: {sum(has_marker(t) for t in df['text'])}")
    # Same number of tokens and tags in every row?
    n_tokens = df["text"].str.split().str.len()
    n_tags = df["lang_tags"].str.split().str.len()
    alignment_ok[split] = (int((n_tokens != n_tags).sum()), len(df))

print("\n3. Before / after examples")
for before, after in examples:
    print(f"   {before!r} -> {after!r}")

print("\n4. Rows where token count != lang_tags count")
for split, (bad, total) in alignment_ok.items():
    print(f"   {split}: {bad} of {total} rows misaligned")
