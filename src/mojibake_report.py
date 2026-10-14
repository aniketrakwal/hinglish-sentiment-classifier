"""Report what the mojibake fix changed. Run from the project root:

    python src/mojibake_report.py

Reads the RAW files, applies fix_mojibake to every token, and prints how many tokens changed
(split into Devanagari / emoji / punctuation / other), 5 Devanagari before/after examples,
and how many saved tweets contain real Devanagari.
"""

import re
import unicodedata

import pandas as pd

from load_data import fix_mojibake

RAW_FILES = {
    "train": "data/raw/train_14k_split_conll.txt",
    "val": "data/raw/dev_3k_split_conll.txt",
    "test": "data/raw/Hindi_test_unalbelled_conll_updated.txt",
}
DEVANAGARI = re.compile("[ऀ-ॿ]")


def kind_of(fixed_token):
    """Rough category of a repaired token."""
    if DEVANAGARI.search(fixed_token):
        return "devanagari"
    for ch in fixed_token:
        code = ord(ch)
        # Emoji and pictograph blocks, plus the variation selector / joiner used inside emoji
        if code >= 0x1F000 or 0x2190 <= code <= 0x2BFF or code in (0xFE0F, 0x200D):
            return "emoji"
    if all(unicodedata.category(ch).startswith("P") for ch in fixed_token):
        return "punctuation"
    return "other"


print("1. Tokens changed per split (previous run, markers list: test = 2,739)")
devanagari_examples = []
for split, path in RAW_FILES.items():
    kinds = {"devanagari": 0, "emoji": 0, "punctuation": 0, "other": 0}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = [p for p in re.split(r"[ \t]+", line.strip(" \t\r\n")) if p]
            if not parts or parts[0] == "meta":
                continue
            fixed = fix_mojibake(parts[0])
            if fixed != parts[0]:
                kind = kind_of(fixed)
                kinds[kind] += 1
                if kind == "devanagari" and len(devanagari_examples) < 5:
                    devanagari_examples.append((parts[0], fixed))
    print(f"   {split}: {sum(kinds.values())}  {kinds}")

print("\n2. Tweets containing real Devanagari in the saved CSVs")
for split in RAW_FILES:
    df = pd.read_csv(f"data/processed/{split}.csv", dtype=str, keep_default_na=False)
    count = sum(1 for text in df["text"] if DEVANAGARI.search(text))
    print(f"   {split}: {count} of {len(df)}")

print("\n3. Devanagari before / after examples")
for before, after in devanagari_examples:
    print(f"   {before!r} -> {after!r}")
