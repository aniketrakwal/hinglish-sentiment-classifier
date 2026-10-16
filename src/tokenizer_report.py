"""Compare the sub-word tokenizers of the three transformer models on the TRAIN set (after clean_text).

Run from the project root:
    python src/tokenizer_report.py

Prints a comparison and saves it to results/tokenizer_report.txt. Only the tokenizers (and the model
configs, to count parameters) are downloaded, not the model weights. The test set is not used.
"""

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # silence a harmless Windows warning

import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_text

sys.stdout.reconfigure(encoding="utf-8")   # print emoji / Devanagari safely on Windows

MODELS = ["google/muril-base-cased", "l3cube-pune/hing-roberta", "xlm-roberta-base"]
CANDIDATE_MAX_LENGTHS = [32, 48, 64, 96, 128]
lines = []                                  # everything we print is also saved to the report file


def say(text=""):
    print(text)
    lines.append(text)


def count_parameters(name):
    """Number of weights of the classification model, counted without loading the real weights.
    'meta' device = build the model structure only, no memory is used for the numbers."""
    config = AutoConfig.from_pretrained(name, num_labels=3)
    with torch.device("meta"):
        model = AutoModelForSequenceClassification.from_config(config)
    return sum(p.numel() for p in model.parameters())


def main():
    train = pd.read_csv("data/processed/train.csv", dtype=str, keep_default_na=False)
    texts = [clean_text(t) for t in train["text"]]
    n_words = [len(t.split(" ")) for t in texts]        # words = space-separated tokens after cleaning
    say(f"Train tweets: {len(texts)}; words per tweet (after clean_text): mean {np.mean(n_words):.1f}")

    # Tweets to show: first Devanagari tweet, 2 random tweets (fixed seed)
    devanagari = next(t for t in texts if any("ऀ" <= ch <= "ॿ" for ch in t))
    rng = np.random.RandomState(42)
    random_tweets = [texts[i] for i in rng.choice(len(texts), 2, replace=False)]
    demo_texts = ["nahi nhi nai bahut bohot bahot", "@user congratulations bhai 🎉 🎉", devanagari] + random_tweets

    stats = {}
    tokenizers = {}
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        tokenizers[name] = tok
        # Token ids of every tweet WITH the special tokens ([CLS]/[SEP] or <s>/</s>), no truncation
        ids = tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
        lengths = np.array([len(x) for x in ids])
        n_special = len(tok("", add_special_tokens=True)["input_ids"])   # an empty text gives only the special tokens
        pieces = lengths - n_special                                    # sub-word pieces without specials
        unk_id = tok.unk_token_id
        unk_tokens = sum(sum(1 for i in x if i == unk_id) for x in ids)
        tweets_with_unk = sum(1 for x in ids if unk_id in x)
        stats[name] = {
            "parameters (M)": f"{count_parameters(name) / 1e6:.0f}",
            "vocab size": tok.vocab_size,
            "tokens/tweet mean": f"{lengths.mean():.1f}",
            "median": f"{np.median(lengths):.0f}",
            "p95": f"{np.percentile(lengths, 95):.0f}",
            "p99": f"{np.percentile(lengths, 99):.0f}",
            "max": f"{lengths.max()}",
            "pieces per word": f"{pieces.sum() / sum(n_words):.2f}",
            "% truncated at 64": f"{100 * (lengths > 64).mean():.2f}",
            "% truncated at 128": f"{100 * (lengths > 128).mean():.2f}",
            "unknown tokens": unk_tokens,
            "tweets with >=1 unknown": tweets_with_unk,
            "_lengths": lengths,
        }

    say("\n=== Comparison (lengths include the 2 special tokens) ===")
    table = pd.DataFrame({name: {k: v for k, v in s.items() if not k.startswith("_")} for name, s in stats.items()})
    say(table.to_string())

    # Recommend ONE max_length for all three: the smallest candidate covering >= 99% of tweets for every tokenizer
    say("\n=== Coverage of candidate max_length values (share of tweets that fit, %) ===")
    coverage = pd.DataFrame({name: {m: 100 * (s["_lengths"] <= m).mean() for m in CANDIDATE_MAX_LENGTHS}
                             for name, s in stats.items()}).round(2)
    say(coverage.to_string())
    # compare the UNROUNDED coverage so a value like 98.996 is not rounded up to 99.00
    recommended = next(m for m in CANDIDATE_MAX_LENGTHS
                       if all((s["_lengths"] <= m).mean() >= 0.99 for s in stats.values()))
    say(f"\nRecommended max_length for all three models: {recommended} "
        f"(smallest candidate with >= 99% coverage for every tokenizer)")

    say("\n=== How each tokenizer splits some text ===")
    for text in demo_texts:
        say(f"\nTEXT: {text}")
        for name in MODELS:
            say(f"  {name}: {tokenizers[name].tokenize(text)}")

    os.makedirs("results", exist_ok=True)
    with open("results/tokenizer_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved results/tokenizer_report.txt")


if __name__ == "__main__":
    main()
