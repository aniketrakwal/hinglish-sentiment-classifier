"""Parse SemEval-2020 Task 9 (SentiMix) Hinglish files into CSVs.

Input format (word-per-line), one tweet per block, blocks separated by a blank line:

    meta<TAB>1234<TAB>positive        <- tweet id and sentiment label
    yaar<TAB>Hin                      <- token<TAB>language_tag
    this<TAB>Eng
    movie<TAB>Eng
    <blank line>

The test file may have no label: "meta<TAB>1234".

Output CSV columns: id, text, label, lang_tags (lang_tags is space-separated).

Usage:
    python src/load_data.py --train data/raw/train.txt --val data/raw/val.txt --test data/raw/test.txt
Any of the three arguments may be left out. If the test labels live in a separate
CSV (columns Uid,Sentiment), add: --test_labels data/raw/test_labels.txt
"""

import argparse
import os

import pandas as pd


def parse_file(path):
    """Read one SentiMix file and return a list of dicts (one per tweet)."""
    tweets = []
    current = None  # the tweet we are currently building; None between tweets

    def finish(tweet):
        """Turn the collected tokens into the final row and store it."""
        if tweet is not None and tweet["tokens"]:
            tweets.append({
                "id": tweet["id"],
                "text": " ".join(tweet["tokens"]),
                "label": tweet["label"],
                "lang_tags": " ".join(tweet["tags"]),
            })

    # utf-8-sig silently strips a BOM if the file has one
    with open(path, encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Blank line = end of the current tweet
            if not line:
                finish(current)
                current = None
                continue

            # Split on tabs or any whitespace, so extra spaces don't break us
            parts = line.split()

            if parts[0] == "meta":
                # A new meta line also ends any tweet that had no blank line after it
                finish(current)
                tweet_id = parts[1] if len(parts) > 1 else ""
                label = parts[2].lower() if len(parts) > 2 else ""  # missing label -> ""
                current = {"id": tweet_id, "label": label, "tokens": [], "tags": []}
            elif current is not None:
                # Token line: first item is the word, last item is the language tag
                # (a lone token with no tag gets the tag "O" = other)
                token = parts[0]
                tag = parts[1] if len(parts) > 1 else "O"
                current["tokens"].append(token)
                current["tags"].append(tag)
            # Token lines appearing before any meta line are ignored

    finish(current)  # the last tweet may not be followed by a blank line
    return tweets


def process_split(name, in_path, out_dir, labels_path=None):
    """Parse one split, save it as CSV, and print its shape and label counts."""
    df = pd.DataFrame(parse_file(in_path), columns=["id", "text", "label", "lang_tags"])

    # Optional: fill in labels from a separate CSV with columns Uid,Sentiment
    if labels_path:
        labels = pd.read_csv(labels_path, dtype=str)
        id_to_label = dict(zip(labels["Uid"].str.strip(), labels["Sentiment"].str.strip().str.lower()))
        df["label"] = df["id"].map(id_to_label).fillna("")
        print(f"{(df['label'] == '').sum()} tweets in {name} had no label in {labels_path}")

    out_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n=== {name} ({in_path}) ===")
    print("shape:", df.shape)
    # Label counts; empty labels (e.g. unlabeled test set) are shown as "<missing>"
    print(df["label"].replace("", "<missing>").value_counts())
    print("saved to", out_path)
    return df


def main():
    parser = argparse.ArgumentParser(description="Parse SentiMix files into CSVs.")
    parser.add_argument("--train", help="path to the raw train file")
    parser.add_argument("--val", help="path to the raw validation file")
    parser.add_argument("--test", help="path to the raw test file")
    parser.add_argument("--test_labels", help="optional CSV (Uid,Sentiment) with the test labels")
    parser.add_argument("--out_dir", default="data/processed", help="where to write the CSVs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    splits = {"train": args.train, "val": args.val, "test": args.test}
    if not any(splits.values()):
        parser.error("give at least one of --train, --val, --test")

    for name, path in splits.items():
        if path:
            labels_path = args.test_labels if name == "test" else None
            process_split(name, path, args.out_dir, labels_path)


if __name__ == "__main__":
    main()
