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
import re

import pandas as pd


# Characters that typically start a mojibake sequence (UTF-8 bytes shown as cp1252 text).
# For example the emoji bytes F0 9F 98 85 show up as "ð" "Ÿ" "˜" "…".
MOJIBAKE_MARKERS = "ðÃâÂ"


def fix_mojibake_once(token):
    """Do ONE repair pass. Returns the token unchanged if it has no marker or the repair fails."""
    # Only try the repair if the token contains a marker character
    has_marker = False
    for ch in token:
        if ch in MOJIBAKE_MARKERS:
            has_marker = True
    if not has_marker:
        return token

    try:
        # Turn every character back into the byte it came from
        raw_bytes = bytearray()
        for ch in token:
            try:
                raw_bytes += ch.encode("cp1252")   # normal case, e.g. "ð" -> 0xF0
            except UnicodeEncodeError:
                raw_bytes += ch.encode("latin-1")  # bytes cp1252 leaves undefined (0x81, 0x8D, ...)
        # Those bytes are the real UTF-8 encoding of the original text
        return raw_bytes.decode("utf-8")
    except UnicodeError:
        # Not valid mojibake after all (or contains characters above 255): keep original
        return token


def fix_mojibake(token):
    """Repair a token whose UTF-8 bytes were wrongly decoded as Windows-1252.

    Example: "ðŸ˜…" -> "😅". Some tokens were damaged twice (e.g. "â€¢" damaged again),
    so we repeat the repair until nothing changes (at most 3 passes).
    Returns the token unchanged if it does not look broken or the repair fails.
    """
    for _ in range(3):
        fixed = fix_mojibake_once(token)
        if fixed == token:  # nothing changed, so we are done
            break
        token = fixed
    return token


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
            # Strip only plain whitespace (space, tab, newline). A bare .strip() would also
            # remove characters like the non-breaking space U+00A0, which can be part of a token.
            line = raw_line.strip(" \t\r\n")

            # Blank line = end of the current tweet
            if not line:
                finish(current)
                current = None
                continue

            # Split on tabs and plain spaces only (not .split(), which also splits on U+00A0)
            parts = [p for p in re.split(r"[ \t]+", line) if p]

            if parts[0] == "meta":
                # A new meta line also ends any tweet that had no blank line after it
                finish(current)
                tweet_id = parts[1] if len(parts) > 1 else ""
                label = parts[2].lower() if len(parts) > 2 else ""  # missing label -> ""
                current = {"id": tweet_id, "label": label, "tokens": [], "tags": []}
            elif current is not None:
                # Token line: first item is the word, last item is the language tag
                # (a lone token with no tag gets the tag "O" = other)
                token = fix_mojibake(parts[0])  # one token in, one token out: tags stay aligned
                tag = parts[1] if len(parts) > 1 else "O"
                # A token that is only whitespace (e.g. a repaired non-breaking space) carries no
                # information and would break later .split() calls, so drop it WITH its tag.
                if token.strip() == "":
                    continue
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
        unmatched = df[df["label"] == ""]
        if len(unmatched) > 0:
            print(f"WARNING: {len(unmatched)} {name} tweets have no matching label in {labels_path}")
            print("  first unmatched ids:", list(unmatched["id"].head(5)))
        else:
            print(f"All {len(df)} {name} tweets matched a label.")

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
