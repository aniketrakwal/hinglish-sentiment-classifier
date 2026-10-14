"""Independent mojibake check. Run from the project root:

    python src/check_mojibake.py

This script deliberately does NOT import anything from load_data.py, so it cannot share
the parser's blind spots. It brute-forces the repair on every suspicious token instead of
looking for specific marker characters.

For every token in train/val/test CSVs:
  * if it has any character in U+0080-U+00FF or a cp1252 special character (like "€" or "Ÿ"),
  * try to reverse the damage: characters -> cp1252 bytes -> decode as UTF-8,
  * if that WORKS and gives a DIFFERENT string, the token is a remaining mojibake candidate.
Also checks that token count == lang_tags count for every row.
"""

import pandas as pd

SPLITS = ["train", "val", "test"]

# Characters cp1252 shows for bytes 0x80-0x9F (€ ‚ ƒ „ … † ‡ ˆ ‰ Š ‹ Œ Ž ‘ ’ “ ” • – — ˜ ™ š › œ ž Ÿ)
CP1252_SPECIALS = set()
for byte_value in range(0x80, 0xA0):
    try:
        CP1252_SPECIALS.add(bytes([byte_value]).decode("cp1252"))
    except UnicodeDecodeError:
        pass  # undefined byte; its latin-1 control character is already in U+0080-U+00FF


def is_suspicious(token):
    """True if the token has a character that mojibake could be made of."""
    for ch in token:
        if 0x80 <= ord(ch) <= 0xFF or ch in CP1252_SPECIALS:
            return True
    return False


def try_reverse(token):
    """Reverse the damage once. Returns the repaired string, or None if impossible."""
    raw_bytes = bytearray()
    for ch in token:
        if ch in CP1252_SPECIALS:
            raw_bytes += ch.encode("cp1252")
        elif ord(ch) <= 0xFF:
            raw_bytes.append(ord(ch))  # latin-1: character number == byte value
        else:
            return None  # a character above U+00FF can't come from a single byte
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None


print("1. Remaining mojibake candidates (reversal works AND changes the token)")
total_candidates = 0
for split in SPLITS:
    df = pd.read_csv(f"data/processed/{split}.csv", dtype=str, keep_default_na=False)
    candidates = []
    n_suspicious = 0
    for tweet_id, text in zip(df["id"], df["text"]):
        for token in text.split(" "):  # tokens are joined by a plain space in the CSV
            if not is_suspicious(token):
                continue
            n_suspicious += 1
            repaired = try_reverse(token)
            if repaired is not None and repaired != token:
                candidates.append((tweet_id, token, repaired))
    total_candidates += len(candidates)
    print(f"   {split}: {len(candidates)} candidates ({n_suspicious} suspicious tokens scanned)")
    for tweet_id, token, repaired in candidates[:10]:
        print(f"      id {tweet_id}: {ascii(token)} -> {repaired!r}")

print("\n2. Rows where token count != lang_tags count")
for split in SPLITS:
    df = pd.read_csv(f"data/processed/{split}.csv", dtype=str, keep_default_na=False)
    # Split on a plain space (how the parser joined them) ...
    by_space = (df["text"].str.split(" ").str.len() != df["lang_tags"].str.split(" ").str.len()).sum()
    # ... and on any whitespace (how later code using str.split() would see them)
    by_any_ws = (df["text"].str.split().str.len() != df["lang_tags"].str.split().str.len()).sum()
    print(f"   {split}: {by_space} of {len(df)} misaligned (plain space), {by_any_ws} (any whitespace)")

print("\nRESULT:", "OK, no candidates" if total_candidates == 0 else f"{total_candidates} candidates remain")
