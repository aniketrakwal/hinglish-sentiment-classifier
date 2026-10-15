"""Independent check of clean_text. Run from the project root:

    python src/check_preprocess.py

This script shares NO regular expressions with preprocess.py. It only imports the function
clean_text, then inspects the cleaned text token by token with plain Python (loops and ==),
so it cannot share the patterns' blind spots. Every count below should be 0.
"""

import emoji
import pandas as pd

from preprocess import clean_text

SPLITS = ["train", "val", "test"]


def has_triple_letter(token):
    """True if some letter appears 3+ times in a row (checked with a simple loop)."""
    run = 1
    for i in range(1, len(token)):
        if token[i].isalpha() and token[i] == token[i - 1]:
            run += 1
            if run >= 3:
                return True
        else:
            run = 1
    return False


def is_only_underscores(token):
    return len(token) > 0 and all(ch == "_" for ch in token)


def looks_like_username_start(rest):
    """True if the text right after an "@" starts like a user name (letter, digit or "_" or the
    cut-off mark). Censored swear words such as "@$" or "@&#%$" do NOT, so they are not mentions."""
    return len(rest) > 0 and (rest[0].isalnum() or rest[0] in "_…")


def problems_in(raw, cleaned):
    """Return a dict {check name: True} for every problem found in one cleaned tweet."""
    found = {}
    raw_tokens = raw.split()
    tokens = cleaned.split(" ")                        # tokens are separated by single spaces now

    for i, tok in enumerate(tokens):
        low = tok.lower()
        # --- URL leftovers
        if low.startswith("https"):
            found["token starting with 'https'"] = True
        # "//" is a leftover only right after "http"/"https"; elsewhere it is ordinary text ("// yah")
        if "//" in tok and i > 0 and tokens[i - 1] in ("http", "https"):
            found["'http //' left"] = True
        if tok == "tco" or "t.co" in low:
            found["'tco' / 't.co' token"] = True
        if tok == "t" and tokens[i + 1:i + 3] == [".", "co"]:
            found["'t . co' sequence"] = True
        if tok == "t" and tokens[i + 1:i + 2] == ["co"]:
            found["'t co' sequence"] = True
        # --- mention leftovers
        if tok == "@":
            found["lone '@' token"] = True
        if tok.startswith("@user"):
            # "@user" may be followed by punctuation ("@user.") but not by more name letters
            if looks_like_username_start(tok[len("@user"):]):
                found["'@user' followed by more name characters"] = True
        elif tok.startswith("@") and looks_like_username_start(tok[1:]):
            found["'@' followed by something other than 'user'"] = True
        if tok == "@user" and i + 1 < len(tokens) and tokens[i + 1] == "@user":
            found["'@user @user' not collapsed"] = True
        # --- hashtag / underscore leftovers
        if tok == "#":
            found["lone '#' token"] = True
        if is_only_underscores(tok):
            found["lone '_' token"] = True
        # --- other rules
        if tok != low:
            found["uppercase letter left"] = True
        if has_triple_letter(tok):
            found["letter repeated 3+ times"] = True

    if "️" in cleaned:
        found["U+FE0F present"] = True
    # The leading retweet marker must be gone. (A later "RT", e.g. "# RT @ a", is meant to stay.)
    if raw_tokens and raw_tokens[0] == "RT" and tokens[0] == "rt" and raw_tokens[1:2] not in (["RT"], ["rt"]):
        found["leading 'RT' still there"] = True
    if cleaned != cleaned.strip() or "  " in cleaned:
        found["extra spaces"] = True
    if cleaned == "":
        found["empty result"] = True

    # emoji must have a space (or the text edge) on both sides
    for item in emoji.emoji_list(cleaned):
        before = cleaned[item["match_start"] - 1] if item["match_start"] > 0 else " "
        after = cleaned[item["match_end"]] if item["match_end"] < len(cleaned) else " "
        if before != " " or after != " ":
            found["emoji not surrounded by spaces"] = True

    if clean_text(cleaned) != cleaned:
        found["not idempotent"] = True
    return found


total_problems = 0
for split in SPLITS:
    df = pd.read_csv(f"data/processed/{split}.csv", dtype=str, keep_default_na=False)
    counts = {}
    examples = {}
    for raw in df["text"]:
        cleaned = clean_text(raw)
        for name in problems_in(raw, cleaned):
            counts[name] = counts.get(name, 0) + 1
            if len(examples.setdefault(name, [])) < 3:
                examples[name].append((raw[:90], cleaned[:90]))

    print(f"\n=== {split} ({len(df)} tweets) ===")
    if not counts:
        print("   all checks: 0 problems")
    for name, n in counts.items():
        total_problems += n
        print(f"   {name}: {n}")
        for raw, cleaned in examples[name]:
            print(f"      before: {raw!r}\n      after:  {cleaned!r}")

print("\nRESULT:", "OK, every count is 0" if total_problems == 0 else f"{total_problems} problems found")
