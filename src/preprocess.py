"""Shared text cleaning for ALL models (TF-IDF, BiLSTM, transformers).

Usage:
    from preprocess import clean_text
    clean_text("RT @ Nehr _ who # Hit https // t . co / AbC123xyz 😔😔")   # -> "@user hit http 😔 😔"

The input is one tweet exactly as stored in data/processed/*.csv: tokens joined by single spaces,
so URLs, mentions and hashtags are split into pieces ("https // t . co / xxx", "@ name", "# word").

We do NOT remove punctuation, stopwords or accents, and we do NOT normalise spellings or apply NFKC.
Every rule is a small function below, applied in the order listed in clean_text().
"""

import re

import emoji

# ---------------------------------------------------------------------------------------------
# Patterns (each defined once, here)
# ---------------------------------------------------------------------------------------------

# Rule 2: a leading "RT" token (one or more), case-sensitive. "RT" must be a whole token, so
# "RTX" or "RT:" do not match, and "rt" inside a sentence is never touched.
RT_PATTERN = re.compile(r"^\s*(?:RT(?:\s+|$))+")

# Rule 3 (first): a link glued to the end of a word ("Yadavhttps // youtu . be / x"). Put a space
# in front of the "http" so the patterns below see a normal link. Only happens before "http(s) //".
GLUED_URL_PATTERN = re.compile(r"(?<=\w)(?=https? //)")

# Rule 3a: any non-twitter link such as "https // www . youtube . com / watch ? v = abc".
# After "http(s) //" comes a word, then pairs of (punctuation token, word). The (?!...) makes it
# skip twitter's own t.co links, which rule 3b handles.
GENERIC_URL_PATTERN = re.compile(
    r"(?<!\w)https? // (?!(?:t|co|tco|tc|c)(?: |$))[\w-]+"   # "https //" + first word (not t / co / tco)
    r"(?: [./:?=&%-]+ [\w-]+)*"                               # more "punctuation word" pairs
    r"(?: [./:?=&%-]+)?(?!\S)"                                # optional trailing punctuation token
)

# Rule 3b: twitter t.co links, complete or cut off. The real data has these shapes:
#   "https // t . co / xxx"   "https // t co / xxx"   "https // tco / xxx"
#   cut-off: "https …"  "https // t …"  "https // t . c …"  "https // t . co /…"  "https //…"
# Built from: the word http/https; then (if the link continues) a first piece that starts with "/"
# or "…"; then any number of pieces (a token made only of / . : … characters, or one of the domain
# words t, co, tco, tc, c); then optionally the random id that follows a "/" token; then any "…"
# cut-off marks. Requiring that first "/" or "…" piece means a plain "http ." is left alone, which
# keeps clean_text(clean_text(x)) == clean_text(x).
# (?<!\w) = "not glued to the end of a word"; an emoji or space before it is fine.
TCO_URL_PATTERN = re.compile(
    r"(?<!\w)https?"
    r"(?:(?: [/…][/.:…]*)"
    r"(?: (?:[/.:…]+|t|co|tco|tc|c))*"
    r"(?:(?<= /) [A-Za-z0-9]+)?"
    r"(?: …)*)?"
    r"(?!\S)"
)

# Rule 4: a mention. Pieces: one or more "@" (a stray "@ @ name" happens), an optional "_" run
# ("@_ name" is the name "_name" split in two), an optional space, the user name, and any number
# of "_ part" pieces for names split at underscores ("@ Nehr _ who"), and finally an optional
# lone "_" token (a name ending in "_"). Swallowing that last "_" lets rule 5 join "@ a _ @ b".
# A mention cut off by twitter ("@…") also counts. Also matches joined "@name". (?<!\w) = not glued to a word, so "a@b" is not a mention but
# "🌸@ name" is (rule 9 only separates emoji later).
MENTION_PATTERN = re.compile(
    r"(?<!\w)(?:@_* ?)+(?:"
    r"[A-Za-z0-9_]+(?: _+ [A-Za-z0-9]+)*(?: _+(?!\S))?"   # the name (+ underscore pieces)
    r"|…)"                                                  # or a cut-off name
)

# Rule 5: two or more "@user" in a row.
# \s+ (not a plain space) because some tweets have a non-breaking space between mentions.
REPEATED_MENTION_PATTERN = re.compile(r"@user(?:\s+@user)+")

# Rule 6: a "#" that is a token on its own ("# HappyBirthday"), and tokens made only of "_".
# The lookbehinds allow an emoji glued in front ("👉# word"), because emoji are only separated in
# rule 9. They still skip "c#", "####" and "@#" (preceded by a word character, "#" or "@").
LONE_HASH_PATTERN = re.compile(r"(?<![\w#@])#(?!\S)")
LONE_UNDERSCORE_PATTERN = re.compile(r"(?<!\w)_+(?!\S)")

# Rule 8: one LETTER (any language; digits and "_" are excluded) followed by itself 2+ more times.
REPEATED_LETTER_PATTERN = re.compile(r"([^\W\d_])\1{2,}")

# Rule 10: any run of whitespace.
WHITESPACE_PATTERN = re.compile(r"\s+")


# ---------------------------------------------------------------------------------------------
# One small function per rule
# ---------------------------------------------------------------------------------------------

def remove_variation_selector(text: str) -> str:
    """Rule 1. U+FE0F is an invisible mark that makes some emoji look 'emoji-style'.
    Some tweets have "❤️" (with it) and others "❤" (without). Removing it makes them identical."""
    return text.replace("️", "")


def remove_leading_rt(text: str) -> str:
    """Rule 2. Drop the retweet marker at the very start. Every retweet starts with it, so it says
    nothing about sentiment. Done BEFORE lowercasing so only the real marker "RT" is removed."""
    return RT_PATTERN.sub("", text)


def replace_urls(text: str) -> str:
    """Rule 3. Replace every link with the single word "http". The random id inside a link is
    meaningless, but 'this tweet has a link' can still be a useful signal. Without this, a link
    would be about 7 junk tokens."""
    text = GLUED_URL_PATTERN.sub(" ", text)
    text = GENERIC_URL_PATTERN.sub("http", text)
    return TCO_URL_PATTERN.sub("http", text)


def replace_mentions(text: str) -> str:
    """Rule 4. Replace every @mention with "@user". The name of the person tagged is not
    sentiment; whether someone was tagged at all is kept."""
    return MENTION_PATTERN.sub("@user", text)


def collapse_repeated_mentions(text: str) -> str:
    """Rule 5. "@user @user @user" becomes one "@user". Reply chains tag many people, and the
    number of people has no meaning for sentiment."""
    return REPEATED_MENTION_PATTERN.sub("@user", text)


def remove_hashtag_symbols(text: str) -> str:
    """Rule 6 (hashtags). "# HappyBirthday" -> "HappyBirthday". The word is useful, the lone "#" is not."""
    return LONE_HASH_PATTERN.sub("", text)


def remove_lone_underscores(text: str) -> str:
    """Rule 6 (underscores). Remove "_" tokens left over from split user names (e.g. a name ending in "_")."""
    return LONE_UNDERSCORE_PATTERN.sub("", text)


def lowercase(text: str) -> str:
    """Rule 7. "Bahut", "BAHUT" and "bahut" become one word, so each word is seen more often."""
    return text.lower()


def shorten_repeated_letters(text: str) -> str:
    """Rule 8. A letter repeated 3+ times is cut to 2 ("bahuttt" -> "bahutt", "sooooo" -> "soo").
    Stretched spellings are very common in tweets. Two letters are kept so a real double letter
    ("hello") is unharmed. Digits are not touched, so "1000" stays "1000"."""
    return REPEATED_LETTER_PATTERN.sub(r"\1\1", text)


def space_out_emoji(text: str) -> str:
    """Rule 9. Put a space on both sides of each emoji, so "😔😔" -> "😔 😔" and "love😍" -> "love 😍".
    Models split text on spaces, so glued emoji would otherwise be one unknown 'word'.
    emoji_list() reports where each WHOLE emoji starts and ends, so multi-character emoji
    (skin tones like 👍🏽, joined emoji like 👨‍👩‍👧, flags like 🇮🇳) are never cut in half."""
    pieces = []
    last_end = 0
    for item in emoji.emoji_list(text):
        pieces.append(text[last_end:item["match_start"]])   # text before this emoji
        pieces.append(" " + item["emoji"] + " ")             # the emoji with a space on each side
        last_end = item["match_end"]
    pieces.append(text[last_end:])                           # text after the last emoji
    return "".join(pieces)


def tidy_spaces(text: str) -> str:
    """Rule 10. Earlier rules leave double spaces and edge spaces; make it single spaces, no edges."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


# ---------------------------------------------------------------------------------------------
# The one function every model imports
# ---------------------------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean one tweet. The order matters (e.g. RT is removed before lowercasing, and URLs go
    before mentions so a URL is never mistaken for a name)."""
    text = remove_variation_selector(text)      # 1
    text = remove_leading_rt(text)              # 2
    text = replace_urls(text)                   # 3
    text = replace_mentions(text)               # 4
    text = collapse_repeated_mentions(text)     # 5
    text = remove_lone_underscores(text)        # 6 (underscores first: "#_ word" must expose its "#")
    text = remove_hashtag_symbols(text)         # 6
    text = lowercase(text)                      # 7
    text = shorten_repeated_letters(text)       # 8
    text = space_out_emoji(text)                # 9
    text = tidy_spaces(text)                    # 10
    return text
