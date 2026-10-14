"""Tests for the SentiMix parser, using a tiny fake file."""

import os
import sys

# Make `src/` importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from load_data import fix_mojibake, parse_file  # noqa: E402

# Messy on purpose: extra spaces, repeated blank lines, no trailing blank line
LABELED = (
    "meta\t1\tpositive\n"
    "yaar\tHin\n"
    "movie  \t Eng\n"
    "\n\n"
    "meta\t2\tNegative\n"
    "bakwas\tHin\n"
    "!!!\tO\n"
    "\n"
    "meta\t3\tneutral\n"
    "ok\tEng"
)

# Test-style file: no labels
UNLABELED = "meta\t10\nkya\tHin\nbaat\tHin\n\nmeta\t11\nnice\tEng\n"


def write(tmp_path, content):
    path = tmp_path / "sample.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_parses_all_tweets(tmp_path):
    rows = parse_file(write(tmp_path, LABELED))
    assert [r["id"] for r in rows] == ["1", "2", "3"]


def test_text_labels_and_tags(tmp_path):
    rows = parse_file(write(tmp_path, LABELED))
    assert rows[0]["text"] == "yaar movie"
    assert rows[0]["label"] == "positive"
    assert rows[0]["lang_tags"] == "Hin Eng"
    assert rows[1]["label"] == "negative"  # label is lowercased
    assert rows[2]["text"] == "ok"  # last tweet kept despite no trailing blank line


def test_missing_labels(tmp_path):
    rows = parse_file(write(tmp_path, UNLABELED))
    assert len(rows) == 2
    assert all(r["label"] == "" for r in rows)
    assert rows[0]["text"] == "kya baat"


def test_empty_file(tmp_path):
    assert parse_file(write(tmp_path, "")) == []


def test_fix_mojibake_simple_emoji():
    assert fix_mojibake("ðŸ˜…") == "😅"


def test_fix_mojibake_undefined_cp1252_byte():
    # 😍 is F0 9F 98 8D; byte 0x8D is undefined in cp1252, so it appears as U+008D
    assert fix_mojibake("ðŸ˜\u008d") == "😍"


def test_fix_mojibake_leaves_normal_words_alone():
    for word in ["movie", "yaar", "kya", "@", "!!!", "😅", "😍"]:
        assert fix_mojibake(word) == word


def test_fix_mojibake_leaves_accented_words_alone():
    for word in ["á", "ě", "vypadá", "ů"]:
        assert fix_mojibake(word) == word


def test_fix_mojibake_keeps_unrepairable_token():
    # Has a marker but is not valid mojibake: must come back unchanged
    assert fix_mojibake("ðabc") == "ðabc"


def test_parser_fixes_mojibake_and_keeps_tags_aligned(tmp_path):
    rows = parse_file(write(tmp_path, "meta\t5\tpositive\nnice\tEng\nðŸ˜…\tEMT\n"))
    assert rows[0]["text"] == "nice 😅"
    assert rows[0]["lang_tags"] == "Eng EMT"


def test_parser_handles_mojibake_non_breaking_space(tmp_path):
    # "Â" + U+00A0 is mojibake for a non-breaking space; it must not break tag alignment.
    # The whitespace-only token is dropped together with its tag.
    content = "meta\t6\tneutral\nhi\tEng\nÂ \tO\nbye\tEng\n"
    rows = parse_file(write(tmp_path, content))
    assert rows[0]["text"] == "hi bye"
    assert rows[0]["lang_tags"] == "Eng Eng"


def test_parser_fixes_emoji_ending_in_a0_byte(tmp_path):
    # 😠 is F0 9F 98 A0, so its mojibake ends with U+00A0
    content = "meta\t7\tneutral\nðŸ˜ \tEMT\n"
    rows = parse_file(write(tmp_path, content))
    assert rows[0]["text"] == "😠"


def test_fix_mojibake_double_encoded():
    # "•" damaged twice: once gives "â\u0080¢", twice gives the string below
    assert fix_mojibake("â€¢") == "•"          # single damage ("â€¢")
    assert fix_mojibake("Ã¢Â€Â¢") == "•"  # double damage


def test_fix_mojibake_devanagari():
    # "सईया" damaged: E0 A4 B8 / E0 A4 88 / E0 A4 AF / E0 A4 BE read as cp1252 text
    assert fix_mojibake("à¤¸à¤ˆà¤¯à¤¾") == "सईया"


def test_fix_mojibake_devanagari_built_from_bytes():
    # Same idea, but damage real Devanagari ourselves so no typed characters can be wrong
    for word in ["सईया", "मैं", "कैसे", "भारत"]:
        damaged = word.encode("utf-8").decode("cp1252", errors="replace")
        assert fix_mojibake(damaged) == word


def test_fix_mojibake_leaves_real_devanagari_alone():
    assert fix_mojibake("सईया") == "सईया"


def test_fix_mojibake_leaves_french_a_grave_alone():
    for word in ["à", "voilà", "où", "à"]:
        assert fix_mojibake(word) == word
