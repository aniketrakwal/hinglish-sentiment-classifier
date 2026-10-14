"""Tests for the SentiMix parser, using a tiny fake file."""

import os
import sys

# Make `src/` importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from load_data import parse_file  # noqa: E402

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
