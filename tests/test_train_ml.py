"""Smoke test for src/train_ml.py: train on 50 tiny fake rows and check the predictions are valid."""

import os
import sys

# Make `src/` importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_ml import LABELS, SETUPS, make_pipeline, top_features  # noqa: E402

# Each label gets its own small vocabulary; words repeat so that min_df=2 keeps them.
FAKE = {
    "positive": "bahut accha movie hai \U0001F60D love it",
    "negative": "bakwas film nahi dekhna hate it \U0001F621",
    "neutral": "kal match hai bhai ok then",
}


def fake_rows(n=50):
    labels = [LABELS[i % 3] for i in range(n)]
    return [FAKE[lab] for lab in labels], labels


def test_each_setup_trains_and_predicts_valid_labels():
    texts, labels = fake_rows(50)
    for setup in SETUPS:
        model = make_pipeline(setup, C=1)
        model.fit(texts, labels)
        predictions = model.predict(texts)
        assert len(predictions) == 50
        assert set(predictions) <= set(LABELS), setup


def test_probabilities_sum_to_one_and_model_learns_fake_data():
    texts, labels = fake_rows(50)
    model = make_pipeline("word (1,2)", C=4).fit(texts, labels)
    probs = model.predict_proba(texts)
    assert probs.shape == (50, 3)
    assert abs(probs.sum(axis=1) - 1).max() < 1e-6
    assert (model.predict(texts) == labels).mean() > 0.9   # the fake data is trivially separable


def test_top_features_lists_every_class():
    texts, labels = fake_rows(50)
    report = top_features(make_pipeline("word + char", C=1).fit(texts, labels), n=5)
    for lab in LABELS:
        assert f"=== {lab}" in report
