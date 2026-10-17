"""Tests for src/bootstrap.py, src/calibration.py and the helpers of src/val_test_gap.py (tiny fixed examples)."""

import os
import sys

import numpy as np
import pytest
from sklearn.metrics import f1_score

# Make `src/` importable when running pytest from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bootstrap import bootstrap_scores, interval, macro_f1, resample_indices  # noqa: E402
from calibration import (apply_temperature, confidence_split, expected_calibration_error, fit_temperature,  # noqa: E402
                         log_loss, reliability_bins)
from val_test_gap import near_key  # noqa: E402


# ---------------------------------------------------------------- bootstrap
def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    expected = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro")
    assert macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_resample_indices_are_repeatable_and_in_range():
    a, b = resample_indices(10, n_resamples=5, seed=1), resample_indices(10, n_resamples=5, seed=1)
    assert a.shape == (5, 10) and (a == b).all()
    assert a.min() >= 0 and a.max() <= 9


def test_perfect_predictions_give_a_zero_width_interval_at_one():
    y_true = np.array([0, 1, 2] * 10)
    scores = bootstrap_scores(y_true, [y_true.copy()], resample_indices(len(y_true), n_resamples=200))
    assert interval(scores) == (1.0, 1.0)


def test_interval_contains_the_point_estimate_and_has_width():
    y_true = np.array([0, 1, 2] * 20)
    y_pred = y_true.copy()
    y_pred[:12] = (y_pred[:12] + 1) % 3                      # 12 of 60 tweets wrong
    scores = bootstrap_scores(y_true, [y_pred], resample_indices(len(y_true), n_resamples=1000))
    low, high = interval(scores)
    assert low < macro_f1(y_true, y_pred) < high and high - low > 0.02


def test_paired_difference_of_identical_models_is_zero_and_of_good_vs_bad_excludes_zero():
    y_true = np.array([0, 1, 2] * 20)
    good, bad = y_true.copy(), (y_true + 1) % 3
    indices = resample_indices(len(y_true), n_resamples=300)
    same = bootstrap_scores(y_true, [good], indices) - bootstrap_scores(y_true, [good], indices)
    assert interval(same) == (0.0, 0.0)
    gap = bootstrap_scores(y_true, [good], indices) - bootstrap_scores(y_true, [bad], indices)
    assert interval(gap)[0] > 0                               # the whole interval is above 0


def test_seed_mean_is_used_for_several_seeds():
    y_true = np.array([0, 1, 2] * 10)
    wrong = (y_true + 1) % 3
    scores = bootstrap_scores(y_true, [y_true, wrong], resample_indices(len(y_true), n_resamples=50))
    assert scores == pytest.approx(0.5)                       # (1.0 + 0.0) / 2 on every resample


# ---------------------------------------------------------------- calibration
PROBS = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]])


def test_temperature_scaling_never_changes_the_prediction():
    for temperature in [0.3, 1.0, 2.5, 8.0]:
        assert (apply_temperature(PROBS, temperature).argmax(axis=1) == PROBS.argmax(axis=1)).all()


def test_temperature_one_changes_nothing_and_larger_temperature_lowers_confidence():
    assert apply_temperature(PROBS, 1.0) == pytest.approx(PROBS)
    assert (apply_temperature(PROBS, 3.0).max(axis=1) < PROBS.max(axis=1)).all()
    assert apply_temperature(PROBS, 3.0).sum(axis=1) == pytest.approx(1.0)


def test_fit_temperature_finds_t_above_one_for_an_overconfident_model():
    # 90 of 100 predictions are 99% confident but only 70 are right: over-confident, so T should be > 1
    y_true = np.array([0] * 100)
    probs = np.tile([0.99, 0.005, 0.005], (100, 1))
    probs[30:] = [0.005, 0.99, 0.005]                         # 70 wrong predictions
    probs[:30] = [0.99, 0.005, 0.005]                         # 30 right ones
    temperature = fit_temperature(probs, y_true)
    assert temperature > 1.0
    assert log_loss(apply_temperature(probs, temperature), y_true) < log_loss(probs, y_true)


def test_ece_is_zero_when_confidence_equals_accuracy_and_large_when_not():
    y_true = np.array([0] * 10)
    right = np.tile([0.5, 0.3, 0.2], (10, 1))
    right[5:] = [0.2, 0.5, 0.3]                               # 5 of 10 right, confidence 0.5 everywhere
    assert expected_calibration_error(right, y_true) == pytest.approx(0.0, abs=1e-9)
    sure_but_wrong = np.tile([0.1, 0.9, 0.0], (10, 1))
    assert expected_calibration_error(sure_but_wrong, y_true) == pytest.approx(0.9)


def test_reliability_bins_and_confidence_split():
    y_true = np.array([0, 1, 2, 1])                                   # the first three predictions are right
    count, mean_conf, accuracy = reliability_bins(PROBS, y_true)
    assert count.sum() == 4 and count[9] == 0 and count[7] == 1 and count[8] == 1   # 0.7 and 0.8 -> bins 7 and 8
    assert accuracy[7] == 1.0 and accuracy[5] == 0.0                  # the 0.5-confident tweet is wrong
    correct, wrong = confidence_split(PROBS, y_true)
    assert correct == pytest.approx((0.7 + 0.8 + 0.6) / 3) and wrong == pytest.approx(0.5)


# ---------------------------------------------------------------- val_test_gap helper
def test_near_key_ignores_case_and_urls():
    a = "Wow NICE https // t . co / AbC123xyz bye"
    b = "wow nice https // t . co / ZZZ999 bye"
    assert near_key(a) == near_key(b) == "wow nice bye"
