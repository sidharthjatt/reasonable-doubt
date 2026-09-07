"""Metrics: macro-F1 leads, and unmatched outputs count as wrong."""

import pytest

from src.eval.metrics import score

LABELS = ["A", "B", "C"]


def test_perfect_predictions():
    r = score(["A", "B", "C"], ["A", "B", "C"], labels=LABELS)
    assert r.macro_f1 == 1.0 and r.accuracy == 1.0
    assert r.n_unmatched == 0


def test_unmatched_counts_as_wrong_not_dropped():
    """Dropping unmatched outputs would flatter the model by removing its worst
    failures from the denominator."""
    r = score(["A", "B"], ["A", None], labels=LABELS)
    assert r.n == 2  # denominator intact
    assert r.accuracy == 0.5
    assert r.n_unmatched == 1
    assert r.format_failure_rate == 0.5


def test_unmatched_can_never_be_scored_correct():
    r = score(["A", "A"], [None, None], labels=LABELS)
    assert r.accuracy == 0.0
    assert r.macro_f1 == 0.0


def test_macro_f1_is_not_accuracy_on_an_imbalanced_split():
    """The whole reason macro-F1 leads: a head-only predictor scores well on accuracy."""
    gold = ["A"] * 90 + ["B"] * 5 + ["C"] * 5
    pred = ["A"] * 100
    r = score(gold, pred, labels=LABELS)
    assert r.accuracy == 0.9
    assert r.macro_f1 < 0.35


def test_averaging_set_is_reported():
    r = score(["A", "B"], ["A", "B"], labels=LABELS)
    assert r.classes_in_gold == 2  # averaged over 3, only 2 present — must be stated
    assert set(r.per_class_f1) == {"A", "B", "C"}


def test_defaults_to_classes_present_in_gold():
    r = score(["A", "B"], ["A", "B"])
    assert set(r.per_class_f1) == {"A", "B"}


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="predictions"):
        score(["A", "B"], ["A"])


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="nothing to score"):
        score([], [])


def test_render_leads_with_macro_f1():
    text = score(["A"], ["A"], labels=LABELS).render()
    assert text.splitlines()[0].strip().startswith("macro-F1")
    assert "primary metric" in text
