"""E1/E2/E3 are scored offline through the guarded scorer; the notebook's number is a
convenience figure.

The load-bearing claim in PREREGISTRATION 3af is that the two conventions are IDENTICAL
on test_3000 and diverge only on dev_2000. That claim is demonstrated here, in both
directions, rather than asserted in prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.score_tier0 import (
    notebook_convention_macro_f1,
    score_from_logits,
    score_npz,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"C{i:03d}" for i in range(100)]


def logits_for(labels: np.ndarray, *, n_classes: int = 100, sharpness: float = 2.0,
               seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lg = rng.normal(0, 1, (len(labels), n_classes))
    lg[np.arange(len(labels)), labels] += sharpness
    return lg


def test_manifests_state_the_coverage_the_equivalence_depends_on():
    """The whole argument rests on test_3000 covering 100 classes and dev_2000 covering
    99. Read it from the committed manifests, so a manifest change breaks this test."""
    def cov(name):
        return json.loads((ROOT / "configs" / "manifests" / f"{name}.json").read_text())
    assert cov("test_3000")["classes_represented"] == 100
    assert cov("test_3000")["min_class_count"] >= 1
    assert cov("train_holdout_3000")["classes_represented"] == 100
    assert cov("dev_2000")["classes_represented"] == 99
    assert cov("dev_2000")["min_class_count"] == 0


def test_conventions_are_identical_when_all_classes_are_present():
    """test_3000's case. Every class appears in gold, so nothing is averaged in at 0.0
    by either route and the two numbers coincide exactly."""
    rng = np.random.default_rng(1)
    labels = np.concatenate([np.arange(100), rng.integers(0, 100, 2900)])  # all 100 present
    lg = logits_for(labels)
    auth = score_from_logits(lg, labels, NAMES, split="test_3000", precision="fp32")
    assert auth.report.classes_averaged == 100
    # Equal to floating-point precision, NOT bit-identical: the two routes average the
    # same 100 per-class F1 values in a different ORDER (sorted(set(gold)) vs the label
    # list), so the sums round differently in the last ulp. Arithmetically identical;
    # 1e-12 is ~4 orders of magnitude tighter than any difference that could matter to a
    # macro-F1 reported to 4 decimals.
    assert auth.report.macro_f1 == pytest.approx(
        notebook_convention_macro_f1(lg, labels, NAMES), abs=1e-12)


def test_conventions_still_agree_when_a_class_is_absent_from_gold():
    """dev_2000's case, DEFAULT label space.

    Both routes default to the classes present in gold, so they agree even at 99/100 —
    the divergence is NOT caused by absence alone. What changes is `classes_averaged`,
    which the notebook never reports and which the authoritative scorer records.
    """
    rng = np.random.default_rng(2)
    labels = np.concatenate([np.arange(99), rng.integers(0, 99, 1901)])   # class 99 absent
    lg = logits_for(labels)
    auth = score_from_logits(lg, labels, NAMES, split="dev_2000", precision="fp32")
    assert auth.report.classes_averaged == 99
    assert auth.report.macro_f1 == pytest.approx(
        notebook_convention_macro_f1(lg, labels, NAMES), abs=1e-12)


def test_the_divergence_appears_only_over_the_FULL_label_space():
    """Averaging dev_2000 over all 100 classes deflates macro-F1 by the absent class.

    This is the failure 3e instance 3 records, and it is exactly the case a router
    threshold calibrated on dev would meet if someone averaged over the full space.
    """
    rng = np.random.default_rng(3)
    labels = np.concatenate([np.arange(99), rng.integers(0, 99, 1901)])
    lg = logits_for(labels)
    present = score_from_logits(lg, labels, NAMES, split="dev_2000", precision="fp32")
    full = score_from_logits(lg, labels, NAMES, split="dev_2000", precision="fp32",
                             over_full_label_space=True)
    assert full.report.classes_averaged == 100
    assert full.report.macro_f1 < present.report.macro_f1
    # deflation is exactly the 99/100 ratio: one class contributes 0.0
    assert full.report.macro_f1 == pytest.approx(present.report.macro_f1 * 99 / 100,
                                                 abs=1e-12)
    assert full.allow_absent_classes is True and present.allow_absent_classes is False


def test_full_label_space_requires_the_flag_and_never_defaults_it():
    """`score` must refuse rather than silently averaging a 0.0 in (hard rule 11)."""
    from src.eval.metrics import score
    labels = np.concatenate([np.arange(99), np.zeros(101, int)])
    gold = [NAMES[i] for i in labels]
    with pytest.raises(ValueError):
        score(gold, gold, labels=NAMES)          # no allow_absent_classes
    score(gold, gold, labels=NAMES, allow_absent_classes=True)   # explicit: fine


def test_e1_headline_is_unaffected_by_the_convention_change():
    """E1's rule is macro-F1 on test_3000 measured on INT8. Since test_3000 is 100/100,
    switching to the authoritative scorer cannot move that number."""
    rng = np.random.default_rng(4)
    labels = np.concatenate([np.arange(100), rng.integers(0, 100, 2900)])
    lg = logits_for(labels, sharpness=4.0)
    auth = score_from_logits(lg, labels, NAMES, split="test_3000", precision="int8")
    assert auth.report.macro_f1 == pytest.approx(
        notebook_convention_macro_f1(lg, labels, NAMES), abs=1e-12)
    assert auth.report.classes_averaged == 100


def test_npz_int8_is_scored_against_fp32_labels_with_a_length_check(tmp_path):
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 100, 300)
    p = tmp_path / "l.npz"
    np.savez_compressed(p, test_logits=logits_for(labels), test_labels=labels,
                        test_3000_logits=logits_for(labels, seed=9))
    s = score_npz(p, NAMES, split="test_3000", precision="int8")
    assert s.precision == "int8" and s.report.n == 300

    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, test_logits=logits_for(labels), test_labels=labels,
                        test_3000_logits=logits_for(labels[:100], seed=9))
    with pytest.raises(ValueError, match="same rows in the same order"):
        score_npz(bad, NAMES, split="test_3000", precision="int8")


def test_scores_carry_their_provenance():
    rng = np.random.default_rng(6)
    labels = np.concatenate([np.arange(100), rng.integers(0, 100, 900)])
    d = score_from_logits(logits_for(labels), labels, NAMES,
                          split="test_3000", precision="int8").as_dict()
    assert d["scored_by"].endswith("(AUTHORITATIVE)")
    assert d["classes_averaged"] == 100 and d["n_classes_in_label_space"] == 100
    assert d["allow_absent_classes"] is False
