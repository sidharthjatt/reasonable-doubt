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


def _realistic_npz(tmp_path, name="l.npz", *, n_full=10_000, n_split=3000, seed=5):
    """The REAL Tier 0 npz layout: `test_logits` is the FULL test split, and
    `test_3000_indices` selects the manifest rows out of it. INT8 arrays are already
    stored at the split's size."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 100, n_full)
    idx = np.sort(rng.choice(n_full, size=n_split, replace=False))
    p = tmp_path / name
    np.savez_compressed(p, test_logits=logits_for(labels), test_labels=labels,
                        test_3000_indices=idx,
                        test_3000_logits=logits_for(labels[idx], seed=9))
    return p, labels, idx


def test_npz_int8_is_scored_against_fp32_labels_with_a_length_check(tmp_path):
    p, labels, idx = _realistic_npz(tmp_path)
    s = score_npz(p, NAMES, split="test_3000", precision="int8")
    assert s.precision == "int8" and s.report.n == 3000

    rng = np.random.default_rng(5)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, test_logits=logits_for(labels), test_labels=labels,
                        test_3000_indices=idx,
                        test_3000_logits=logits_for(labels[idx][:100], seed=9))
    with pytest.raises(ValueError, match="same rows in the same order"):
        score_npz(bad, NAMES, split="test_3000", precision="int8")


# ------------------------------------------- the 10k-vs-3000 mix-up cannot recur


def test_fp32_test_3000_is_subset_by_indices_not_scored_whole(tmp_path):
    """`test_logits` is the FULL 10,000-row test split in EVERY Tier 0 npz. Scoring it
    whole and labelling it test_3000 gives a plausible wrong number — on the real
    logits_ce_seed1.npz it was 0.766747 over n=10,000 against 0.763591 over n=3,000."""
    p, labels, idx = _realistic_npz(tmp_path)
    s = score_npz(p, NAMES, split="test_3000", precision="fp32")
    assert s.report.n == 3000, "fp32 test_3000 must be the manifest rows, not all 10,000"

    whole = score_from_logits(logits_for(labels), labels, NAMES,
                              split="test_3000", precision="fp32")
    assert whole.report.n == 10_000
    assert s.report.macro_f1 != pytest.approx(whole.report.macro_f1, abs=1e-9), \
        "the subset and the whole split must not coincide, or this test proves nothing"


def test_an_oversized_array_with_no_index_is_refused(tmp_path):
    """No silent fallback to scoring whatever is there (hard rule 11)."""
    from src.eval.score_tier0 import SplitRowCountError

    rng = np.random.default_rng(7)
    labels = rng.integers(0, 100, 10_000)
    p = tmp_path / "noidx.npz"
    np.savez_compressed(p, test_logits=logits_for(labels), test_labels=labels)
    with pytest.raises(SplitRowCountError, match="No usable"):
        score_npz(p, NAMES, split="test_3000", precision="fp32")


def test_a_wrong_sized_index_array_is_refused(tmp_path):
    from src.eval.score_tier0 import SplitRowCountError

    rng = np.random.default_rng(8)
    labels = rng.integers(0, 100, 10_000)
    p = tmp_path / "badidx.npz"
    np.savez_compressed(p, test_logits=logits_for(labels), test_labels=labels,
                        test_3000_indices=np.arange(2500))
    with pytest.raises(SplitRowCountError, match="2500 entries"):
        score_npz(p, NAMES, split="test_3000", precision="fp32")


def test_real_npz_matches_the_recorded_test_3000_figure():
    """End to end against a committed artefact and its recorded metric."""
    import json
    from pathlib import Path

    from src.data.labels import load_labels

    root = Path(__file__).resolve().parents[1]
    npz = root / "results" / "logits_ce_seed1.npz"
    rec = root / "results" / "tier0_ce_seed1.json"
    from tests.conftest import require_artifact

    require_artifact(npz, "E1 seed-1 FP32 logits; results/* is gitignored so experiment "
                          "outputs are absent from a clean checkout")
    require_artifact(rec, "E1 seed-1 recorded metrics; results/* is gitignored")

    s = score_npz(npz, load_labels(), split="test_3000", precision="fp32")
    expected = json.loads(rec.read_text())["test_3000_fp32"]["macro_f1"]
    assert s.report.n == 3000
    assert s.report.macro_f1 == pytest.approx(expected, abs=1e-9)


def test_scores_carry_their_provenance():
    rng = np.random.default_rng(6)
    labels = np.concatenate([np.arange(100), rng.integers(0, 100, 900)])
    d = score_from_logits(logits_for(labels), labels, NAMES,
                          split="test_3000", precision="int8").as_dict()
    assert d["scored_by"].endswith("(AUTHORITATIVE)")
    assert d["classes_averaged"] == 100 and d["n_classes_in_label_space"] == 100
    assert d["allow_absent_classes"] is False
