"""E5 routing analysis: signals, calibration, the join, and the Pareto frontier.

No API calls, no GPU, no real logits — E5 must be runnable and testable the moment
Tier 0's .npz lands, which means every property it depends on is pinned here first.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.pareto import cascade_points, oracle_point, pareto_frontier
from src.router.calibrate import (
    SplitMisuseError,
    assert_threshold_split,
    auroc,
    compare_signals,
    sweep_thresholds,
)
from src.router.join import join_by_row_index, parse_row_index
from src.router.signals import CLOSED_FORM, margin, max_softmax, neg_entropy, softmax


@pytest.fixture
def synthetic():
    """Logits that are confident when right and unsure when wrong — so a routing signal
    has something real to find, and an AUROC well above the null is meaningful."""
    rng = np.random.default_rng(20260908)
    n, c = 2000, 100
    labels = rng.integers(0, c, n)
    logits = rng.normal(0, 1.0, (n, c))
    logits[np.arange(n), labels] += rng.normal(3.0, 1.5, n)
    return logits, labels


# ---------------------------------------------------------------- signals


def test_every_signal_is_oriented_high_equals_confident():
    """The convention the whole module depends on. A flipped sign turns AUROC x into
    1 - x, which is a plausible number and would not look like a bug."""
    confident = np.array([[10.0, 0.0, 0.0]])
    unsure = np.array([[0.1, 0.0, -0.1]])
    for name, fn in CLOSED_FORM.items():
        assert fn(confident)[0] > fn(unsure)[0], f"{name} is inverted"


def test_softmax_rows_sum_to_one_and_are_shift_invariant():
    lg = np.array([[1.0, 2.0, 3.0], [-5.0, -5.0, -5.0]])
    assert np.allclose(softmax(lg).sum(axis=-1), 1.0)
    assert np.allclose(softmax(lg), softmax(lg + 100.0))


def test_margin_and_max_softmax_are_not_the_same_signal():
    """Both peak on confident rows, but they disagree on a two-way tie at high mass —
    which is why E5 reports them separately instead of assuming one stands in."""
    two_way = np.array([[5.0, 5.0, 0.0]])
    assert max_softmax(two_way)[0] > 0.4      # high absolute confidence
    assert margin(two_way)[0] < 1e-9          # but no decisiveness at all


def test_neg_entropy_is_maximal_for_a_point_mass():
    assert neg_entropy(np.array([[100.0, 0.0, 0.0]]))[0] > -1e-6
    uniform = neg_entropy(np.array([[0.0, 0.0, 0.0]]))[0]
    assert np.isclose(uniform, -np.log(3), atol=1e-9)


# ---------------------------------------------------------------- AUROC


def test_auroc_perfect_and_inverted_and_random():
    correct = np.array([True] * 50 + [False] * 50)
    perfect = np.where(correct, 1.0, 0.0)
    assert auroc(perfect, correct) == pytest.approx(1.0)
    assert auroc(-perfect, correct) == pytest.approx(0.0)
    assert auroc(np.zeros(100), correct) == pytest.approx(0.5)  # all ties


def test_auroc_refuses_a_single_class_rather_than_returning_half():
    """0.5 for an undefined AUROC is a plausible number for a quantity that does not
    exist — hard rule 11. It must raise."""
    with pytest.raises(ValueError, match="undefined"):
        auroc(np.arange(10.0), np.ones(10, bool))


# ---------------------------------------------------------------- hard rule 1


@pytest.mark.parametrize("bad", ["test_3000", "test", "test_stratified_764"])
def test_calibration_refuses_reporting_splits(bad):
    with pytest.raises(SplitMisuseError, match="REPORTING"):
        assert_threshold_split(bad)


@pytest.mark.parametrize("bad", ["train_holdout_3000", "train"])
def test_calibration_refuses_the_selection_split(bad):
    with pytest.raises(SplitMisuseError, match="MODEL-SELECTION"):
        assert_threshold_split(bad)


def test_calibration_refuses_an_unknown_split():
    with pytest.raises(SplitMisuseError, match="unknown split"):
        assert_threshold_split("dev_5000")


def test_sweep_and_compare_refuse_test_data(synthetic):
    logits, labels = synthetic
    with pytest.raises(SplitMisuseError):
        compare_signals(logits, labels, split="test_3000")
    with pytest.raises(SplitMisuseError):
        sweep_thresholds(logits, labels, signal="margin", split="test_3000")


# ---------------------------------------------------------------- comparison


def test_signals_beat_the_random_null_and_the_null_is_centred_on_half(synthetic):
    logits, labels = synthetic
    r = compare_signals(logits, labels, split="dev_2000", n_null=200)
    assert r.random_null == pytest.approx(0.5, abs=0.02)
    lo, hi = r.random_null_ci95
    assert lo < 0.5 < hi
    for name, a in r.aurocs.items():
        assert a > hi, f"{name} AUROC {a:.4f} is inside the random-escalation null"
    assert r.oracle == 1.0
    assert set(r.aurocs) == {"max_softmax", "margin", "neg_entropy", "trained_difficulty"}


def test_trained_predictor_is_cross_fitted_not_in_sample(synthetic):
    """An in-sample AUROC for the learned signal against out-of-sample AUROCs for the
    closed-form ones would flatter the learner for reasons unrelated to routing. If this
    ever regresses to in-sample fitting the trained score jumps well above the others on
    pure noise, which is what this detects."""
    rng = np.random.default_rng(7)
    n, c = 1500, 20
    labels = rng.integers(0, c, n)
    noise = rng.normal(0, 1, (n, c))           # logits carry NO information about labels
    r = compare_signals(noise, labels, split="dev_2000", n_null=100)
    lo, hi = r.random_null_ci95
    assert r.aurocs["trained_difficulty"] < hi + 0.08, (
        f"trained AUROC {r.aurocs['trained_difficulty']:.3f} on pure noise — the "
        f"predictor is being scored in-sample"
    )


def test_sweep_is_monotone_in_escalation_and_spans_the_range(synthetic):
    logits, labels = synthetic
    s = sweep_thresholds(logits, labels, signal="margin", split="dev_2000")
    assert np.all(np.diff(s.escalation_rate) >= -1e-12)
    assert s.escalation_rate[0] == pytest.approx(0.0, abs=1e-9)
    assert s.escalation_rate[-1] > 0.95
    assert 0.0 <= s.threshold_for_escalation(0.30) <= 1.0


def test_retained_accuracy_rises_as_more_is_escalated(synthetic):
    """The premise of routing: the rows a signal keeps should be the ones Tier 0 gets
    right. If retained accuracy did not rise with escalation the signal is worthless,
    whatever its AUROC."""
    logits, labels = synthetic
    s = sweep_thresholds(logits, labels, signal="max_softmax", split="dev_2000")
    ok = ~np.isnan(s.retained_accuracy)
    assert s.retained_accuracy[ok][-1] > s.retained_accuracy[ok][0] + 0.05


# ---------------------------------------------------------------- the join


def test_parse_row_index_matches_the_real_custom_id_format():
    assert parse_row_index("s1h_test_3000-000001") == 1
    assert parse_row_index("s1s_test_3000-009992") == 9992
    with pytest.raises(ValueError, match="does not match"):
        parse_row_index("nonsense")


def test_join_is_by_row_index_not_list_position():
    """The bug this module exists to prevent. Batch results come back in ANY order, so a
    position join pairs Tier 0 row k with whichever result happened to arrive k-th."""
    idx = np.array([1, 2, 4, 9992])
    logits = np.arange(16, dtype=float).reshape(4, 4)
    api = {                                     # deliberately shuffled
        "s1h_test_3000-009992": {"text": "D"},
        "s1h_test_3000-000002": {"text": "B"},
        "s1h_test_3000-000004": {"text": "C"},
        "s1h_test_3000-000001": {"text": "A"},
    }
    j = join_by_row_index(logits=logits, labels=np.zeros(4, int),
                          logit_row_indices=idx, api_rows=api)
    assert j.api_text == ["A", "B", "C", "D"], "join followed arrival order, not index"


def test_join_refuses_a_partial_match():
    idx = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="not one-to-one"):
        join_by_row_index(logits=np.zeros((3, 2)), labels=np.zeros(3, int),
                          logit_row_indices=idx,
                          api_rows={"s1h_test_3000-000001": {"text": "A"}})


def test_join_refuses_duplicate_row_indices():
    with pytest.raises(ValueError, match="appears twice"):
        join_by_row_index(logits=np.zeros((1, 2)), labels=np.zeros(1, int),
                          logit_row_indices=np.array([1]),
                          api_rows={"s1h_test_3000-000001": {"text": "A"},
                                    "s1s_test_3000-000001": {"text": "B"}})


# ---------------------------------------------------------------- Pareto


@pytest.fixture
def cascade():
    rng = np.random.default_rng(3)
    n = 3000
    t0 = rng.random(n) < 0.82
    api = rng.random(n) < 0.93
    score = np.where(t0, rng.normal(1.0, 0.5, n), rng.normal(0.0, 0.5, n))
    return score, t0, api


def test_cost_is_tier0_plus_api_only_for_escalated_rows(cascade):
    score, t0, api = cascade
    pts = cascade_points(score=score, tier0_correct=t0, api_correct=api,
                         tier0_usd_per_1k=0.0002, api_usd_per_1k=0.4483)
    assert pts[0].escalation_rate == pytest.approx(0.0, abs=1e-9)
    assert pts[0].usd_per_1k == pytest.approx(0.0002)
    full = max(pts, key=lambda p: p.escalation_rate)
    assert full.usd_per_1k == pytest.approx(0.0002 + 0.4483 * full.escalation_rate, rel=1e-9)


def test_frontier_is_non_dominated_and_monotone(cascade):
    score, t0, api = cascade
    fr = pareto_frontier(cascade_points(score=score, tier0_correct=t0, api_correct=api,
                                        tier0_usd_per_1k=0.0002, api_usd_per_1k=0.4483))
    costs = [p.usd_per_1k for p in fr]
    accs = [p.accuracy for p in fr]
    assert costs == sorted(costs)
    assert accs == sorted(accs), "a frontier point costs more without buying accuracy"


def test_oracle_dominates_every_real_operating_point(cascade):
    """The oracle needs the labels, so nothing achievable may beat it. If a real signal
    ever did, the escalation logic and the oracle disagree about what escalation means."""
    score, t0, api = cascade
    o = oracle_point(tier0_correct=t0, api_correct=api,
                     tier0_usd_per_1k=0.0002, api_usd_per_1k=0.4483)
    for p in cascade_points(score=score, tier0_correct=t0, api_correct=api,
                            tier0_usd_per_1k=0.0002, api_usd_per_1k=0.4483):
        if p.usd_per_1k <= o.usd_per_1k + 1e-12:
            assert p.accuracy <= o.accuracy + 1e-9


def test_cascade_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="SAME rows"):
        cascade_points(score=np.zeros(5), tier0_correct=np.zeros(5, bool),
                       api_correct=np.zeros(4, bool),
                       tier0_usd_per_1k=0.0, api_usd_per_1k=1.0)


# ---------------------------------------------------------------- npz loading


import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from src.router.load_logits import load_split, macro_f1_kwargs  # noqa: E402


@pytest.fixture
def npz(tmp_path):
    rng = np.random.default_rng(11)
    n, c = 2000, 100
    p = tmp_path / "logits_ce_seed1.npz"
    np.savez_compressed(
        p,
        dev_logits=rng.normal(size=(n, c)), dev_labels=rng.integers(0, c, n),
        dev_2000_indices=np.arange(n), dev_absent_classes=np.array([14]),
        sel_logits=rng.normal(size=(30, c)), sel_labels=rng.integers(0, c, 30),
        test_logits=rng.normal(size=(50, c)), test_labels=rng.integers(0, c, 50),
        test_3000_indices=np.arange(50),
    )
    return p


def test_loading_dev_for_calibration_is_allowed(npz):
    s = load_split(npz, "dev_2000", for_calibration=True)
    assert s.logits.shape == (2000, 100)
    assert list(s.absent_classes) == [14]
    assert not s.covers_all_classes


def test_loading_test_for_calibration_is_refused_at_the_loader(npz):
    """Hard rule 1 enforced on the way IN, not only at the calibration call. A caller
    that never consults the guard still cannot fit a threshold to test data."""
    with pytest.raises(SplitMisuseError, match="REPORTING"):
        load_split(npz, "test_3000", for_calibration=True)


def test_allow_absent_classes_is_returned_only_for_the_split_that_needs_it(npz):
    """`Books` is absent from dev_2000 by design, so macro-F1 there must be told to
    allow it. test_3000 covers all 100, so the flag must NOT be handed over — carrying it
    across would silence a real coverage problem on the reporting split."""
    assert macro_f1_kwargs(load_split(npz, "dev_2000")) == {"allow_absent_classes": True}
    assert macro_f1_kwargs(load_split(npz, "test_3000")) == {}


def test_a_pre_dev_npz_is_refused_with_an_actionable_message(tmp_path):
    """Tier 0 runs before 2026-09-08 wrote no dev logits. Calibrating on whatever else
    happens to be in the file would be silent misuse of the wrong split."""
    p = tmp_path / "old.npz"
    np.savez_compressed(p, sel_logits=np.zeros((2, 100)), sel_labels=np.zeros(2))
    with pytest.raises(KeyError, match="predates dev_2000 inference"):
        load_split(p, "dev_2000", for_calibration=True)


def test_dev_macro_f1_is_over_99_classes_not_100(npz):
    """PREREGISTRATION 3a's reporting caveat, pinned so it cannot be forgotten."""
    s = load_split(npz, "dev_2000")
    assert s.n_classes == 100
    assert s.n_classes - len(s.absent_classes) == 99
