"""Paired bootstrap over ROWS.

Why paired, and why this module exists at all
---------------------------------------------
Two arms evaluated on the *same rows* are not two independent samples. Quoting an
unpaired spread — or, worse, an across-SEED standard deviation — for a same-rows
comparison answers a different question than the one being asked: the across-seed sd
measures how much *retraining* moves the score, while the claim under test is how much
*this change* moves it on *these clauses*. The paired bootstrap resamples the rows and
recomputes BOTH arms on the same resample, so the shared row-draw noise cancels in the
difference instead of being counted twice.

This matters concretely here. The E6 cascade gain of +0.0063 macro-F1 sits at 1.1-1.9x
the across-seed sd (0.0034-0.0056) — not "inside one sd", as was first said — but that
comparison is the wrong yardstick either way, because Tier 0 alone and Tier 0 plus
escalation are scored on identical rows with identical seeds.

Statistic-agnostic by design: the caller passes functions of a row-index array, so the
same machinery serves a two-prompt comparison and a seed-averaged cascade comparison
without either one reimplementing resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["PairedBootstrap", "fast_macro_f1", "paired_bootstrap"]


@dataclass(frozen=True)
class PairedBootstrap:
    """A paired-bootstrap interval on ``stat_b - stat_a``."""

    delta: float
    lo: float
    hi: float
    ci: float
    n_resamples: int
    n_rows: int
    p_sign: float
    """Two-sided bootstrap p: 2x the smaller tail mass on either side of 0."""

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def render(self, name: str = "delta") -> str:
        verdict = "EXCLUDES 0" if self.excludes_zero else "INCLUDES 0"
        return (
            f"  {name:<28}: {self.delta:+.4f}   "
            f"{self.ci:.0%} CI [{self.lo:+.4f}, {self.hi:+.4f}]  {verdict}\n"
            f"  {'':<28}  paired over {self.n_rows} rows, "
            f"{self.n_resamples} resamples, p={self.p_sign:.4f}"
        )


def fast_macro_f1(gold: np.ndarray, pred: np.ndarray, n_labels: int) -> float:
    """Macro-F1 over label ids ``0..n_labels-1``.

    ``pred`` may contain ``-1`` for an output that did not normalize to any label; it can
    never be counted correct but stays in the denominator, matching
    :func:`src.eval.metrics.score`. Averaging is over ALL ``n_labels``, so the caller is
    responsible for passing a label space the gold actually covers — the slow scorer
    raises on absent classes, and :func:`paired_bootstrap` callers must have cleared that
    check there first.

    Exists only because the reference scorer is far too slow for 10,000 resamples; it is
    asserted equal to :func:`src.eval.metrics.score` on the real data before use.
    """
    correct = gold == pred
    tp = np.bincount(gold[correct], minlength=n_labels)
    pred_pos = np.bincount(pred[pred >= 0], minlength=n_labels)
    gold_pos = np.bincount(gold, minlength=n_labels)
    denom = pred_pos + gold_pos
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def paired_bootstrap(
    n_rows: int,
    stat_a: Callable[[np.ndarray], float],
    stat_b: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 10_000,
    seed: int,
    ci: float = 0.95,
) -> PairedBootstrap:
    """Percentile bootstrap on ``stat_b(idx) - stat_a(idx)``, resampling rows.

    Args:
        n_rows: number of rows both arms are evaluated on. The arms MUST be aligned
            row-for-row; this function cannot check that and a misalignment would
            silently produce a confident wrong interval.
        stat_a, stat_b: each maps a row-index array to a scalar. Both are called on the
            SAME resample, which is the whole point.
        seed: required, not defaulted — the interval is a reported number and must be
            reproducible from the recorded seed.

    Raises:
        ValueError: on a non-positive row count or a ci outside (0, 1).
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    rng = np.random.default_rng(seed)
    base = np.arange(n_rows)
    point = stat_b(base) - stat_a(base)

    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n_rows, size=n_rows)
        deltas[i] = stat_b(idx) - stat_a(idx)

    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(deltas, [tail, 1.0 - tail])
    below = float((deltas <= 0.0).mean())
    p = min(1.0, 2.0 * min(below, 1.0 - below))
    return PairedBootstrap(
        delta=float(point),
        lo=float(lo),
        hi=float(hi),
        ci=ci,
        n_resamples=n_resamples,
        n_rows=n_rows,
        p_sign=p,
    )
