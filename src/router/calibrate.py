"""E5 threshold calibration and signal comparison.

HARD RULE 1: router thresholds are calibrated on the DEV split ONLY, never on test.
That is enforced here in code (`assert_threshold_split`), not left to discipline — the
same split-guard pattern the Kaggle notebooks use, for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.router.signals import CLOSED_FORM, feature_matrix

THRESHOLD_SPLITS = {"dev_2000", "validation", "dev"}
SELECTION_SPLITS = {"train_holdout_3000", "train"}
REPORTING_SPLITS = {"test_3000", "test_stratified_764", "test"}


class SplitMisuseError(RuntimeError):
    pass


def assert_threshold_split(name: str) -> str:
    """Hard rule 1, enforced. Calibration may only ever see the dev split."""
    if name in REPORTING_SPLITS:
        raise SplitMisuseError(
            f"REFUSING to calibrate a threshold on {name!r}: it is a REPORTING split. "
            f"Hard rule 1 — thresholds are calibrated on dev only. Calibrating on test "
            f"and then reporting on it is the result being invented rather than measured."
        )
    if name in SELECTION_SPLITS:
        raise SplitMisuseError(
            f"REFUSING to calibrate a threshold on {name!r}: it is a MODEL-SELECTION "
            f"split. Tier 0 chose its epoch on this data, so its errors here are not "
            f"representative of errors on unseen rows and a threshold fitted to them "
            f"would be optimistic."
        )
    if name not in THRESHOLD_SPLITS:
        raise SplitMisuseError(f"unknown split {name!r}; refusing rather than guessing")
    return name


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """AUROC via the rank (Mann-Whitney U) identity, ties handled by average ranks.

    `positive` marks the class the score should rank HIGH. For routing that is
    "Tier 0 was correct", so AUROC is the probability a randomly chosen correct row
    scores above a randomly chosen wrong one.
    """
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"AUROC undefined with {n_pos} correct and {n_neg} wrong rows — one class is "
            f"empty. Returning 0.5 here would be a plausible number for an undefined "
            f"quantity (hard rule 11)."
        )
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s = score[order]
    i = 0
    while i < len(s):                       # average ranks within tie groups
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass(frozen=True)
class SignalComparison:
    aurocs: dict[str, float]
    random_null: float
    random_null_ci95: tuple[float, float]
    oracle: float
    n: int
    n_correct: int
    split: str


def compare_signals(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    n_folds: int = 5,
    seed: int = 20260908,
    n_null: int = 1000,
) -> SignalComparison:
    """AUROC of every signal, plus the random-escalation null and the oracle bound.

    The trained predictor is scored CROSS-FITTED. An in-sample AUROC for a learned signal
    against out-of-sample AUROCs for closed-form ones would flatter the learner for
    reasons that have nothing to do with routing — the comparison has to be like for
    like, or the headline "learned beats margin" is an artefact of fitting.
    """
    assert_threshold_split(split)
    correct = np.asarray(labels) == np.asarray(logits).argmax(-1)

    aurocs = {name: auroc(fn(logits), correct) for name, fn in CLOSED_FORM.items()}
    aurocs["trained_difficulty"] = _cross_fitted_auroc(
        feature_matrix(logits), correct, n_folds=n_folds, seed=seed
    )

    rng = np.random.default_rng(seed)
    null = np.array([auroc(rng.random(len(correct)), correct) for _ in range(n_null)])
    return SignalComparison(
        aurocs=aurocs,
        random_null=float(null.mean()),
        random_null_ci95=(float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))),
        oracle=1.0,          # a signal equal to `correct` ranks perfectly, by construction
        n=len(correct),
        n_correct=int(correct.sum()),
        split=split,
    )


def _cross_fitted_auroc(X: np.ndarray, correct: np.ndarray, *, n_folds: int, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    oof = np.zeros(len(correct))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, correct):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, random_state=seed))
        model.fit(X[tr], correct[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return auroc(oof, correct)


@dataclass(frozen=True)
class Sweep:
    """Escalation rate vs Tier-0-retained accuracy, over candidate thresholds."""
    signal: str
    split: str
    thresholds: np.ndarray = field(repr=False)
    escalation_rate: np.ndarray = field(repr=False)
    retained_accuracy: np.ndarray = field(repr=False)

    def threshold_for_escalation(self, target: float) -> float:
        """Lowest threshold whose escalation rate is at least `target`."""
        idx = int(np.argmin(np.abs(self.escalation_rate - target)))
        return float(self.thresholds[idx])


def sweep_thresholds(logits: np.ndarray, labels: np.ndarray, *, signal: str, split: str,
                     n_points: int = 201) -> Sweep:
    """Calibrate on DEV ONLY. Escalate when the signal falls BELOW the threshold."""
    assert_threshold_split(split)
    score = CLOSED_FORM[signal](logits)
    correct = np.asarray(labels) == np.asarray(logits).argmax(-1)
    qs = np.linspace(0.0, 1.0, n_points)
    ths = np.quantile(score, qs)
    esc = np.array([(score < t).mean() for t in ths])
    ret = np.array([
        correct[score >= t].mean() if (score >= t).any() else np.nan for t in ths
    ])
    return Sweep(signal=signal, split=split, thresholds=ths,
                 escalation_rate=esc, retained_accuracy=ret)
