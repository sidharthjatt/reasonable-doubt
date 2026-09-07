"""Classification metrics.

Macro-F1 is the PRIMARY metric (hard constraint from the dataset section of
CLAUDE.md): LEDGAR's class distribution is heavily long-tailed, so accuracy is
dominated by the head and would hide total failure on the tail. Accuracy is reported
alongside, never instead.

Unmatched model outputs — text that does not normalize to a canonical label — are
counted as **wrong**, never dropped. Dropping them would flatter the model by silently
removing its worst failures from the denominator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = ["ClassificationReport", "score"]


@dataclass
class ClassificationReport:
    """Scores over one evaluation run. ``macro_f1`` leads."""

    macro_f1: float
    accuracy: float
    n: int
    n_unmatched: int
    format_failure_rate: float
    classes_in_gold: int
    classes_predicted: int
    per_class_f1: dict[str, float] = field(default_factory=dict)
    unmatched_samples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "n": self.n,
            "n_unmatched": self.n_unmatched,
            "format_failure_rate": self.format_failure_rate,
            "classes_in_gold": self.classes_in_gold,
            "classes_predicted": self.classes_predicted,
            "per_class_f1": self.per_class_f1,
        }

    def render(self) -> str:
        return (
            f"  macro-F1            : {self.macro_f1:.4f}   <-- primary metric\n"
            f"  accuracy            : {self.accuracy:.4f}\n"
            f"  n                   : {self.n}\n"
            f"  unmatched outputs   : {self.n_unmatched} "
            f"({self.format_failure_rate:.2%}) — counted as WRONG\n"
            f"  classes in gold     : {self.classes_in_gold}\n"
            f"  classes predicted   : {self.classes_predicted}"
        )


def score(
    gold: Sequence[str],
    predicted: Sequence[str | None],
    *,
    labels: Sequence[str] | None = None,
    unmatched_samples: Sequence[str] = (),
) -> ClassificationReport:
    """Score predictions against gold labels.

    Args:
        gold: Canonical gold label strings.
        predicted: Canonical predicted labels, or ``None`` for an output that did not
            normalize to any label. ``None`` counts as wrong.
        labels: Label space to average over. Defaults to the classes present in gold —
            macro-F1 over classes with no gold examples is undefined, and averaging in
            a zero for them would silently deflate the score.

    Note:
        Macro-F1 here averages over ``labels``. When a manifest does not cover every
        class (``dev_2000`` covers 99 of 100), the number of classes averaged over must
        be stated wherever the score is reported — see ``classes_in_gold``.
    """
    from sklearn.metrics import f1_score

    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} gold labels but {len(predicted)} predictions")
    if not gold:
        raise ValueError("nothing to score")

    # A sentinel keeps unmatched outputs in the denominator without inventing a class
    # that could ever be counted as correct.
    UNMATCHED = "\x00__UNMATCHED__"
    preds = [UNMATCHED if p is None else p for p in predicted]

    average_over = list(labels) if labels is not None else sorted(set(gold))
    per_class = f1_score(gold, preds, labels=average_over, average=None, zero_division=0)

    n_unmatched = sum(1 for p in predicted if p is None)
    return ClassificationReport(
        macro_f1=float(sum(per_class) / len(per_class)),
        accuracy=sum(g == p for g, p in zip(gold, preds)) / len(gold),
        n=len(gold),
        n_unmatched=n_unmatched,
        format_failure_rate=n_unmatched / len(gold),
        classes_in_gold=len(set(gold)),
        classes_predicted=len({p for p in predicted if p is not None}),
        per_class_f1={c: float(f) for c, f in zip(average_over, per_class)},
        unmatched_samples=list(unmatched_samples),
    )
