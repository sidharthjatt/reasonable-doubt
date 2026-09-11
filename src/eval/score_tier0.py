"""AUTHORITATIVE offline scoring of E1/E2/E3 from Tier 0's .npz logits.

The Kaggle notebook computes a macro-F1 inline with a bare
``f1_score(..., zero_division=0)`` call. That is the library default PREREGISTRATION 3e
instance 3 records as wrong for this project, it reports no ``classes_averaged``, and it
bypasses ``src.eval.metrics.score`` entirely.

**The notebook's number is a CONVENIENCE number.** It exists so a running seed prints
something legible, and it is labelled as such wherever it appears. **E1, E2 and E3 are
scored here**, from the logits, through ``src.eval.metrics.score``, with
``allow_absent_classes`` passed explicitly and ``classes_averaged`` recorded on every
result.

On ``test_3000`` the two conventions are arithmetically identical, because that manifest
contains all 100 classes (``classes_represented: 100``, ``min_class_count: 1``), so no
class is averaged in at 0.0 by either route. The divergence is confined to ``dev_2000``,
which covers 99/100 by design, and therefore to **threshold calibration only** — never to
E1's or E4's headline. ``tests/test_score_tier0.py`` demonstrates that equivalence rather
than asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eval.metrics import ClassificationReport, score

# Which arrays in the npz belong to which split, and which precision each one is.
_FP32 = {"test_3000": ("test_logits", "test_labels"),
         "train_holdout_3000": ("sel_logits", "sel_labels"),
         "dev_2000": ("dev_logits", "dev_labels")}
_INT8 = {"test_3000": ("test_3000_logits", None),
         "dev_2000": ("dev_2000_logits", None)}

# HOW MANY ROWS EACH SPLIT MUST HAVE, and how to get them if the array holds more.
#
# THIS EXISTS BECAUSE THE ARRAYS DO NOT ALL MEAN WHAT THEIR SPLIT NAME SAYS.
# `test_logits` in every Tier 0 npz is the FULL 10,000-row test split, NOT the 3,000-row
# `test_3000` manifest. Scoring it whole and labelling the answer `test_3000` yields a
# plausible, wrong number: on logits_ce_seed1.npz it gives 0.766747 over n=10,000 against
# the true test_3000 figure of 0.763591, a +0.0032 difference with nothing saying so. The
# npz carries `test_3000_indices` precisely so the subset can be taken, and taking it is
# now mandatory rather than remembered.
#
# `sel_logits` and `dev_logits` are already stored pre-subset at their split's size, so
# they need no indices — but they are still row-count checked, because "already correct"
# is a property of today's writer, not a guarantee.
_EXPECTED_N = {"test_3000": 3000, "dev_2000": 2000, "train_holdout_3000": 3000}
_INDEX_KEY = {"test_3000": "test_3000_indices", "dev_2000": "dev_2000_indices"}


class SplitRowCountError(ValueError):
    """An npz array did not hold the number of rows its split name claims."""


def subset_to_split(logits, labels, z, *, split: str, precision: str):
    """Return exactly the rows of ``split``, or raise. Never returns a superset.

    Takes the index array when the stored array is larger than the split, and refuses
    outright when the count cannot be reconciled — an approximation here would be a
    differently-sized evaluation reported under the split's name (hard rule 11).
    """
    import numpy as _np

    want = _EXPECTED_N.get(split)
    if want is None or len(logits) == want:
        return logits, labels

    idx_key = _INDEX_KEY.get(split)
    if len(logits) > want and idx_key and idx_key in z.files:
        idx = _np.asarray(z[idx_key])
        if len(idx) != want:
            raise SplitRowCountError(
                f"{split}/{precision}: {idx_key} has {len(idx)} entries but {split} is "
                f"{want} rows; refusing to subset with an index array of the wrong size")
        if idx.max() >= len(logits):
            raise SplitRowCountError(
                f"{split}/{precision}: {idx_key} indexes row {int(idx.max())} but the "
                f"array has only {len(logits)} rows")
        return logits[idx], (None if labels is None else labels[idx])

    raise SplitRowCountError(
        f"{split}/{precision}: array has {len(logits)} rows, but {split} is {want}. "
        f"No usable {idx_key!r} is present to subset it.\n"
        f"Scoring the array whole would report a differently-sized evaluation under this "
        f"split's name — `test_logits` is the FULL 10,000-row test split, not test_3000.")


@dataclass(frozen=True)
class TierZeroScores:
    split: str
    precision: str
    report: ClassificationReport
    allow_absent_classes: bool
    n_classes_in_label_space: int

    def as_dict(self) -> dict:
        d = self.report.as_dict()
        d.update({
            "split": self.split,
            "precision": self.precision,
            "scored_by": "src.eval.metrics.score (AUTHORITATIVE)",
            "allow_absent_classes": self.allow_absent_classes,
            "classes_averaged": self.report.classes_averaged,
            "n_classes_in_label_space": self.n_classes_in_label_space,
        })
        return d


def score_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    *,
    split: str,
    precision: str,
    over_full_label_space: bool = False,
) -> TierZeroScores:
    """Score argmax predictions through the project's guarded scorer.

    ``over_full_label_space=False`` (default) averages over the classes present in gold,
    which is what E1's rule means by macro-F1 on ``test_3000``. Setting it True averages
    over all 100 and requires ``allow_absent_classes``, which ``score`` will demand — the
    flag is never passed on the caller's behalf.
    """
    logits, labels = np.asarray(logits), np.asarray(labels)
    if len(logits) != len(labels):
        raise ValueError(f"{split}/{precision}: {len(logits)} logit rows, {len(labels)} labels")
    gold = [names[int(i)] for i in labels]
    pred = [names[int(i)] for i in logits.argmax(-1)]

    if over_full_label_space:
        rep = score(gold, pred, labels=names, allow_absent_classes=True)
        allow = True
    else:
        rep = score(gold, pred)          # labels default to the classes present in gold
        allow = False
    return TierZeroScores(split=split, precision=precision, report=rep,
                          allow_absent_classes=allow, n_classes_in_label_space=len(names))


def score_npz(npz_path: Path | str, names: list[str], *, split: str,
              precision: str = "fp32", **kw) -> TierZeroScores:
    table = _FP32 if precision == "fp32" else _INT8
    if split not in table:
        raise KeyError(f"{precision} logits are not stored for split {split!r}; "
                       f"available: {sorted(table)}")
    lk, gk = table[split]
    with np.load(npz_path) as z:
        if lk not in z.files:
            raise KeyError(
                f"{npz_path} has no {lk!r} (found {sorted(z.files)}). A pre-2026-09-08 "
                f"Tier 0 run predates dev_2000 inference; rerun the eval block."
            )
        logits = z[lk]
        if gk is not None:
            labels = z[gk]
            logits, labels = subset_to_split(logits, labels, z, split=split,
                                             precision=precision)
        else:
            # INT8 arrays carry no labels of their own — they are the SAME rows as the
            # FP32 arrays for that split, so the FP32 labels apply. Assert the shapes
            # agree rather than assuming the row order matches.
            flk, fgk = _FP32[split]
            if fgk not in z.files:
                raise KeyError(f"{npz_path}: INT8 {split} logits present but no {fgk} "
                               f"to score them against")
            labels = z[fgk]
            # The FP32 labels are stored at FULL-SPLIT size while the INT8 logits are
            # already the split's rows, so the labels are subset to match before the
            # shape agreement is asserted.
            labels, _ = subset_to_split(labels, None, z, split=split, precision="fp32")
            if len(labels) != len(logits):
                raise ValueError(
                    f"{split}: INT8 has {len(logits)} rows but FP32 labels have "
                    f"{len(labels)}. These must be the same rows in the same order; "
                    f"refusing to score one against the other."
                )
    want = _EXPECTED_N.get(split)
    if want is not None and len(logits) != want:
        raise SplitRowCountError(
            f"{split}/{precision}: about to score {len(logits)} rows for a {want}-row "
            f"split. This is the last guard before the number is produced.")
    return score_from_logits(logits, labels, names, split=split, precision=precision, **kw)


def notebook_convention_macro_f1(logits: np.ndarray, labels: np.ndarray,
                                 names: list[str]) -> float:
    """Reproduce the NOTEBOOK's inline number, for comparison only.

    Not authoritative. Exists so the equivalence on test_3000 can be demonstrated, and so
    any divergence elsewhere is a measured number rather than an argument.
    """
    from sklearn.metrics import f1_score
    gold = [names[int(i)] for i in np.asarray(labels)]
    pred = [names[int(i)] for i in np.asarray(logits).argmax(-1)]
    return float(f1_score(gold, pred, labels=sorted(set(gold)), average="macro",
                          zero_division=0))
