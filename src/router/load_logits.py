"""Load Tier 0's .npz and hand E5 exactly the split it asked for — never another.

The npz carries three splits (sel_, dev_, test_). Reaching for the wrong array is a
one-character mistake that produces a full-length, plausible result, so the split is
named and guarded on the way out rather than indexed by hand at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.router.calibrate import assert_threshold_split

_PREFIX = {"dev_2000": "dev", "train_holdout_3000": "sel", "test_3000": "test"}


@dataclass(frozen=True)
class SplitLogits:
    split: str
    logits: np.ndarray
    labels: np.ndarray
    row_indices: np.ndarray | None
    absent_classes: np.ndarray
    n_classes: int

    @property
    def covers_all_classes(self) -> bool:
        return len(self.absent_classes) == 0


def load_split(npz_path: Path | str, split: str, *, for_calibration: bool = False) -> SplitLogits:
    """Read one split's logits.

    `for_calibration=True` applies hard rule 1 — the load itself refuses anything that is
    not the dev split, so a router threshold cannot be fitted to selection or test data
    even by a caller that never consults the guard.
    """
    if for_calibration:
        assert_threshold_split(split)
    if split not in _PREFIX:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(_PREFIX)}")
    p = _PREFIX[split]

    with np.load(npz_path) as z:
        keys = set(z.files)
        if f"{p}_logits" not in keys:
            raise KeyError(
                f"{npz_path} has no {p}_logits (found {sorted(keys)}). If this is a "
                f"pre-2026-09-08 Tier 0 run it predates dev_2000 inference and CANNOT be "
                f"used to calibrate a router threshold — rerun the eval block with the "
                f"trained fp32 checkpoint attached."
            )
        logits = np.asarray(z[f"{p}_logits"], dtype=float)
        labels = np.asarray(z[f"{p}_labels"] if f"{p}_labels" in keys else z["dev_labels"])
        idx_key = {"dev": "dev_2000_indices", "test": "test_3000_indices"}.get(p)
        row_idx = np.asarray(z[idx_key]) if idx_key in keys else None
        absent = np.asarray(z["dev_absent_classes"]) if "dev_absent_classes" in keys else np.array([], int)

    if len(logits) != len(labels):
        raise ValueError(f"{split}: {len(logits)} logit rows but {len(labels)} labels")
    return SplitLogits(split=split, logits=logits, labels=labels, row_indices=row_idx,
                       absent_classes=absent if p == "dev" else np.array([], int),
                       n_classes=logits.shape[1])


def macro_f1_kwargs(sl: SplitLogits) -> dict:
    """The kwargs `src.eval.metrics.score` needs for THIS split, and no others.

    `score` raises unless `allow_absent_classes=True` when a class has no support, which
    is correct: averaging a 0.0 over a class nobody could predict deflates macro-F1 in
    proportion to how many are missing. dev_2000 is 99/100 by design (PREREGISTRATION 3a),
    so the flag is genuinely needed there — but it is returned only for the split that
    actually needs it, so a caller cannot carry it over to test_3000 and silence a real
    coverage problem.
    """
    if sl.covers_all_classes:
        return {}
    return {"allow_absent_classes": True}
