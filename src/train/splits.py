"""Split guard: the test split must be unreachable from selection code.

Hard rule 1 protects test from threshold calibration. This module makes the wider
constraint mechanical: **no split used to pick anything may be the test split** —
not an epoch, not a hyperparameter, not a loss arm, not a routing signal.

The guard is a function that raises, not a comment that asks. Selection code calls
:func:`assert_selection_split` on whatever it was handed, so passing the reporting
split into a `select_*` path fails immediately rather than producing a number.
"""

from __future__ import annotations

__all__ = [
    "REPORTING_SPLITS",
    "SELECTION_SPLITS",
    "THRESHOLD_SPLITS",
    "ForbiddenSplitError",
    "assert_selection_split",
    "assert_threshold_split",
]

# Choosing an epoch, a hyperparameter, a loss arm, a max_length, a routing signal.
SELECTION_SPLITS = frozenset({"train_holdout_3000", "train"})

# Calibrating a router threshold. dev ONLY, and never also used for selection —
# a threshold tuned on data the model was already selected against is tuned twice.
THRESHOLD_SPLITS = frozenset({"dev_2000", "validation", "dev"})

# Touched once, for the reported number. Never for any decision.
REPORTING_SPLITS = frozenset({"test_3000", "test_stratified_764", "test"})


class ForbiddenSplitError(RuntimeError):
    """A split was used for a purpose it is not permitted to serve."""


def assert_selection_split(name: str) -> str:
    """Permit only selection splits. Raises on dev or test.

    Called by anything that PICKS: best epoch, best hyperparameter, best loss arm,
    best routing signal, best max_length.
    """
    if name in REPORTING_SPLITS:
        raise ForbiddenSplitError(
            f"REFUSING: {name!r} is a REPORTING split and must never be used to select "
            "anything (hard rule 6). Selection runs on "
            f"{sorted(SELECTION_SPLITS)}. Test is touched once, for the reported number."
        )
    if name in THRESHOLD_SPLITS:
        raise ForbiddenSplitError(
            f"REFUSING: {name!r} is reserved for router THRESHOLD calibration only "
            "(hard rule 1). Selecting a checkpoint on it too would tune the threshold "
            "against data the model was already chosen to fit. Use "
            f"{sorted(SELECTION_SPLITS)}."
        )
    if name not in SELECTION_SPLITS:
        raise ForbiddenSplitError(
            f"unknown split {name!r}; selection splits are {sorted(SELECTION_SPLITS)}"
        )
    return name


def assert_threshold_split(name: str) -> str:
    """Permit only the dev split. Raises on train, selection, or test."""
    if name not in THRESHOLD_SPLITS:
        raise ForbiddenSplitError(
            f"REFUSING: router thresholds calibrate on the DEV split only (hard rule 1), "
            f"not on {name!r}. Permitted: {sorted(THRESHOLD_SPLITS)}."
        )
    return name
