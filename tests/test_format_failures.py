"""Format-failure breakdown: make the cost of exact matching measurable.

PREREGISTRATION 3ag keeps exact matching. This module is what makes that choice
accountable — it reports how many failures a fold WOULD have repaired, so the price is
a number in the report rather than an argument.
"""

from __future__ import annotations

import re
import unicodedata

import pytest

from src.eval.format_failures import breakdown, collision_audit

EDGE = " \t\r\n\"'`“”‘’*_.,;:!?()[]{}<>"


def key(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).strip(EDGE)).casefold()


NAMES = ["Governing Laws", "Notices", "Indemnity", "Indemnifications", "Arbitration"]
LOOKUP = {key(n): n for n in NAMES}


def normalize(s): return LOOKUP.get(key(s))


def test_a_singular_near_miss_is_categorised_as_pluralisation():
    b = breakdown(["Governing Law"], [None], NAMES, key, normalize)
    assert b.pluralisation == 1 and b.other == 0
    assert b.recoverable_by_plural_fold == 1


def test_a_hallucinated_label_is_other_not_pluralisation():
    """'Indemnities' is not a label and is not one trailing 's' from one — it is a
    different stem. Counting it as a pluralisation near-miss would overstate what a fold
    could repair."""
    b = breakdown(["Indemnities"], [None], NAMES, key, normalize)
    assert b.other == 1 and b.pluralisation == 0
    assert b.recoverable_by_plural_fold == 0


def test_successful_rows_are_not_counted_as_failures():
    b = breakdown(["Notices", "Governing Law"], ["Notices", None], NAMES, key, normalize)
    assert b.n_rows == 2 and b.n_failures == 1


def test_ambiguous_near_miss_is_flagged_and_not_called_recoverable():
    """If a string is one 's' from TWO labels, no fold repairs it without choosing
    arbitrarily. Such rows must never be counted as recoverable."""
    names = ["Fee", "Fees", "Feess"]
    lookup = {key(n): n for n in names}
    b = breakdown(["Fees "], [None], names, key, lambda s: None)
    assert b.ambiguous == 1 and b.recoverable_by_plural_fold == 0
    assert b.ambiguous_pairs and len(b.ambiguous_pairs[0][1]) > 1


def test_case_or_edge_is_zero_unless_the_normaliser_regressed():
    """The normaliser already folds case and edge punctuation, so a failure of that kind
    means it broke. A non-zero count here is a bug signal, not a finding."""
    b = breakdown(["  governing laws.  "], [normalize("  governing laws.  ")],
                  NAMES, key, normalize)
    assert b.n_failures == 0


def test_the_real_label_space_has_no_plural_fold_collisions():
    """The empirical basis for 3ag's correction.

    A fold was argued against on the grounds that folding would merge real labels. It
    would not: LEDGAR's 100 names contain NO pair differing only by a trailing 's'.
    'Indemnity' and 'Indemnifications' are different stems. The decision to keep exact
    matching therefore rests on the format-failure-rate argument alone, and this test
    pins the fact so the discarded justification cannot creep back.
    """
    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = list(ds["train"].features["label"].names)
    assert len(names) == 100
    assert collision_audit(names, key) == []
    assert "Indemnity" in names and "Indemnifications" in names


def test_collision_audit_does_find_a_collision_when_one_exists():
    """The audit above returning [] must mean 'none', not 'the audit is broken'."""
    assert collision_audit(["Fee", "Fees"], key) == [("Fee", "Fees")]
