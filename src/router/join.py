"""Join Tier 0 logits to the Stage 1 Claude results BY DATASET ROW INDEX.

The join key is the dataset row index, NOT the custom_id string and NOT list position.

Why this is not a stylistic preference. Stage 1's custom_ids look like
`s1h_test_3000-000001`, `s1h_test_3000-000002`, `s1h_test_3000-000004` — the numeric
part is the LEDGAR row index and it is NOT contiguous, because test_3000 is a stratified
sample of the 10k test split. So:

  * joining by list position silently pairs Tier 0's row k with Claude's k-th RESULT,
    and the Batch API returns results in ANY ORDER (CLAUDE.md, verified batch fact), so
    the pairing is arbitrary;
  * joining by the raw custom_id string fails across legs, because the same row carries
    `s1h_...` under Haiku and `s1s_...` under Sonnet — the tag differs, the row does not.

Both failures produce a fully populated, plausible-looking joined table. This module
therefore parses the index out of the custom_id, and refuses on any mismatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

CUSTOM_ID = re.compile(r"^(?P<tag>[a-z0-9]+)_(?P<manifest>[a-z0-9_]+)-(?P<index>\d+)$")


def parse_row_index(custom_id: str) -> int:
    m = CUSTOM_ID.match(custom_id)
    if not m:
        raise ValueError(
            f"custom_id {custom_id!r} does not match {CUSTOM_ID.pattern!r}. Refusing to "
            f"guess a row index from an unrecognised format."
        )
    return int(m.group("index"))


@dataclass(frozen=True)
class Joined:
    row_indices: np.ndarray
    tier0_logits: np.ndarray
    tier0_labels: np.ndarray
    api_text: list[str]
    n: int


def join_by_row_index(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    logit_row_indices: np.ndarray,
    api_rows: dict[str, dict],
) -> Joined:
    """Align Tier 0 logits with one Claude leg's rows.

    `logit_row_indices` is the npz's `test_3000_indices`. `api_rows` maps custom_id to a
    result body. Every Tier 0 row must find exactly one API row and vice versa; anything
    else raises, because a partial join that silently drops rows changes the denominator
    of every metric computed afterwards.
    """
    logit_row_indices = np.asarray(logit_row_indices)
    by_index: dict[int, dict] = {}
    for cid, body in api_rows.items():
        idx = parse_row_index(cid)
        if idx in by_index:
            raise ValueError(
                f"row index {idx} appears twice in this leg (custom_id {cid!r}). The leg "
                f"is not one-result-per-row and the join is ambiguous."
            )
        by_index[idx] = body

    missing = [int(i) for i in logit_row_indices if int(i) not in by_index]
    extra = sorted(set(by_index) - {int(i) for i in logit_row_indices})
    if missing or extra:
        raise ValueError(
            f"join is not one-to-one: {len(missing)} logit rows have no API result "
            f"(first: {missing[:5]}), {len(extra)} API results have no logit row "
            f"(first: {extra[:5]}). Refusing to join a subset — dropping rows silently "
            f"changes the denominator of every metric computed from this table."
        )
    ordered = [by_index[int(i)] for i in logit_row_indices]
    return Joined(
        row_indices=logit_row_indices,
        tier0_logits=np.asarray(logits),
        tier0_labels=np.asarray(labels),
        api_text=[b.get("text", "") for b in ordered],
        n=len(logit_row_indices),
    )
