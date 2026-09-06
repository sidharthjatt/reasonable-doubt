"""LEDGAR loading with loud structural assertions.

Dataset: ``coastalcph/lex_glue``, config ``ledgar`` — ~80k contract provisions from
SEC EDGAR Exhibit-10 filings, 100 single-label classes.

The split is CHRONOLOGICAL: train 2016-2017, dev 2018, test 2019. We never reshuffle
across splits, never concatenate and re-split, and never reorder rows. Row order in
each split is the dataset's own order and is the stable addressing scheme that the
frozen manifests index into.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid importing datasets at module import time
    from datasets import Dataset, DatasetDict

__all__ = [
    "DATASET_CONFIG",
    "DATASET_NAME",
    "EXPECTED_NUM_LABELS",
    "EXPECTED_SPLIT_SIZES",
    "SPLIT_ALIASES",
    "DatasetShapeError",
    "get_split",
    "hash_texts",
    "label_names",
    "load_ledgar",
    "resolve_split",
    "text_sha256",
]

DATASET_NAME = "coastalcph/lex_glue"
DATASET_CONFIG = "ledgar"

# Verified 2026-09-07 by loading the dataset. Chronological, not random.
EXPECTED_SPLIT_SIZES: dict[str, int] = {
    "train": 60_000,      # 2016-2017
    "validation": 10_000,  # 2018 — the dev split; router calibration only
    "test": 10_000,       # 2019
}
EXPECTED_NUM_LABELS = 100

# We say "dev" in prose and configs; the dataset calls it "validation".
SPLIT_ALIASES = {"dev": "validation", "val": "validation"}

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


class DatasetShapeError(AssertionError):
    """The loaded dataset does not match the verified LEDGAR shape."""


def resolve_split(split: str) -> str:
    """Map ``dev``/``val`` onto the dataset's ``validation``; pass others through."""
    return SPLIT_ALIASES.get(split, split)


def load_ledgar(**load_kwargs: Any) -> "DatasetDict":
    """Load LEDGAR and assert its shape. Raises :class:`DatasetShapeError` on drift.

    Every structural assumption the rest of the project relies on is checked here,
    once, loudly — split names, split sizes, column names, and the 100-class label
    space. A silent change in any of them would invalidate every downstream number.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, **load_kwargs)

    missing = set(EXPECTED_SPLIT_SIZES) - set(ds)
    if missing:
        raise DatasetShapeError(
            f"missing splits {sorted(missing)}; got {sorted(ds)}. "
            f"Expected {sorted(EXPECTED_SPLIT_SIZES)}."
        )

    bad_sizes = {
        name: (len(ds[name]), expected)
        for name, expected in EXPECTED_SPLIT_SIZES.items()
        if len(ds[name]) != expected
    }
    if bad_sizes:
        detail = ", ".join(f"{k}: got {g}, expected {e}" for k, (g, e) in bad_sizes.items())
        raise DatasetShapeError(
            f"split sizes changed ({detail}). The chronological 60k/10k/10k split is "
            "load-bearing — do NOT proceed by re-splitting or reshuffling."
        )

    for name in EXPECTED_SPLIT_SIZES:
        cols = ds[name].column_names
        if TEXT_COLUMN not in cols or LABEL_COLUMN not in cols:
            raise DatasetShapeError(
                f"split {name!r} columns are {cols}; expected "
                f"{TEXT_COLUMN!r} and {LABEL_COLUMN!r}"
            )

    names = label_names(ds)
    if len(names) != EXPECTED_NUM_LABELS:
        raise DatasetShapeError(
            f"expected {EXPECTED_NUM_LABELS} label names, got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise DatasetShapeError("label names contain duplicates")

    return ds


def label_names(ds: "DatasetDict | Dataset") -> list[str]:
    """The 100 human-readable class names, in the dataset's canonical index order.

    The LLM tiers classify into label *strings*, so the names — not the indices — are
    the interface. Index order is preserved so ``names[i]`` is always class ``i``.
    """
    split = ds if hasattr(ds, "features") else ds[next(iter(ds))]
    feature = split.features[LABEL_COLUMN]
    if not hasattr(feature, "names"):
        raise DatasetShapeError(
            f"label column is {feature!r}, not a ClassLabel with .names"
        )
    return list(feature.names)


def get_split(ds: "DatasetDict", split: str) -> "Dataset":
    """Fetch a split by name, accepting the ``dev`` alias."""
    resolved = resolve_split(split)
    if resolved not in ds:
        raise KeyError(f"no split {split!r} (resolved to {resolved!r}); have {sorted(ds)}")
    return ds[resolved]


def text_sha256(text: str) -> str:
    """Stable per-example hash, used for leakage and duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_texts(texts: list[str]) -> str:
    """Order-sensitive sha256 over a sequence of texts.

    Each text is fed in as ``len(utf8_bytes)`` (decimal, newline-terminated) followed
    by the bytes themselves. The explicit length prefix means no concatenation of
    different texts can collide with another sequence, and the hash changes if the
    order changes — which is exactly the drift we want to catch.
    """
    h = hashlib.sha256()
    for text in texts:
        payload = text.encode("utf-8")
        h.update(f"{len(payload)}\n".encode("ascii"))
        h.update(payload)
    return h.hexdigest()
