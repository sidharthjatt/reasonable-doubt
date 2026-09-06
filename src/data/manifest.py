"""Frozen evaluation manifests.

The examples we evaluate on must be FIXED and committed, so that every run — local,
Kaggle, API — sees exactly the same rows. Silent drift here invalidates every
comparison in the project, so a manifest records enough to detect drift rather than
merely describe the sample:

* the sampling seed and the exact allocation policy,
* the ordered list of source row indices,
* a sha256 over the concatenated example texts,
* a sha256 over the label space,
* per-class counts, classes represented, and the minimum class count.

:func:`verify_manifest` reloads the dataset, re-reads the indices, recomputes both
hashes and asserts they match. Every downstream script calls it before running.

Stratification policy
---------------------
Proportional allocation by class with the largest-remainder method, so the sample
sums to exactly ``n`` and per-class shares track the split's own distribution. With
100 long-tailed classes over 3000 rows, rare classes legitimately receive very few or
zero examples. We do NOT drop them from the label space and we do NOT oversample —
both would misrepresent the distribution the cascade actually faces. The shortfall is
recorded (``classes_represented``, ``min_class_count``) and reported.

Indices are stored ASCENDING, preserving the split's chronological row order.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.data.loading import (
    DATASET_CONFIG,
    DATASET_NAME,
    get_split,
    hash_texts,
    label_names,
    resolve_split,
)

if TYPE_CHECKING:
    from datasets import DatasetDict

__all__ = [
    "DEFAULT_MANIFEST_DIR",
    "MANIFEST_SPECS",
    "Manifest",
    "ManifestVerificationError",
    "build_manifest",
    "load_manifest",
    "proportional_random_indices",
    "stratified_indices",
    "verify_manifest",
]

DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "configs" / "manifests"

MANIFEST_FORMAT_VERSION = 1

# The two frozen evaluation sets. dev_2000 is for router threshold calibration ONLY
# (hard rule 1); test_3000 is the reporting set.
MANIFEST_SPECS: dict[str, dict[str, Any]] = {
    "test_3000": {
        "split": "test",
        "n": 3000,
        "seed": 20260907,
        "sampling": "stratified-proportional-largest-remainder",
        "purpose": "reporting",
    },
    "dev_2000": {
        "split": "validation",
        "n": 2000,
        "seed": 20260907,
        "sampling": "stratified-proportional-largest-remainder",
        "purpose": "router threshold calibration ONLY (hard rule 1)",
    },
    # Few-shot exemplars. Frozen exactly like the evaluation sets so the cacheable
    # prompt prefix cannot drift between runs. Train split only — never dev, never
    # test. A 4-exemplar run takes the FIRST 4 of these 8, so the smaller set is a
    # subset of the larger and the two stay comparable.
    "exemplars_8": {
        "split": "train",
        "n": 8,
        "seed": 20260907,
        "sampling": "proportional-random",
        "purpose": "few-shot exemplars (train only); N may be cut 8 -> 4 by the stage-2 budget gate",
    },
}

SAMPLERS = {}


class ManifestVerificationError(AssertionError):
    """A manifest no longer matches the dataset it was built from."""


@dataclass
class Manifest:
    """A frozen, verifiable evaluation sample."""

    name: str
    dataset: str
    config: str
    split: str
    purpose: str
    seed: int
    n: int
    sampling: str
    indices: list[int]
    text_sha256: str
    labels_sha256: str
    num_labels: int
    classes_represented: int
    min_class_count: int
    max_class_count: int
    class_counts: dict[str, int]
    format_version: int = MANIFEST_FORMAT_VERSION
    notes: list[str] = field(default_factory=list)

    def path(self, directory: str | Path = DEFAULT_MANIFEST_DIR) -> Path:
        return Path(directory) / f"{self.name}.json"

    def save(self, directory: str | Path = DEFAULT_MANIFEST_DIR) -> Path:
        path = self.path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return path


def load_manifest(name: str, directory: str | Path = DEFAULT_MANIFEST_DIR) -> Manifest:
    """Load a manifest by name. Does not verify it — call :func:`verify_manifest`."""
    path = Path(directory) / f"{name}.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ManifestVerificationError(
            f"{path}: format_version {data.get('format_version')} != "
            f"{MANIFEST_FORMAT_VERSION}"
        )
    return Manifest(**data)


def stratified_indices(labels: list[int], n: int, seed: int) -> list[int]:
    """Proportionally stratified sample of ``n`` row indices, ascending.

    Largest-remainder allocation over classes, then a seeded draw within each class.
    Deterministic: same ``labels``, ``n`` and ``seed`` always give the same indices.
    """
    import numpy as np

    total = len(labels)
    if not 0 < n <= total:
        raise ValueError(f"n must be in (0, {total}], got {n}")

    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    # Largest remainder: floor the proportional share, then hand out the leftover
    # seats to the largest fractional parts. Ties break on class index for
    # determinism, never on iteration order.
    quotas = {cls: len(rows) * n / total for cls, rows in by_class.items()}
    alloc = {cls: int(q) for cls, q in quotas.items()}
    remaining = n - sum(alloc.values())
    order = sorted(quotas, key=lambda cls: (-(quotas[cls] - alloc[cls]), cls))
    for cls in order[:remaining]:
        alloc[cls] += 1

    # Safety: allocation can never exceed a class's population under this scheme,
    # but assert rather than silently oversample.
    for cls, take in alloc.items():
        if take > len(by_class[cls]):
            raise ValueError(f"class {cls}: allocated {take} > available {len(by_class[cls])}")

    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for cls in sorted(by_class):
        take = alloc[cls]
        if take == 0:
            continue
        rows = by_class[cls]
        chosen = rng.choice(len(rows), size=take, replace=False)
        picked.extend(rows[i] for i in chosen)

    picked.sort()  # ascending == the split's chronological order
    if len(picked) != n:
        raise ValueError(f"sampled {len(picked)} rows, expected {n}")
    return picked


def proportional_random_indices(labels: list[int], n: int, seed: int) -> list[int]:
    """Uniform random draw of ``n`` row indices, ascending. Proportional in expectation.

    Used for few-shot exemplars. A uniform draw over rows IS a draw from the class
    prior, so it does not misrepresent the distribution. Deliberately NOT the
    largest-remainder stratifier: at n=8 over 100 classes every quota is below 1, so
    largest-remainder would deterministically return the 8 head classes every time —
    over-representing the head as a certainty rather than in expectation.
    """
    import numpy as np

    total = len(labels)
    if not 0 < n <= total:
        raise ValueError(f"n must be in (0, {total}], got {n}")
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(total, size=n, replace=False))


SAMPLERS.update(
    {
        "stratified-proportional-largest-remainder": stratified_indices,
        "proportional-random": proportional_random_indices,
    }
)


def _class_stats(names: list[str], labels: list[int]) -> dict[str, Any]:
    counts = Counter(int(lbl) for lbl in labels)
    per_class = {name: counts.get(i, 0) for i, name in enumerate(names)}
    present = [c for c in per_class.values() if c > 0]
    return {
        "class_counts": per_class,
        "classes_represented": len(present),
        "min_class_count": min(per_class.values()),
        "max_class_count": max(per_class.values()),
    }


def build_manifest(name: str, ds: "DatasetDict", spec: dict[str, Any] | None = None) -> Manifest:
    """Build a manifest from a loaded dataset. Pure given ``(ds, spec)``."""
    spec = dict(spec or MANIFEST_SPECS[name])
    split_name = resolve_split(spec["split"])
    split = get_split(ds, split_name)
    names = label_names(ds)

    sampling = spec["sampling"]
    if sampling not in SAMPLERS:
        raise ValueError(f"unknown sampling policy {sampling!r}; have {sorted(SAMPLERS)}")
    indices = SAMPLERS[sampling](list(split["label"]), spec["n"], spec["seed"])
    rows = split.select(indices)
    texts = list(rows["text"])
    stats = _class_stats(names, list(rows["label"]))

    num_absent = len(names) - stats["classes_represented"]
    notes = ["Indices are ascending, preserving the split's chronological row order."]
    if sampling == "stratified-proportional-largest-remainder":
        notes.append(
            "Proportional stratification with largest-remainder allocation; rare "
            "classes are neither dropped nor oversampled."
        )
    else:
        notes.append(
            "Uniform random draw over rows, i.e. a draw from the class prior. No "
            "stratification, and no label-coverage claim is made."
        )
    notes.append(f"{num_absent} of {len(names)} classes received zero examples.")

    return Manifest(
        name=name,
        dataset=DATASET_NAME,
        config=DATASET_CONFIG,
        split=split_name,
        purpose=spec.get("purpose", ""),
        seed=spec["seed"],
        n=spec["n"],
        sampling=sampling,
        indices=indices,
        text_sha256=hash_texts(texts),
        labels_sha256=hash_texts(names),
        num_labels=len(names),
        notes=notes,
        **stats,
    )


def verify_manifest(
    manifest: "Manifest | str",
    ds: "DatasetDict",
    directory: str | Path = DEFAULT_MANIFEST_DIR,
) -> Manifest:
    """Re-read the indices from the dataset and assert the hashes still match.

    Detects: changed dataset content, changed row order, a tampered index list, a
    changed label space, and any edit to the recorded per-class counts.

    Raises:
        ManifestVerificationError: on any mismatch. Callers must not proceed.
    """
    m = load_manifest(manifest, directory) if isinstance(manifest, str) else manifest

    if m.dataset != DATASET_NAME or m.config != DATASET_CONFIG:
        raise ManifestVerificationError(
            f"{m.name}: built from {m.dataset}:{m.config}, "
            f"current code targets {DATASET_NAME}:{DATASET_CONFIG}"
        )

    split = get_split(ds, m.split)
    if len(m.indices) != m.n:
        raise ManifestVerificationError(
            f"{m.name}: {len(m.indices)} indices recorded but n={m.n}"
        )
    if len(set(m.indices)) != len(m.indices):
        raise ManifestVerificationError(f"{m.name}: duplicate indices")
    if m.indices != sorted(m.indices):
        raise ManifestVerificationError(
            f"{m.name}: indices are not ascending — row order must be preserved"
        )
    out_of_range = [i for i in m.indices if not 0 <= i < len(split)]
    if out_of_range:
        raise ManifestVerificationError(
            f"{m.name}: {len(out_of_range)} indices out of range for split "
            f"{m.split!r} (size {len(split)}); first few: {out_of_range[:5]}"
        )

    names = label_names(ds)
    actual_labels_hash = hash_texts(names)
    if actual_labels_hash != m.labels_sha256:
        raise ManifestVerificationError(
            f"{m.name}: label space changed (expected {m.labels_sha256[:16]}…, "
            f"got {actual_labels_hash[:16]}…)"
        )

    rows = split.select(m.indices)
    actual_text_hash = hash_texts(list(rows["text"]))
    if actual_text_hash != m.text_sha256:
        raise ManifestVerificationError(
            f"{m.name}: TEXT DRIFT. Expected sha256 {m.text_sha256}, got "
            f"{actual_text_hash}. The rows this manifest addresses are not the rows "
            "it was built from — every result computed against it is invalid."
        )

    stats = _class_stats(names, list(rows["label"]))
    for key, expected in (
        ("class_counts", m.class_counts),
        ("classes_represented", m.classes_represented),
        ("min_class_count", m.min_class_count),
        ("max_class_count", m.max_class_count),
    ):
        if stats[key] != expected:
            raise ManifestVerificationError(f"{m.name}: recorded {key} does not match dataset")

    return m
