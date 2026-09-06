"""Manifest determinism, tamper detection, and the leakage assertion itself."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from src.data.loading import hash_texts, text_sha256
from src.data.manifest import (
    DEFAULT_MANIFEST_DIR,
    MANIFEST_SPECS,
    ManifestVerificationError,
    build_manifest,
    load_manifest,
    stratified_indices,
    verify_manifest,
)

# ---------------------------------------------------------------- sampling policy


def _synthetic_labels(counts: dict[int, int]) -> list[int]:
    labels = []
    for cls, n in counts.items():
        labels.extend([cls] * n)
    return labels


def test_stratified_sampling_is_deterministic():
    labels = _synthetic_labels({0: 500, 1: 300, 2: 150, 3: 40, 4: 10})
    a = stratified_indices(labels, 200, seed=7)
    b = stratified_indices(labels, 200, seed=7)
    assert a == b


def test_different_seed_gives_different_sample():
    labels = _synthetic_labels({0: 500, 1: 300, 2: 150, 3: 40, 4: 10})
    assert stratified_indices(labels, 200, 7) != stratified_indices(labels, 200, 8)


def test_sample_is_ascending_and_unique():
    labels = _synthetic_labels({0: 500, 1: 300, 2: 200})
    idx = stratified_indices(labels, 137, seed=1)
    assert idx == sorted(idx)
    assert len(set(idx)) == len(idx) == 137


def test_allocation_is_proportional_with_exact_total():
    labels = _synthetic_labels({0: 800, 1: 150, 2: 50})
    idx = stratified_indices(labels, 100, seed=3)
    counts = Counter(labels[i] for i in idx)
    assert sum(counts.values()) == 100
    assert counts == {0: 80, 1: 15, 2: 5}


def test_rare_class_may_get_zero_and_is_never_oversampled():
    """A class too rare to earn a seat gets zero. It is not padded and not dropped."""
    labels = _synthetic_labels({0: 9_990, 1: 10})
    idx = stratified_indices(labels, 100, seed=5)
    counts = Counter(labels[i] for i in idx)
    assert counts[1] == 0
    assert counts[0] == 100
    # never more of a class than the split contains
    for cls, n in counts.items():
        assert n <= labels.count(cls)


def test_n_out_of_range_rejected():
    labels = _synthetic_labels({0: 10})
    for bad in (0, -1, 11):
        with pytest.raises(ValueError):
            stratified_indices(labels, bad, seed=1)


# ------------------------------------------------------- committed manifests (real data)


@pytest.mark.parametrize("name", sorted(MANIFEST_SPECS))
def test_committed_manifest_verifies(ledgar, name):
    verify_manifest(name, ledgar, DEFAULT_MANIFEST_DIR)


@pytest.mark.parametrize("name", sorted(MANIFEST_SPECS))
def test_regenerating_reproduces_the_committed_manifest(ledgar, name):
    """Same seed, same dataset -> byte-identical manifest. This is the guarantee that
    every run (local, Kaggle, API) evaluates on exactly the same rows."""
    rebuilt = build_manifest(name, ledgar)
    committed = load_manifest(name, DEFAULT_MANIFEST_DIR)
    assert rebuilt.indices == committed.indices
    assert rebuilt.text_sha256 == committed.text_sha256
    assert rebuilt.labels_sha256 == committed.labels_sha256
    assert rebuilt.class_counts == committed.class_counts


def test_manifest_sizes_and_splits(ledgar):
    test = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    dev = load_manifest("dev_2000", DEFAULT_MANIFEST_DIR)
    assert (test.n, test.split) == (3000, "test")
    assert (dev.n, dev.split) == (2000, "validation")
    assert "calibration" in dev.purpose.lower()  # hard rule 1


def test_manifests_index_disjoint_splits(ledgar):
    """dev calibration rows and test reporting rows can never be the same rows."""
    test = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    dev = load_manifest("dev_2000", DEFAULT_MANIFEST_DIR)
    assert test.split != dev.split

    test_texts = set(ledgar[test.split].select(test.indices)["text"])
    dev_texts = set(ledgar[dev.split].select(dev.indices)["text"])
    assert not (test_texts & dev_texts)


# ------------------------------------------------------------------ tamper detection


def test_mutating_one_index_fails_verification(ledgar):
    m = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    original = m.indices[0]
    m.indices = [i for i in m.indices]
    # swap in a row not already sampled, keeping the list ascending and unique
    replacement = next(i for i in range(len(ledgar[m.split])) if i not in set(m.indices))
    m.indices = sorted(set(m.indices) - {original} | {replacement})

    with pytest.raises(ManifestVerificationError, match="TEXT DRIFT|class_counts"):
        verify_manifest(m, ledgar, DEFAULT_MANIFEST_DIR)


def test_reordering_indices_fails_verification(ledgar):
    m = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    m.indices = list(reversed(m.indices))
    with pytest.raises(ManifestVerificationError, match="ascending"):
        verify_manifest(m, ledgar, DEFAULT_MANIFEST_DIR)


def test_duplicated_index_fails_verification(ledgar):
    m = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    m.indices = m.indices[:-1] + [m.indices[0]]
    with pytest.raises(ManifestVerificationError, match="duplicate|ascending"):
        verify_manifest(m, ledgar, DEFAULT_MANIFEST_DIR)


def test_out_of_range_index_fails_verification(ledgar):
    m = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    m.indices = m.indices[:-1] + [999_999]
    with pytest.raises(ManifestVerificationError, match="out of range"):
        verify_manifest(m, ledgar, DEFAULT_MANIFEST_DIR)


def test_tampered_text_hash_fails_verification(ledgar):
    m = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    m.text_sha256 = "0" * 64
    with pytest.raises(ManifestVerificationError, match="TEXT DRIFT"):
        verify_manifest(m, ledgar, DEFAULT_MANIFEST_DIR)


def test_tampered_label_space_hash_fails_verification(ledgar):
    m = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    m.labels_sha256 = "0" * 64
    with pytest.raises(ManifestVerificationError, match="label space changed"):
        verify_manifest(m, ledgar, DEFAULT_MANIFEST_DIR)


def test_tampered_class_counts_fail_verification(ledgar):
    m = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    key = next(iter(m.class_counts))
    m.class_counts = {**m.class_counts, key: m.class_counts[key] + 1}
    with pytest.raises(ManifestVerificationError, match="class_counts"):
        verify_manifest(m, ledgar, DEFAULT_MANIFEST_DIR)


def test_wrong_format_version_rejected(tmp_path):
    src = DEFAULT_MANIFEST_DIR / "test_3000.json"
    data = json.loads(src.read_text())
    data["format_version"] = 999
    (tmp_path / "test_3000.json").write_text(json.dumps(data))
    with pytest.raises(ManifestVerificationError, match="format_version"):
        load_manifest("test_3000", tmp_path)


# -------------------------------------------------------------- hashing primitives


def test_hash_texts_is_order_sensitive():
    assert hash_texts(["a", "b"]) != hash_texts(["b", "a"])


def test_hash_texts_length_prefix_prevents_boundary_collisions():
    """['ab','c'] and ['a','bc'] concatenate identically but must hash differently."""
    assert hash_texts(["ab", "c"]) != hash_texts(["a", "bc"])


def test_text_sha256_is_stable():
    assert text_sha256("hello") == text_sha256("hello")
    assert text_sha256("hello") != text_sha256("hello ")
