"""The leakage and shape assertions must actually fire.

A check that never fails is worse than no check: it manufactures confidence. These
tests build deliberately broken datasets and assert the guards trip.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.data.loading import DatasetShapeError, load_ledgar, text_sha256

ROOT = Path(__file__).resolve().parents[1]


def _load_verify_data_module():
    spec = importlib.util.spec_from_file_location(
        "verify_data", ROOT / "scripts" / "verify_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_data"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vd():
    return _load_verify_data_module()


def _hashes(train, validation, test):
    return {
        "train": [text_sha256(t) for t in train],
        "validation": [text_sha256(t) for t in validation],
        "test": [text_sha256(t) for t in test],
    }


def test_leakage_check_fires_on_overlapping_fixture(vd):
    """A clause present in both train and test must trip the leakage check."""
    shared = "This Agreement shall be governed by the laws of the State of Delaware."
    hashes = _hashes(
        train=["clause A", "clause B", shared],
        validation=["clause C"],
        test=["clause D", shared],
    )

    rep = vd.Report()
    vd.check_leakage(rep, None, hashes)

    assert rep.failures, "leakage check passed on an overlapping fixture"
    assert "LEAKAGE" in rep.failures[0]
    assert "train∩test=1" in rep.failures[0]
    assert "**FAIL" in rep.text()  # rendered unmissably in the report


def test_leakage_check_fires_on_dev_test_overlap(vd):
    shared = "shared clause"
    rep = vd.Report()
    vd.check_leakage(rep, None, _hashes(["a"], ["b", shared], ["c", shared]))
    assert rep.failures
    assert "validation∩test=1" in rep.failures[0]


def test_leakage_check_passes_on_disjoint_fixture(vd):
    rep = vd.Report()
    vd.check_leakage(rep, None, _hashes(["a", "b"], ["c"], ["d"]))
    assert rep.failures == []


def test_within_split_duplicates_are_reported_but_not_a_failure(vd):
    """Duplicates inside one split are a corpus property, not a leak."""
    rep = vd.Report()
    vd.check_leakage(rep, None, _hashes(["dup", "dup", "x"], ["c"], ["d"]))
    assert rep.failures == []
    assert "| train | 3 | 2 | 1 | 2× |" in rep.text()


def test_real_dataset_has_no_cross_split_leakage(ledgar, vd):
    """The check that silently invalidates projects, run against the real data."""
    hashes = {
        name: [text_sha256(t) for t in ledgar[name]["text"]]
        for name in ("train", "validation", "test")
    }
    rep = vd.Report()
    vd.check_leakage(rep, ledgar, hashes)
    assert rep.failures == [], rep.failures


# ------------------------------------------------------------------ shape guards


class _FakeSplit:
    def __init__(self, n, columns=("text", "label")):
        self._n = n
        self.column_names = list(columns)

    def __len__(self):
        return self._n


def test_wrong_split_size_raises(monkeypatch):
    import datasets

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **k: {
            "train": _FakeSplit(59_999),
            "validation": _FakeSplit(10_000),
            "test": _FakeSplit(10_000),
        },
    )
    with pytest.raises(DatasetShapeError, match="split sizes changed"):
        load_ledgar()


def test_missing_split_raises(monkeypatch):
    import datasets

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **k: {"train": _FakeSplit(60_000), "test": _FakeSplit(10_000)},
    )
    with pytest.raises(DatasetShapeError, match="missing splits"):
        load_ledgar()


def test_missing_column_raises(monkeypatch):
    import datasets

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **k: {
            "train": _FakeSplit(60_000, columns=("provision", "label")),
            "validation": _FakeSplit(10_000),
            "test": _FakeSplit(10_000),
        },
    )
    with pytest.raises(DatasetShapeError, match="columns"):
        load_ledgar()


def test_real_dataset_shape_holds(ledgar):
    assert len(ledgar["train"]) == 60_000
    assert len(ledgar["validation"]) == 10_000
    assert len(ledgar["test"]) == 10_000
