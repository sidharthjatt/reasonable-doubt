"""The split guard must refuse, not advise."""

import pytest

from src.train.splits import (
    REPORTING_SPLITS,
    SELECTION_SPLITS,
    THRESHOLD_SPLITS,
    ForbiddenSplitError,
    assert_selection_split,
    assert_threshold_split,
)


@pytest.mark.parametrize("name", sorted(SELECTION_SPLITS))
def test_selection_splits_permitted(name):
    assert assert_selection_split(name) == name


@pytest.mark.parametrize("name", sorted(REPORTING_SPLITS))
def test_selection_on_a_reporting_split_is_refused(name):
    """The thing hard rule 6 exists to prevent."""
    with pytest.raises(ForbiddenSplitError, match="REPORTING split"):
        assert_selection_split(name)


@pytest.mark.parametrize("name", sorted(THRESHOLD_SPLITS))
def test_selection_on_the_threshold_split_is_refused(name):
    """Selecting a checkpoint on dev AND calibrating the threshold on dev uses dev
    twice — the threshold would be tuned against data the model was chosen to fit."""
    with pytest.raises(ForbiddenSplitError, match="THRESHOLD calibration only"):
        assert_selection_split(name)


@pytest.mark.parametrize("name", sorted(THRESHOLD_SPLITS))
def test_threshold_splits_permitted(name):
    assert assert_threshold_split(name) == name


@pytest.mark.parametrize("name", sorted(REPORTING_SPLITS | SELECTION_SPLITS))
def test_threshold_calibration_off_dev_is_refused(name):
    with pytest.raises(ForbiddenSplitError, match="DEV split only"):
        assert_threshold_split(name)


def test_the_three_roles_are_disjoint():
    assert not (SELECTION_SPLITS & THRESHOLD_SPLITS)
    assert not (SELECTION_SPLITS & REPORTING_SPLITS)
    assert not (THRESHOLD_SPLITS & REPORTING_SPLITS)


def test_unknown_split_refused():
    with pytest.raises(ForbiddenSplitError, match="unknown"):
        assert_selection_split("some_new_split")


def test_training_script_guards_before_doing_expensive_work():
    """The guard must run before the dataset loads, so a mistake fails in seconds."""
    src = (__import__("pathlib").Path(__file__).parents[1]
           / "src" / "train" / "tier0_encoder.py").read_text()
    guard = src.index("assert_selection_split(args.selection_split)")
    assert guard < src.index("import torch"), "guard must precede heavy imports"
    assert guard < src.index("load_data(args.max_length"), "guard must precede data load"


def test_training_script_is_standalone():
    """It must not import from this repo — Kaggle will not have it."""
    src = (__import__("pathlib").Path(__file__).parents[1]
           / "src" / "train" / "tier0_encoder.py").read_text()
    assert "from src." not in src and "import src." not in src


def test_margin_is_invariant_to_temperature_but_max_softmax_is_not():
    """Why temperature scaling is a no-op for a margin router and not for softmax."""
    import numpy as np

    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 100)) * 3

    def margin(a):
        s = np.sort(a, axis=1)
        return s[:, -1] - s[:, -2]

    def max_softmax(a):
        e = np.exp(a - a.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).max(axis=1)

    for T in (0.5, 2.0, 5.0):
        assert np.allclose(margin(z / T), margin(z) / T)          # exact rescale
        assert np.array_equal(np.argsort(margin(z)), np.argsort(margin(z / T)))
        assert not np.array_equal(                                 # softmax reorders
            np.argsort(max_softmax(z)), np.argsort(max_softmax(z / T))
        )
