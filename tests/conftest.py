"""Shared fixtures. The real dataset is loaded once per session and cached."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def ledgar():
    """The real LEDGAR dataset. Skips if it cannot be fetched (offline CI)."""
    from src.data.loading import DatasetShapeError, load_ledgar

    try:
        return load_ledgar()
    except DatasetShapeError:
        # Hard rule 11: real dataset drift must never be downgraded to a skip.
        raise
    except (OSError, ConnectionError) as exc:
        pytest.skip(f"LEDGAR unreachable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="session")
def canonical_labels(ledgar):
    from src.data.loading import label_names

    return label_names(ledgar)
