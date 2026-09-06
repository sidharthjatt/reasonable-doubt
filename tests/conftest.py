"""Shared fixtures. The real dataset is loaded once per session and cached."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def ledgar():
    """The real LEDGAR dataset. Skips if it cannot be fetched (offline CI)."""
    from src.data.loading import load_ledgar

    try:
        return load_ledgar()
    except Exception as exc:  # network unavailable
        pytest.skip(f"LEDGAR unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="session")
def canonical_labels(ledgar):
    from src.data.loading import label_names

    return label_names(ledgar)
