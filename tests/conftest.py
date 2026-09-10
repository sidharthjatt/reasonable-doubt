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


@pytest.fixture(scope="session", autouse=True)
def _spend_ledger_is_not_written_by_tests():
    """FAIL THE SESSION if results/spend_ledger.jsonl changes while tests run.

    Hard rule 12: the ledger is the version-controlled record of real money spent —
    append only, never rewritten, never deleted. It is also the default for
    `Tier2Claude`, which is right in production and a trap in tests: a fixture that
    forgets to inject a throwaway ledger silently appends fabricated spend to the real
    one. That happened (24 entries, restored from git), and a comment asking future
    tests to be careful would not have caught it.

    Snapshot in, compare out. Guards content, not just length, so a rewrite is caught
    as well as an append.
    """
    import hashlib

    from src.api.ledger import DEFAULT_LEDGER_PATH

    def digest() -> str | None:
        if not DEFAULT_LEDGER_PATH.exists():
            return None
        return hashlib.sha256(DEFAULT_LEDGER_PATH.read_bytes()).hexdigest()

    before = digest()
    yield
    after = digest()
    assert after == before, (
        f"TESTS MODIFIED results/spend_ledger.jsonl (sha256 {before} -> {after}). "
        f"It is the record of real money spent (hard rule 12). A test constructed "
        f"something that writes to the default ledger instead of a throwaway one — "
        f"inject `ledger=SpendLedger(tmp_path / 'spend_ledger.jsonl')`. "
        f"Restore the file with: git checkout -- results/spend_ledger.jsonl"
    )
