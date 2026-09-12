"""Startup phase timing, printed to stdout so a managed platform's logs carry it.

WHY THIS EXISTS. The Cloud Run cold start is ~150s, of which the canary reported 133.5s
against ~6.5s for the same 200 rows locally (200 / 31 req/s). A 20x gap is not explained
by "the runner is slower", so the startup path needed a per-phase split rather than one
total. `/health` cannot answer this: by the time it is reachable, startup is over.

Phases are printed as they complete, not buffered to the end, so a startup that dies
part-way still says how far it got and what it was doing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

__all__ = ["phase", "phases_summary"]

_PHASES: list[tuple[str, float]] = []


@contextmanager
def phase(name: str, **detail):
    """Time a named startup phase and print it when it finishes."""
    t0 = time.monotonic()
    try:
        yield
    finally:
        dt = time.monotonic() - t0
        _PHASES.append((name, dt))
        extra = "".join(f" {k}={v}" for k, v in detail.items())
        print(f"STARTUP PHASE {name}: {dt:.2f}s{extra}", flush=True)


def phases_summary() -> str:
    total = sum(dt for _, dt in _PHASES)
    parts = ", ".join(f"{n} {dt:.1f}s ({100 * dt / total:.0f}%)" for n, dt in _PHASES) \
        if total else "none recorded"
    return f"STARTUP BREAKDOWN (total measured {total:.1f}s): {parts}"
