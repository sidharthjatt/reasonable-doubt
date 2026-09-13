"""The qualification gate. No prediction leaves this service unless the canary passed.

WHY THIS EXISTS SEPARATELY FROM THE CANARY ITSELF. The canary used to run before uvicorn
bound the port, so "the process is serving" and "the artefact is qualified" were the same
fact and needed no representation. Moving the canary off the startup path (§3bp) splits
them apart: the port is open while the answer is still unknown. The property that must
survive that split is

    A PREDICTION IS ONLY EVER RETURNED WHEN THE CANARY HAS PASSED.

so it is made explicit here, as a state machine with one allowing state, rather than left
implicit in the order that things happen at startup.

THE GATE IS CHECKED IN `Cascade.classify`, NOT IN THE HTTP LAYER. An HTTP-level check
would be bypassed by any other caller of the cascade, and "the endpoint forgot to check"
is precisely the failure this is protecting against.

FAILURE IS PERMANENT AND DELIBERATE. A failed canary never retries and never decays back
to pending. The artefact is not qualified on this host, and time does not change that; a
retry loop would turn a hard refusal into an intermittent one, which is the silent
degradation hard rule 11 forbids.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any

__all__ = ["CanaryStatus", "CanaryGate", "Tier0NotQualified"]


class Tier0NotQualified(RuntimeError):
    """Refusing to predict: the canary has not passed on this host."""


class CanaryStatus(str, Enum):
    PENDING = "pending"    # running or not yet started — REFUSES
    PASSED = "passed"      # qualified — the only state that serves
    FAILED = "failed"      # refused permanently, never retried
    SKIPPED = "skipped"    # operator disabled it in configs/serve.yaml — serves, and says so

    @property
    def serves(self) -> bool:
        """PASSED serves. SKIPPED serves because an operator explicitly chose that and
        `/health` reports it. Nothing else serves."""
        return self in (CanaryStatus.PASSED, CanaryStatus.SKIPPED)


class CanaryGate:
    """Thread-safe qualification state, written once by the canary thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = CanaryStatus.PENDING
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        self._reason: str | None = None
        self._started = time.monotonic()
        self._finished_after: float | None = None
        # Progress, so "warming up" can say how far along rather than just spinning.
        self._rows_done = 0
        self._rows_total = 0

    # ---------------------------------------------------------------- transitions
    def _settle(self, status: CanaryStatus, **kw) -> None:
        with self._lock:
            if self._status is not CanaryStatus.PENDING:
                raise RuntimeError(
                    f"canary gate already settled as {self._status.value}; it is "
                    f"write-once and must never be reopened")
            self._status = status
            self._finished_after = time.monotonic() - self._started
            for k, v in kw.items():
                setattr(self, f"_{k}", v)

    def mark_passed(self, result: dict[str, Any]) -> None:
        self._settle(CanaryStatus.PASSED, result=result)

    def mark_failed(self, error: str) -> None:
        self._settle(CanaryStatus.FAILED, error=error)

    def mark_skipped(self, reason: str) -> None:
        self._settle(CanaryStatus.SKIPPED, reason=reason)

    def note_progress(self, done: int, total: int) -> None:
        with self._lock:
            self._rows_done, self._rows_total = done, total

    # ---------------------------------------------------------------- reads
    @property
    def status(self) -> CanaryStatus:
        with self._lock:
            return self._status

    def check_may_serve(self) -> None:
        """Raise unless this host is qualified to answer. Called on EVERY prediction."""
        status = self.status
        if status.serves:
            return
        if status is CanaryStatus.FAILED:
            raise Tier0NotQualified(
                f"TIER 0 IS NOT QUALIFIED ON THIS HOST and this service will not return a "
                f"prediction. The startup canary FAILED: {self._error} This is permanent "
                f"for the life of the process; it is not retried, because the artefact "
                f"does not become qualified by being asked again.")
        raise Tier0NotQualified(
            "WARMING UP. The startup canary has not finished, so this host is not yet "
            "qualified to answer. No prediction is returned until it passes.")

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            d: dict[str, Any] = {
                "status": self._status.value,
                "serves_predictions": self._status.serves,
                "elapsed_s": round(
                    self._finished_after
                    if self._finished_after is not None
                    else time.monotonic() - self._started, 2),
            }
            if self._status is CanaryStatus.PENDING and self._rows_total:
                d["progress"] = {"rows_done": self._rows_done,
                                 "rows_total": self._rows_total}
            if self._result is not None:
                d.update(self._result)
            if self._error is not None:
                d["error"] = self._error
            if self._reason is not None:
                d["reason"] = self._reason
            return d
