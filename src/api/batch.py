"""Batch API client: request construction, submission, polling and strict joining.

Verified Batch API behaviour this module is built to (see CLAUDE.md):

* 50% discount; most batches finish under an hour, bounded at 24.
* Max 10,000 requests per batch; results retrievable for 29 days.
* **Results return in ANY ORDER.** They are joined on ``custom_id``, never on list
  position. This module asserts that every submitted id came back and that no
  unexpected id appears.
* Prompt caching works with batches but hits are BEST-EFFORT, because requests process
  concurrently and out of order. So the cache TTL is set explicitly to ``"1h"`` rather
  than the 5-minute default, and a small warm-up batch writes the prefix first.
* A batch can PARTIALLY succeed; per-request failures are surfaced individually rather
  than failing the whole run.

The batch id is persisted to disk the instant it is known, before any polling, so a
crashed process resumes an in-flight batch instead of resubmitting and double-spending.
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

__all__ = [
    "CACHE_TTL",
    "DEFAULT_BATCH_STATE_DIR",
    "MAX_BATCH_REQUESTS",
    "BatchClient",
    "BatchJoinError",
    "BatchRequest",
    "BatchState",
    "JoinedResults",
    "build_batch_requests",
    "custom_id_for",
    "join_on_custom_id",
    "parse_custom_id",
]

MAX_BATCH_REQUESTS = 10_000
CACHE_TTL = "1h"  # NOT the 5-minute default; see module docstring.
WARMUP_DEFAULT_N = 10
DEFAULT_BATCH_STATE_DIR = Path(__file__).resolve().parents[2] / "results" / "batches"

# Anthropic custom_id charset: alphanumerics, underscore and hyphen.
_CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class BatchJoinError(RuntimeError):
    """Submitted and returned custom_ids do not correspond exactly."""


def custom_id_for(manifest_name: str, row_index: int, *, prefix: str = "") -> str:
    """Deterministic ``custom_id`` for a manifest row.

    Derived from the manifest name and the row index, so any result can be traced back
    to an exact example without consulting submission order. Zero-padded to 6 digits so
    ids sort naturally in logs.
    """
    if row_index < 0:
        raise ValueError(f"row_index must be >= 0, got {row_index}")
    base = manifest_name.replace(":", "_").replace("/", "_")
    cid = f"{prefix}{base}-{row_index:06d}" if prefix else f"{base}-{row_index:06d}"
    if not _CUSTOM_ID_RE.match(cid):
        raise ValueError(f"custom_id {cid!r} is not in the permitted charset")
    return cid


def parse_custom_id(custom_id: str) -> tuple[str, int]:
    """Inverse of :func:`custom_id_for`: ``(manifest_name, row_index)``."""
    manifest, _, index = custom_id.rpartition("-")
    if not manifest or not index.isdigit():
        raise ValueError(f"malformed custom_id {custom_id!r}")
    return manifest, int(index)


@dataclass
class BatchRequest:
    """One request in a batch, in the shape the Messages API expects."""

    custom_id: str
    params: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"custom_id": self.custom_id, "params": self.params}


def build_batch_requests(
    manifest_name: str,
    row_indices: Sequence[int],
    clause_texts: Sequence[str],
    *,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    cache_ttl: str = CACHE_TTL,
    prefix: str = "",
    extra_params: dict[str, Any] | None = None,
) -> list[BatchRequest]:
    """Build the batch payload.

    The system prompt is sent as a single content block carrying
    ``cache_control`` with the given TTL. It is the whole cacheable prefix and never
    varies per request; only the clause differs, and it lives in the user message. That
    keeps the cache breakpoint at the end of the static portion.
    """
    if len(row_indices) != len(clause_texts):
        raise ValueError(
            f"{len(row_indices)} row indices but {len(clause_texts)} clause texts"
        )
    if not row_indices:
        raise ValueError("no rows to submit")
    if len(row_indices) > MAX_BATCH_REQUESTS:
        raise ValueError(
            f"{len(row_indices)} requests exceeds the {MAX_BATCH_REQUESTS:,}-request "
            "batch limit; split the run"
        )
    if cache_ttl != CACHE_TTL:
        # Not an error, but it must be a deliberate choice, not a default.
        if cache_ttl not in ("5m", "1h"):
            raise ValueError(f"cache_ttl must be '5m' or '1h', got {cache_ttl!r}")

    system_block = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
        }
    ]

    requests: list[BatchRequest] = []
    seen: set[str] = set()
    for row_index, clause in zip(row_indices, clause_texts):
        cid = custom_id_for(manifest_name, row_index, prefix=prefix)
        if cid in seen:
            raise ValueError(f"duplicate custom_id {cid!r} — row indices must be unique")
        seen.add(cid)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_block,
            "messages": [{"role": "user", "content": clause}],
        }
        params.update(extra_params or {})
        requests.append(BatchRequest(custom_id=cid, params=params))
    return requests


@dataclass
class BatchState:
    """Persisted the moment a batch id is known, so a crash cannot cause a resubmit."""

    run_id: str
    batch_id: str
    provider: str
    model: str
    manifest_name: str
    n_requests: int
    custom_ids: list[str]
    cache_ttl: str
    submitted_at_utc: str
    is_warmup: bool = False
    status: str = "submitted"
    completed_at_utc: str | None = None

    def path(self, directory: str | Path = DEFAULT_BATCH_STATE_DIR) -> Path:
        return Path(directory) / f"{self.run_id}.json"

    def save(self, directory: str | Path = DEFAULT_BATCH_STATE_DIR) -> Path:
        path = self.path(directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(asdict(self), fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(
        cls, run_id: str, directory: str | Path = DEFAULT_BATCH_STATE_DIR
    ) -> "BatchState | None":
        path = Path(directory) / f"{run_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))


@dataclass
class JoinedResults:
    """Results joined strictly on ``custom_id``.

    ``succeeded`` maps custom_id -> result body. ``failed`` maps custom_id ->
    ``{"type": ..., "error": ...}`` for requests that failed individually; a batch can
    partially succeed and those failures must be visible, not fatal.
    """

    succeeded: dict[str, Any] = field(default_factory=dict)
    failed: dict[str, Any] = field(default_factory=dict)

    @property
    def n_succeeded(self) -> int:
        return len(self.succeeded)

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    def row_indices(self) -> dict[int, Any]:
        """Successful results rekeyed by manifest row index."""
        return {parse_custom_id(cid)[1]: body for cid, body in self.succeeded.items()}


def join_on_custom_id(
    submitted_custom_ids: Iterable[str],
    results: Iterable[Any],
) -> JoinedResults:
    """Join results to submitted requests on ``custom_id`` alone.

    Results arrive in ANY ORDER, so list position carries no information and is never
    used. Both directions are asserted: every submitted id must come back, and no
    unexpected id may appear.

    Raises:
        BatchJoinError: on a missing id, an unexpected id, or a duplicate id.
    """
    expected = list(submitted_custom_ids)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise BatchJoinError("submitted custom_ids contain duplicates")

    joined = JoinedResults()
    seen: set[str] = set()

    for item in results:
        cid = item.get("custom_id") if isinstance(item, dict) else getattr(item, "custom_id", None)
        if cid is None:
            raise BatchJoinError(f"result carries no custom_id: {item!r}")
        if cid in seen:
            raise BatchJoinError(f"duplicate custom_id in results: {cid!r}")
        seen.add(cid)
        if cid not in expected_set:
            raise BatchJoinError(
                f"UNEXPECTED custom_id {cid!r} in results — it was never submitted. "
                "Results may belong to a different batch; refusing to join."
            )

        result = item.get("result") if isinstance(item, dict) else getattr(item, "result", None)
        rtype = (
            result.get("type") if isinstance(result, dict) else getattr(result, "type", None)
        )
        if rtype == "succeeded":
            body = (
                result.get("message") if isinstance(result, dict)
                else getattr(result, "message", None)
            )
            joined.succeeded[cid] = body
        else:
            joined.failed[cid] = result

    missing = expected_set - seen
    if missing:
        sample = sorted(missing)[:5]
        raise BatchJoinError(
            f"{len(missing)} submitted custom_id(s) did not come back, e.g. {sample}. "
            "Never fill these by position — the mapping would be silently wrong."
        )
    return joined


class BatchAPIClient(Protocol):
    """The provider surface this client needs. Injected, so tests stay offline."""

    def create_batch(self, requests: list[dict[str, Any]]) -> Any: ...
    def retrieve_batch(self, batch_id: str) -> Any: ...
    def batch_results(self, batch_id: str) -> Iterable[Any]: ...


@dataclass
class BatchClient:
    """Submit, poll and retrieve batches, with crash-safe state on disk."""

    client: BatchAPIClient
    provider: str = "anthropic"
    state_dir: Path = DEFAULT_BATCH_STATE_DIR
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- submission

    def submit(
        self,
        run_id: str,
        requests: Sequence[BatchRequest],
        *,
        model: str,
        manifest_name: str,
        cache_ttl: str = CACHE_TTL,
        is_warmup: bool = False,
        resume: bool = True,
    ) -> BatchState:
        """Submit a batch, persisting its id BEFORE anything else can fail.

        If ``run_id`` already has persisted state and ``resume`` is True, the existing
        batch is returned rather than resubmitted — resubmitting is how a crashed
        process double-spends.
        """
        existing = BatchState.load(run_id, self.state_dir)
        if existing is not None:
            if not resume:
                raise RuntimeError(
                    f"run_id {run_id!r} already has batch {existing.batch_id!r} on disk. "
                    "Resubmitting would double-spend. Use a new run_id, or resume."
                )
            return existing

        if not requests:
            raise ValueError("refusing to submit an empty batch")

        batch = self.client.create_batch([r.as_dict() for r in requests])
        batch_id = batch["id"] if isinstance(batch, dict) else batch.id

        state = BatchState(
            run_id=run_id,
            batch_id=batch_id,
            provider=self.provider,
            model=model,
            manifest_name=manifest_name,
            n_requests=len(requests),
            custom_ids=[r.custom_id for r in requests],
            cache_ttl=cache_ttl,
            submitted_at_utc=datetime.now(timezone.utc).isoformat(),
            is_warmup=is_warmup,
        )
        state.save(self.state_dir)  # persisted immediately, before any polling
        return state

    def warmup_batch(
        self,
        run_id: str,
        requests: Sequence[BatchRequest],
        *,
        model: str,
        manifest_name: str,
        n: int = WARMUP_DEFAULT_N,
        cache_ttl: str = CACHE_TTL,
    ) -> BatchState:
        """Submit a small batch first, to WRITE the cache prefix.

        Batch cache hits are best-effort because requests process concurrently, so the
        main batch may otherwise race and write the prefix many times. A warm-up of a
        few requests pays the write once and lets the main batch read it.
        """
        if n <= 0:
            raise ValueError(f"warm-up size must be positive, got {n}")
        return self.submit(
            run_id,
            list(requests)[:n],
            model=model,
            manifest_name=manifest_name,
            cache_ttl=cache_ttl,
            is_warmup=True,
        )

    # ------------------------------------------------------------------- polling

    def poll(
        self,
        state: BatchState,
        *,
        initial_delay: float = 30.0,
        max_delay: float = 300.0,
        timeout: float = 24 * 3600,
        jitter: float = 0.1,
    ) -> BatchState:
        """Poll until the batch ends, with exponential backoff and jitter.

        The 24-hour default timeout is the API's own bound; most batches finish inside
        an hour.
        """
        deadline = time.monotonic() + timeout
        delay = initial_delay

        while True:
            batch = self.client.retrieve_batch(state.batch_id)
            status = (
                batch.get("processing_status")
                if isinstance(batch, dict)
                else getattr(batch, "processing_status", None)
            )
            if status == "ended":
                state.status = "ended"
                state.completed_at_utc = datetime.now(timezone.utc).isoformat()
                state.save(self.state_dir)
                return state
            if status in ("canceled", "expired", "errored"):
                state.status = status
                state.save(self.state_dir)
                raise RuntimeError(
                    f"batch {state.batch_id} ended in status {status!r}; results for "
                    "any completed requests remain retrievable for 29 days"
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"batch {state.batch_id} still {status!r} after {timeout / 3600:.1f}h. "
                    "State is on disk; re-run to resume rather than resubmitting."
                )

            self.sleep(delay * (1 + random.uniform(-jitter, jitter)))
            delay = min(delay * 2, max_delay)

    # ----------------------------------------------------------------- retrieval

    def retrieve(self, state: BatchState) -> JoinedResults:
        """Fetch results and join them strictly on ``custom_id``."""
        return join_on_custom_id(state.custom_ids, self.client.batch_results(state.batch_id))
