"""On-disk API response cache (hard rule 3).

Every API response is written here BEFORE use, and no (model, prompt, decoding-params)
combination is ever sent twice.

The cache key
-------------
``(provider, model_id, prompt_sha256, params_sha256)`` — **never the prompt hash
alone**. Two reasons, both of which silently corrupt results if ignored:

* **Model** (hard rule 9). Claude 4.7+ uses a different tokenizer, so the same prompt
  costs different amounts on different models. A prompt-only key would serve Haiku's
  response — and Haiku's token counts — for a Sonnet request. The numbers would be
  cheap, plausible and wrong.
* **Decoding params.** temperature, max_tokens, stop sequences and anything else that
  changes the output. A prompt-only key would serve a temperature-0 response to a
  temperature-1 request.

Both hashes are over canonical serializations, so key equality means the requests were
genuinely identical rather than merely similar.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DEFAULT_CACHE_DIR",
    "CacheCollisionError",
    "CacheEntry",
    "CacheKey",
    "CacheStats",
    "ResponseCache",
    "content_sha256",
    "params_sha256",
]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
CACHE_FORMAT_VERSION = 1

# Request fields that change the output and therefore must be in the key. Anything
# not listed is either non-semantic (metadata) or belongs in the prompt hash.
DECODING_PARAM_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "stop_sequences",
    "seed",
    "response_format",
    "thinking",
)


class CacheCollisionError(RuntimeError):
    """A key already exists on disk with different content."""


def _canonical_json(obj: Any) -> str:
    """Stable serialization: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_sha256(system: str | None, messages: Iterable[Any]) -> str:
    """Hash the full prompt content: system block plus every message.

    Canonical-JSON over the whole structure, so a change anywhere in the content —
    including message roles and ordering — produces a different key.
    """
    return _sha256(_canonical_json({"system": system, "messages": list(messages)}))


def params_sha256(params: dict[str, Any] | None) -> str:
    """Hash the decoding parameters that affect the output.

    Only :data:`DECODING_PARAM_FIELDS` are included, and absent fields are omitted
    rather than defaulted — so a caller that never sets ``top_k`` gets a stable key,
    while a caller that sets it gets a different one.
    """
    params = params or {}
    relevant = {k: params[k] for k in DECODING_PARAM_FIELDS if k in params}
    return _sha256(_canonical_json(relevant))


@dataclass(frozen=True)
class CacheKey:
    """The full identity of a request. All four components are load-bearing."""

    provider: str
    model_id: str
    prompt_sha256: str
    params_sha256: str

    @classmethod
    def build(
        cls,
        provider: str,
        model_id: str,
        *,
        system: str | None = None,
        messages: Iterable[Any] = (),
        params: dict[str, Any] | None = None,
    ) -> "CacheKey":
        """Build a key from the request itself, so callers cannot forget a component."""
        if not provider or not model_id:
            raise ValueError("provider and model_id are required — see hard rule 9")
        return cls(
            provider=provider,
            model_id=model_id,
            prompt_sha256=content_sha256(system, messages),
            params_sha256=params_sha256(params),
        )

    @property
    def digest(self) -> str:
        """One hex digest over all four components; the on-disk filename."""
        return _sha256(
            _canonical_json(
                [self.provider, self.model_id, self.prompt_sha256, self.params_sha256]
            )
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "prompt_sha256": self.prompt_sha256,
            "params_sha256": self.params_sha256,
        }


@dataclass
class CacheEntry:
    """One cached response and everything needed to audit it later."""

    key: CacheKey
    response: Any
    usage: dict[str, Any] | None
    request_params: dict[str, Any]
    timestamp_utc: str
    format_version: int = CACHE_FORMAT_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "key": self.key.as_dict(),
            "provider": self.key.provider,
            "model": self.key.model_id,
            "timestamp_utc": self.timestamp_utc,
            "request_params": self.request_params,
            "usage": self.usage,
            "response": self.response,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            key=CacheKey(**data["key"]),
            response=data["response"],
            usage=data["usage"],
            request_params=data["request_params"],
            timestamp_utc=data["timestamp_utc"],
            format_version=data["format_version"],
        )

    def content_payload(self) -> dict[str, Any]:
        """The parts that must match for two writes of one key to be considered equal.
        Timestamps are excluded — re-receiving an identical response is not a conflict."""
        return {
            "response": self.response,
            "usage": self.usage,
            "request_params": self.request_params,
        }


@dataclass
class CacheStats:
    """Session hit/miss counters plus the on-disk entry count."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    duplicate_writes: int = 0
    entries: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "hits": self.hits,
            "misses": self.misses,
            "lookups": self.lookups,
            "hit_rate": self.hit_rate,
            "writes": self.writes,
            "duplicate_writes": self.duplicate_writes,
        }


class ResponseCache:
    """One JSON file per key under ``cache/``, sharded by digest prefix."""

    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._stats = CacheStats()

    def path_for(self, key: CacheKey) -> Path:
        digest = key.digest
        return self.root / "responses" / digest[:2] / f"{digest}.json"

    def get_or_none(self, key: CacheKey) -> CacheEntry | None:
        """Return the cached entry, or ``None``. Counts a hit or a miss."""
        path = self.path_for(key)
        if not path.exists():
            self._stats.misses += 1
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("format_version") != CACHE_FORMAT_VERSION:
            # Hard rule 11: do not guess at an old layout, and do not silently miss.
            raise CacheCollisionError(
                f"{path}: cache format_version {data.get('format_version')} != "
                f"{CACHE_FORMAT_VERSION}. Migrate or clear the cache explicitly."
            )
        self._stats.hits += 1
        return CacheEntry.from_json(data)

    def put(
        self,
        key: CacheKey,
        response: Any,
        *,
        usage: dict[str, Any] | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Write an entry. Never overwrites silently.

        Re-writing byte-identical content is a no-op. Writing *different* content to an
        existing key raises :class:`CacheCollisionError` — that means either a hash
        collision or, far more likely, a key that is missing a component that actually
        varies the response.
        """
        entry = CacheEntry(
            key=key,
            response=response,
            usage=usage,
            request_params=dict(request_params or {}),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        path = self.path_for(key)

        if path.exists():
            with open(path, encoding="utf-8") as fh:
                existing = CacheEntry.from_json(json.load(fh))
            if existing.content_payload() == entry.content_payload():
                self._stats.duplicate_writes += 1
                return existing
            raise CacheCollisionError(
                f"key {key.digest[:16]}… already cached with DIFFERENT content at "
                f"{path}. Same provider/model/prompt/params must give the same cached "
                "response; if the response legitimately varies, the varying input is "
                "missing from the cache key."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: a crash mid-write must not leave a truncated entry that a later
        # run would read back as a valid response.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entry.to_json(), fh, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

        self._stats.writes += 1
        return entry

    def cache_stats(self) -> CacheStats:
        """Session counters, with ``entries`` recounted from disk."""
        responses = self.root / "responses"
        self._stats.entries = (
            sum(1 for _ in responses.rglob("*.json")) if responses.exists() else 0
        )
        return self._stats
