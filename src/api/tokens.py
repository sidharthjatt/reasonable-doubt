"""Token counting via the provider's own endpoint (hard rules 9 and 11).

Counts come from the target model's own ``count_tokens`` endpoint. Never from a local
tokenizer, never from tiktoken (which undercounts Claude tokens), never shared between
models: Claude 4.7+ uses a newer tokenizer producing roughly 30% more tokens for the
same text, so Sonnet 5 and Haiku 4.5 report different counts for an identical clause.

If the endpoint is unavailable this module RAISES. There is no fallback and no
estimate — a plausible wrong number here propagates straight into the budget.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from src.api.cache import DEFAULT_CACHE_DIR, content_sha256

__all__ = [
    "TokenCountUnavailable",
    "TokenCounter",
    "TokenCountingClient",
]


class TokenCountUnavailable(RuntimeError):
    """The provider's token-counting endpoint could not be reached or refused."""


class TokenCountingClient(Protocol):
    """Minimal interface a provider client must expose. Injected, so tests never
    touch the network."""

    def count_tokens(
        self, *, model: str, system: str | None, messages: list[Any]
    ) -> int: ...


@dataclass
class TokenCounter:
    """Counts tokens for a model, caching results keyed by ``(model, content_sha256)``.

    The model is part of the key, not decoration: a prompt-only key would serve one
    model's count for another's request, which is precisely what hard rule 9 forbids.
    """

    client: TokenCountingClient
    provider: str = "anthropic"
    cache_root: Path = DEFAULT_CACHE_DIR

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root)
        self._dir = self.cache_root / "token_counts"
        self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, model: str, digest: str) -> Path:
        safe_model = model.replace("/", "_")
        return self._dir / self.provider / safe_model / f"{digest}.json"

    def count_tokens(
        self,
        model: str,
        messages: Iterable[Any],
        system: str | None = None,
    ) -> int:
        """Token count for this exact ``(model, system, messages)``.

        Raises:
            TokenCountUnavailable: if the endpoint fails, or returns anything other
                than a positive integer. Never returns an estimate.
        """
        if not model:
            raise ValueError("model is required — token counts are per-model (rule 9)")

        messages = list(messages)
        digest = content_sha256(system, messages)
        path = self._path(model, digest)

        if path.exists():
            with open(path, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("model") != model:
                raise TokenCountUnavailable(
                    f"{path}: cached count is for model {cached.get('model')!r}, not "
                    f"{model!r}. Token counts are never shared across models."
                )
            self.hits += 1
            return int(cached["input_tokens"])

        self.misses += 1
        try:
            count = self.client.count_tokens(model=model, system=system, messages=messages)
        except Exception as exc:  # provider error, auth, network
            # Hard rule 11: raise. A local tokenizer here would be ~30% wrong for
            # Claude 4.7+ and would silently corrupt every budget estimate.
            raise TokenCountUnavailable(
                f"count_tokens failed for model {model!r}: {type(exc).__name__}: {exc}. "
                "No fallback is permitted — token counts must come from the target "
                "model's own endpoint."
            ) from exc

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise TokenCountUnavailable(
                f"count_tokens returned {count!r} for model {model!r}; expected a "
                "positive integer"
            )

        self._write(path, {
            "provider": self.provider,
            "model": model,
            "content_sha256": digest,
            "input_tokens": count,
        })
        return count

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def stats(self) -> dict[str, int | float]:
        lookups = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookups": lookups,
            "hit_rate": self.hits / lookups if lookups else 0.0,
        }
