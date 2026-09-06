"""Provider adapters.

Development iterations go to free-tier providers (Gemini / Groq). Only final locked
runs go to Claude. All providers share the same cache and the same usage parsing.

The important asymmetry: free-tier providers generally do not report cached-token
fields. Where they do not, those fields are recorded as ``None`` — never zero — and
:meth:`Usage.as_cost_kwargs` raises if anyone tries to cost them. Zero would be a claim
that no caching occurred; ``None`` is the truth, which is that we do not know.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from src.api.cache import CacheKey, ResponseCache
from src.api.usage import Usage, parse_usage

__all__ = [
    "PROVIDERS",
    "AnthropicProvider",
    "FreeTierProvider",
    "GeminiProvider",
    "GroqProvider",
    "Provider",
    "get_provider",
]


class Provider(Protocol):
    """The surface the rest of the project depends on."""

    name: str
    supports_batch: bool
    reports_cache_tokens: bool

    def parse_usage(self, usage_block: Any, *, model: str) -> Usage: ...
    def cache_key(self, model: str, *, system: str | None, messages: list[Any],
                  params: dict[str, Any] | None) -> CacheKey: ...


@dataclass
class BaseProvider:
    """Shared cache-key construction and usage parsing."""

    name: str
    supports_batch: bool
    reports_cache_tokens: bool
    is_free_tier: bool = False
    client: Any = None
    cache: ResponseCache | None = None

    def cache_key(
        self,
        model: str,
        *,
        system: str | None = None,
        messages: Iterable[Any] = (),
        params: dict[str, Any] | None = None,
    ) -> CacheKey:
        """Build the four-part cache key. The provider name is part of it, so the same
        model id served by two providers cannot share cached responses."""
        return CacheKey.build(
            self.name, model, system=system, messages=list(messages), params=params
        )

    def parse_usage(self, usage_block: Any, *, model: str) -> Usage:
        """Parse a usage block, recording unreported cache fields as ``None``."""
        return parse_usage(
            usage_block,
            model=model,
            provider=self.name,
            cache_fields_reported=self.reports_cache_tokens,
        )


@dataclass
class AnthropicProvider(BaseProvider):
    """Claude. Batch + 1h prompt caching; reports all four usage fields."""

    name: str = "anthropic"
    supports_batch: bool = True
    reports_cache_tokens: bool = True
    is_free_tier: bool = False


@dataclass
class FreeTierProvider(BaseProvider):
    """Base for development providers.

    ``reports_cache_tokens`` defaults to False: the cache token fields come back
    ``None`` and any attempt to cost the response raises rather than assuming zero.
    """

    supports_batch: bool = False
    reports_cache_tokens: bool = False
    is_free_tier: bool = True


@dataclass
class GeminiProvider(FreeTierProvider):
    name: str = "gemini"


@dataclass
class GroqProvider(FreeTierProvider):
    name: str = "groq"


PROVIDERS: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Construct a provider adapter by name."""
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; have {sorted(PROVIDERS)}")
    return PROVIDERS[name](**kwargs)
