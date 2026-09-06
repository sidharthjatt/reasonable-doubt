"""Usage-block ingestion (hard rule 10).

Anthropic reports four independent token fields:

    input_tokens                  uncached input
    cache_creation_input_tokens   cache writes
    cache_read_input_tokens       cache hits
    output_tokens                 output

They must never be summed into a single input figure. On a cache-heavy run,
``input_tokens + cache_read_input_tokens`` passed as "input" over-bills by roughly 10x
— and the resulting number still looks plausible, which is exactly why this is a rule
rather than a convention.

Every expected field must be *present*. A missing field raises; it is never defaulted
to zero (rules 10 and 11). Zero and absent are different claims: zero means the API
told us there were no cached tokens, absent means we do not know.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "ANTHROPIC_USAGE_FIELDS",
    "Usage",
    "UsageFieldMissing",
    "UsageNotCostable",
    "parse_usage",
]

ANTHROPIC_USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


class UsageFieldMissing(KeyError):
    """A usage block did not report a field we require. Never defaulted to zero."""


class UsageNotCostable(ValueError):
    """Costing was attempted on usage whose cache fields were never reported."""


@dataclass(frozen=True)
class Usage:
    """Four independent token counts. The ONLY thing ``compute_cost`` may be fed
    from a real response.

    ``cache_creation_input_tokens`` and ``cache_read_input_tokens`` are ``None`` when
    the provider does not report them at all — a free-tier provider, typically. ``None``
    is not zero, and :meth:`as_cost_kwargs` refuses to cost it.
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    model: str | None = None
    provider: str | None = None

    @property
    def cache_fields_reported(self) -> bool:
        return (
            self.cache_creation_input_tokens is not None
            and self.cache_read_input_tokens is not None
        )

    def as_cost_kwargs(self) -> dict[str, int]:
        """Keyword arguments for ``compute_cost``, with the four counts kept separate.

        Raises:
            UsageNotCostable: if either cache field was never reported. Costing such a
                response would require inventing a zero.
        """
        if not self.cache_fields_reported:
            raise UsageNotCostable(
                f"provider {self.provider!r} did not report cache token fields for "
                f"model {self.model!r}; they are null, not zero. Refusing to cost a "
                "response whose cache behaviour is unknown (hard rules 10 and 11)."
            )
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_input_tokens,
            "cache_write_tokens": self.cache_creation_input_tokens,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "output_tokens": self.output_tokens,
        }

    def __add__(self, other: "Usage") -> "Usage":
        """Aggregate across responses, keeping the four fields separate throughout.

        Cache fields aggregate to ``None`` if either side is unreported — an aggregate
        containing an unknown is itself unknown, not a partial sum.
        """
        if not isinstance(other, Usage):
            return NotImplemented

        def add(a: int | None, b: int | None) -> int | None:
            return None if a is None or b is None else a + b

        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=add(
                self.cache_creation_input_tokens, other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=add(
                self.cache_read_input_tokens, other.cache_read_input_tokens
            ),
            model=self.model if self.model == other.model else None,
            provider=self.provider if self.provider == other.provider else None,
        )


def _require_int(block: Mapping[str, Any], field: str) -> int:
    if field not in block:
        raise UsageFieldMissing(
            f"usage block is missing {field!r}. Present fields: {sorted(block)}. "
            "Hard rule 10: a missing usage field must fail loudly, never default to 0 "
            "— zero and unknown are different claims and the cost differs by ~10x on "
            "a cache-heavy run."
        )
    value = block[field]
    if value is None:
        raise UsageFieldMissing(f"usage field {field!r} is null; expected an integer")
    if isinstance(value, bool) or not isinstance(value, int):
        raise UsageFieldMissing(f"usage field {field!r} is {value!r}, expected an int")
    if value < 0:
        raise UsageFieldMissing(f"usage field {field!r} is negative: {value}")
    return value


def parse_usage(
    usage_block: Mapping[str, Any] | Any,
    *,
    model: str | None = None,
    provider: str | None = "anthropic",
    cache_fields_reported: bool = True,
) -> Usage:
    """Parse a provider usage block into a :class:`Usage`.

    Args:
        usage_block: The ``usage`` mapping from a response, or an SDK object exposing
            the fields as attributes.
        model: Model that produced it. Recorded so counts cannot be reused across
            models (hard rule 9).
        provider: Provider name.
        cache_fields_reported: Set False **only** for a provider that genuinely does
            not report cache token fields. The two cache fields are then recorded as
            ``None``, and costing the result raises rather than assuming zero.

    Raises:
        UsageFieldMissing: if any required field is absent, null, or not a
            non-negative int.
    """
    if usage_block is None:
        raise UsageFieldMissing("response carried no usage block at all")

    if not isinstance(usage_block, Mapping):
        # SDK objects expose the same names as attributes; absent ones stay absent.
        usage_block = {
            f: getattr(usage_block, f)
            for f in ANTHROPIC_USAGE_FIELDS
            if hasattr(usage_block, f)
        }

    required = ("input_tokens", "output_tokens")
    if cache_fields_reported:
        required = ANTHROPIC_USAGE_FIELDS

    values = {f: _require_int(usage_block, f) for f in required}
    return Usage(
        input_tokens=values["input_tokens"],
        output_tokens=values["output_tokens"],
        cache_creation_input_tokens=values.get("cache_creation_input_tokens"),
        cache_read_input_tokens=values.get("cache_read_input_tokens"),
        model=model,
        provider=provider,
    )
