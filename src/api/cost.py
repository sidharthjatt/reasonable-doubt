"""Cost computation from the rate card in ``configs/costs.yaml``.

Pure functions: no I/O beyond reading (and caching) the rate card, no network.

Hard rule 5 — every price in this codebase comes from ``configs/costs.yaml``.
There are no numeric rates or multipliers in this file.

Pricing model (verified 2026-09-07 against the official Anthropic pricing docs):

* Rates are quoted per 1,000,000 tokens.
* The Batch API discount and the prompt-caching multipliers **stack**. A cache-read
  token inside a batch bills at ``batch_discount * cache_read_multiplier * base_input``.
* ``input_tokens`` means *uncached* input. Cache reads and cache writes are counted
  separately and billed at their own multipliers — do not double-count them in
  ``input_tokens``.

Token counts are per-model (hard rule 9). This module does not count tokens; it only
prices counts that the caller obtained from the correct model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "CostBreakdown",
    "compute_cost",
    "cost_breakdown",
    "effective_rates",
    "load_rate_card",
    "resolve_model",
    "UnknownModelError",
]

# repo_root/src/api/cost.py -> repo_root/configs/costs.yaml
DEFAULT_RATE_CARD = Path(__file__).resolve().parents[2] / "configs" / "costs.yaml"

TOKENS_PER_UNIT = 1_000_000


class UnknownModelError(KeyError):
    """Raised when a model id or alias is absent from the rate card."""


@lru_cache(maxsize=8)
def load_rate_card(path: str | Path = DEFAULT_RATE_CARD) -> dict[str, Any]:
    """Load and cache the rate card. Cached because it is read on every price call."""
    with open(path, encoding="utf-8") as fh:
        card = yaml.safe_load(fh)
    unit = card.get("rate_unit")
    if unit != "per_1m_tokens":
        raise ValueError(
            f"{path}: rate_unit is {unit!r}; this module only prices 'per_1m_tokens'"
        )
    return card


def resolve_model(model: str, card: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Resolve a model id or alias to ``(provider, model_id, entry)``.

    Matching is exact on the model id first, then on ``alias``. An alias that is
    ambiguous across providers is an error rather than a silent pick.
    """
    matches: list[tuple[str, str, dict[str, Any]]] = []
    for provider, pdata in (card.get("providers") or {}).items():
        models = (pdata or {}).get("models") or {}
        if model in models:
            return provider, model, models[model]
        for model_id, entry in models.items():
            if entry and entry.get("alias") == model:
                matches.append((provider, model_id, entry))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        found = ", ".join(f"{p}/{m}" for p, m, _ in matches)
        raise UnknownModelError(f"alias {model!r} is ambiguous across: {found}")

    known = sorted(
        name
        for pdata in (card.get("providers") or {}).values()
        for name in ((pdata or {}).get("models") or {})
    )
    raise UnknownModelError(f"model {model!r} not in rate card. Known ids: {known}")


def effective_rates(
    model: str,
    *,
    batch: bool = False,
    cache_ttl: str = "5m",
    rate_card: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return per-token USD rates for one model under the given modifiers.

    Keys: ``input``, ``output``, ``cache_read``, ``cache_write``. Multipliers stack
    multiplicatively, so the batch discount applies to cached tokens too.
    """
    card = rate_card if rate_card is not None else load_rate_card()
    provider, model_id, entry = resolve_model(model, card)

    mods = card.get("modifiers") or {}
    write_mults = mods.get("cache_write_multipliers") or {}
    if cache_ttl not in write_mults:
        raise ValueError(
            f"unknown cache_ttl {cache_ttl!r}; rate card defines {sorted(write_mults)}"
        )

    if batch and not entry.get("supports_batch", False):
        raise ValueError(f"{provider}/{model_id} does not support the Batch API")

    base_input = float(entry["input"]) / TOKENS_PER_UNIT
    base_output = float(entry["output"]) / TOKENS_PER_UNIT

    batch_mult = float(mods["batch_discount"]) if batch else 1.0
    read_mult = float(mods["cache_read_multiplier"])
    write_mult = float(write_mults[cache_ttl])

    return {
        "input": base_input * batch_mult,
        "output": base_output * batch_mult,
        "cache_read": base_input * read_mult * batch_mult,
        "cache_write": base_input * write_mult * batch_mult,
    }


class CostBreakdown(dict):
    """Per-component costs in USD, plus ``total``. A plain dict with a ``.total``."""

    @property
    def total(self) -> float:
        return self["total"]


def cost_breakdown(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    *,
    cache_ttl: str = "5m",
    batch: bool = False,
    rate_card: dict[str, Any] | None = None,
) -> CostBreakdown:
    """Itemised cost in USD. Same arguments as :func:`compute_cost`."""
    counts = {
        "input": input_tokens,
        "output": output_tokens,
        "cache_read": cache_read_tokens,
        "cache_write": cache_write_tokens,
    }
    for name, n in counts.items():
        if n < 0:
            raise ValueError(f"{name}_tokens must be >= 0, got {n}")

    rates = effective_rates(model, batch=batch, cache_ttl=cache_ttl, rate_card=rate_card)
    out = CostBreakdown({k: counts[k] * rates[k] for k in counts})
    out["total"] = sum(out.values())
    return out


def compute_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    *,
    cache_ttl: str = "5m",
    batch: bool = False,
    rate_card: dict[str, Any] | None = None,
) -> float:
    """Total cost in USD for one request's token counts.

    Args:
        model: Model id or alias as it appears in the rate card.
        input_tokens: **Uncached** input tokens only.
        output_tokens: Output tokens.
        cache_read_tokens: From ``usage.cache_read_input_tokens``.
        cache_write_tokens: From ``usage.cache_creation_input_tokens``.
        cache_ttl: ``"5m"`` or ``"1h"``; selects the cache-write multiplier. Has no
            effect when ``cache_write_tokens`` is 0.
        batch: Whether this request went through the Batch API.
        rate_card: Pre-loaded rate card; defaults to ``configs/costs.yaml``.

    Returns:
        Cost in USD.
    """
    return cost_breakdown(
        model,
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens,
        cache_ttl=cache_ttl,
        batch=batch,
        rate_card=rate_card,
    )["total"]
