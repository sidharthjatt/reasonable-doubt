"""Tier 2: Claude Sonnet 5, the escalation target (E4b-A).

THREE RULES SHAPE THIS FILE.

* **Hard rule 3.** Every response is written to the on-disk cache BEFORE it is used,
  under the four-part key (provider, model, prompt_sha, params_sha) — never the prompt
  hash alone, because the same prompt costs different amounts on different models.
* **Hard rule 10.** Cost is computed from `input_tokens`, `cache_creation_input_tokens`
  and `cache_read_input_tokens` SEPARATELY. `Usage.as_cost_kwargs()` refuses to cost a
  response whose cache fields were never reported rather than defaulting them to zero.
* **No pretending (hard rule 11).** With no `ANTHROPIC_API_KEY` this tier reports itself
  unavailable and the cascade returns Tier 0's answer marked `escalation_skipped=true`.
  It never fabricates an escalated answer and never silently downgrades to Tier 0 while
  reporting `tier_used="tier2"`.

INTERACTIVE SERVING IS NOT BATCH. The 50% Batch API discount does not apply to a live
request and is not claimed here. Hard rule 4 governs OFFLINE evaluation, which still
runs through the Batch API; this path is the deployed service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from src.api.cache import ResponseCache
from src.api.providers import AnthropicProvider
from src.api.usage import Usage
from src.data.labels import LabelNormalizer
from src.data.prompts import render_zeroshot
from src.data.schema import ParseFailure, parse_response
from src.serve.tiers import TierResult

__all__ = ["Tier2Claude", "Tier2Unavailable"]

CACHE_TTL = "1h"          # CLAUDE.md: 1h, never the 5-minute default
API_KEY_ENV = "ANTHROPIC_API_KEY"


class Tier2Unavailable(RuntimeError):
    """Escalation was attempted with no API key. The caller must not fake a result."""


@dataclass
class Tier2Claude:
    """Zero-shot Claude classification, cached to disk before use."""

    model: str
    labels: list[str]
    max_output_tokens: int
    temperature: float = 0.0
    batch: bool = False
    name: str = "tier2"
    is_stub: bool = False
    cache: ResponseCache | None = None
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.cache = self.cache or ResponseCache()
        self.provider = AnthropicProvider()
        self._normalizer = LabelNormalizer(list(self.labels))
        self._system = render_zeroshot(self.labels)

    # ------------------------------------------------------------------ availability

    @staticmethod
    def api_key_present() -> bool:
        """True only if a non-empty key is in the environment. Never logs the value."""
        return bool(os.environ.get(API_KEY_ENV, "").strip())

    @property
    def available(self) -> bool:
        return self.api_key_present()

    # ------------------------------------------------------------------- the request

    def _params(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_output_tokens}

    def _messages(self, text: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": text}]

    def classify(self, text: str) -> TierResult:
        if not self.available:
            raise Tier2Unavailable(
                f"{API_KEY_ENV} is not set. Refusing to report an escalated result that "
                f"was never obtained; the cascade must mark escalation_skipped instead.")

        key = self.provider.cache_key(
            self.model, system=self._system.text,
            messages=self._messages(text), params=self._params())

        entry = self.cache.get_or_none(key)
        if entry is not None:
            return self._to_result(entry.response, entry.usage, cache_hit=True)

        raw, usage_block = self._call_api(text)
        # BEFORE use, not after: a response consumed and then lost on a crash would be
        # money spent with no record and would be re-sent on the next request.
        self.cache.put(key, raw, usage=usage_block, request_params=self._params())
        return self._to_result(raw, usage_block, cache_hit=False)

    def _call_api(self, text: str) -> tuple[str, dict[str, Any]]:
        import anthropic

        client = self._client or anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            system=[{"type": "text", "text": self._system.text,
                     "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}],
            messages=self._messages(text),
        )
        raw = "".join(getattr(b, "text", "") for b in resp.content)
        usage = resp.usage
        block = {f: getattr(usage, f, None) for f in
                 ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens")}
        return raw, block

    # -------------------------------------------------------------------- the result

    def _to_result(self, raw: str, usage_block: dict[str, Any] | None,
                   *, cache_hit: bool) -> TierResult:
        if usage_block is None:
            raise ValueError(
                "cached Tier 2 entry carries no usage block; it cannot be costed and "
                "will not be defaulted to zero (hard rules 10 and 11)")
        # parse_usage raises UsageFieldMissing rather than defaulting an absent field,
        # and records unreported cache fields as None rather than 0.
        usage = self.provider.parse_usage(usage_block, model=self.model)

        # A format failure is a MEASURED outcome in this project, not a crash: the
        # response was paid for and must still be costed and reported. This catches
        # ParseFailure specifically — never a bare except — and substitutes nothing:
        # the label becomes None, which routes and reports as "no answer", rather than
        # a plausible guess (hard rule 11).
        try:
            parsed = parse_response(raw)
        except ParseFailure:
            label, confidence = None, None
        else:
            label = self._normalizer.normalize(parsed.label)
            confidence = float(parsed.confidence)

        return TierResult(
            label=label,
            confidence=confidence,
            tier=self.name,
            is_stub=False,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            usage=usage,
            api_cache_hit=cache_hit,
        )

    def describe(self) -> dict[str, Any]:
        return {"model": self.model, "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature, "batch": self.batch,
                "cache_ttl": CACHE_TTL, "available": self.available,
                "system_prompt_sha256": self._system.sha256}
