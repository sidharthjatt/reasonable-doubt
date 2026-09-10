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
from src.api.ledger import DEFAULT_LEDGER_PATH, SpendLedger
from src.api.providers import AnthropicProvider
from src.api.usage import Usage
from src.data.labels import LabelNormalizer
from src.data.prompts import render_zeroshot
from src.data.schema import ParseFailure, parse_response
from src.serve.tiers import TierResult

__all__ = ["SpendCapReached", "Tier2Claude", "Tier2Unavailable"]

CACHE_TTL = "1h"          # CLAUDE.md: 1h, never the 5-minute default
API_KEY_ENV = "ANTHROPIC_API_KEY"

# MATCHED TO STAGE 1 (scripts/submit_batch.py -> build_batch_requests). Both constants
# below are differences that were found by diffing this request against the one the
# zero-shot baseline actually sent, and both would have changed the result:
#
# 1. TEMPERATURE IS OMITTED, NOT SET TO 0.0. Stage 1 passes temperature=None with the
#    note "Sonnet 5 rejects it (`temperature` is deprecated for this model)". Sending
#    temperature=0.0 here would have been rejected by the API on EVERY escalation. None
#    means omit the key entirely — a null would still be a rejected field.
# 2. THINKING IS EXPLICITLY DISABLED. Stage 1 sends {"thinking": {"type": "disabled"}}.
#    Leaving it unset lets the model's default apply, which changes both the output shape
#    (a reasoning block ahead of the JSON) and the output token bill. The deployed tier
#    must send the same request the baseline was measured with, or the two are not
#    comparable and the cost figure describes a different request.
TEMPERATURE = None
THINKING = {"type": "disabled"}


class Tier2Unavailable(RuntimeError):
    """Escalation was attempted but could not be made. The caller must not fake one."""


class SpendCapReached(Tier2Unavailable):
    """The configured spend cap is reached. Escalation is refused, not deferred.

    A subclass of Tier2Unavailable on purpose: the cascade already routes that to
    `escalation_skipped=true` with the reason reported, which is exactly the required
    behaviour. A distinct type exists so a test can tell the two causes apart.
    """


@dataclass
class Tier2Claude:
    """Zero-shot Claude classification, cached to disk before use."""

    model: str
    labels: list[str]
    max_output_tokens: int
    temperature: float | None = TEMPERATURE
    batch: bool = False
    name: str = "tier2"
    is_stub: bool = False
    cache: ResponseCache | None = None
    # Spend controls. `spend_cap_usd` is None only when the operator has explicitly
    # disabled the cap; it is never defaulted to "unlimited" by omission.
    spend_cap_usd: float | None = None
    ledger: SpendLedger | None = None
    run_id: str = "serve_tier0_sonnet5"
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.cache = self.cache or ResponseCache()
        self.ledger = self.ledger or SpendLedger(DEFAULT_LEDGER_PATH)
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

    def spent_usd(self) -> float:
        """Real money recorded so far, read from the ledger on every call.

        Not cached in memory: the ledger is append-only and may be written by another
        process, and a stale in-memory total would let the cap be overrun silently.
        """
        return self.ledger.cumulative_usd()

    def cap_reached(self) -> bool:
        if self.spend_cap_usd is None:
            return False
        return self.spent_usd() >= self.spend_cap_usd

    def _assert_can_spend(self) -> None:
        if self.spend_cap_usd is None:
            return
        spent = self.spent_usd()
        if spent >= self.spend_cap_usd:
            raise SpendCapReached(
                f"SPEND CAP REACHED: ${spent:.4f} recorded in "
                f"results/spend_ledger.jsonl against a cap of "
                f"${self.spend_cap_usd:.2f} (tier2.spend_cap_usd in configs/serve.yaml). "
                f"Escalation is REFUSED; the Tier 0 answer is returned unescalated. "
                f"This is a stop, not a throttle — raise the cap deliberately or let "
                f"the service keep answering from Tier 0.")

    # ------------------------------------------------------------------- the request

    def _params(self) -> dict[str, Any]:
        """Decoding params for the cache key AND the request. Kept in one place so the
        key cannot describe a request that differs from the one actually sent."""
        params: dict[str, Any] = {"max_tokens": self.max_output_tokens,
                                  "thinking": THINKING}
        if self.temperature is not None:      # None means OMIT, never send null
            params["temperature"] = self.temperature
        return params

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

        # A CACHE HIT SPENDS NOTHING, so it is served before the cap is consulted and
        # is never written to the ledger. Blocking a free answer on a spend cap, or
        # billing it a second time, would both be wrong.
        entry = self.cache.get_or_none(key)
        if entry is not None:
            return self._to_result(entry.response, entry.usage, cache_hit=True)

        # Only a real call reaches the cap check.
        self._assert_can_spend()

        raw, usage_block = self._call_api(text)
        # BEFORE use, not after: a response consumed and then lost on a crash would be
        # money spent with no record and would be re-sent on the next request.
        self.cache.put(key, raw, usage=usage_block, request_params=self._params())
        result = self._to_result(raw, usage_block, cache_hit=False)
        # The money is already gone by this point, so the ledger write is not optional
        # and is not conditional on anything downstream succeeding (hard rule 12:
        # append-only, version-controlled, the record of real money spent).
        self.ledger.record_actual(
            self.run_id, result.usage, provider="anthropic", model=self.model,
            batch=self.batch, cache_ttl=CACHE_TTL, n_requests=1,
            notes="live escalation from src/serve (interactive, not batch)")
        return result

    def _call_api(self, text: str) -> tuple[str, dict[str, Any]]:
        import anthropic

        client = self._client or anthropic.Anthropic()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": [{"type": "text", "text": self._system.text,
                        "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}],
            "messages": self._messages(text),
            **self._params(),          # max_tokens + thinking (+ temperature if set)
        }
        resp = client.messages.create(**kwargs)
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
                "thinking": THINKING, "cache_ttl": CACHE_TTL,
                "available": self.available,
                "spend_cap_usd": self.spend_cap_usd,
                "spent_usd": self.spent_usd(),
                "cap_reached": self.cap_reached(),
                "system_prompt_sha256": self._system.sha256}
