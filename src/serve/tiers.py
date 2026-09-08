"""Tier protocol and stand-ins. The real models plug in behind `Tier`.

Structure only: no model is loaded here. Each tier is a `Tier` implementation returning a
`TierResult`, so the cascade can be wired, tested and costed before Tier 0 or Tier 1
exists. A stand-in that returns a fixed answer is honest about being one — it reports
`is_stub=True`, and the service refuses to hide that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TierResult:
    label: str | None
    confidence: float | None
    tier: str
    is_stub: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@runtime_checkable
class Tier(Protocol):
    name: str
    is_stub: bool

    def classify(self, text: str) -> TierResult: ...


class StubTier:
    """Placeholder until a real model is wired in.

    Deliberately NOT a "reasonable default": it returns a fixed label and a confidence
    that always routes onward, so a stub silently standing in for a trained model shows
    up as 100% escalation rather than as plausible output (hard rule 11).
    """

    def __init__(self, name: str, *, label: str = "Adjustments",
                 confidence: float = 0.0, input_tokens: int = 0, output_tokens: int = 0):
        self.name, self.is_stub = name, True
        self._label, self._conf = label, confidence
        self._in, self._out = input_tokens, output_tokens

    def classify(self, text: str) -> TierResult:
        return TierResult(label=self._label, confidence=self._conf, tier=self.name,
                          is_stub=True, input_tokens=self._in, output_tokens=self._out)
