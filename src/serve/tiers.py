"""Tier protocol and stand-ins. The real models plug in behind `Tier`.

Structure only: no model is loaded here. Each tier is a `Tier` implementation returning a
`TierResult`, so the cascade can be wired, tested and costed before Tier 0 or Tier 1
exists. A stand-in that returns a fixed answer is honest about being one — it reports
`is_stub=True`, and the service refuses to hide that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.api.usage import Usage


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
    # The parsed usage block, carried WHOLE rather than flattened into the four ints
    # above. `Usage` distinguishes "the provider reported 0" from "the provider never
    # reported this field" (None), and `as_cost_kwargs()` refuses to cost the latter.
    # Flattening with `or 0` would erase that distinction at the one point where hard
    # rules 10 and 11 need it to survive.
    usage: "Usage | None" = None
    # True when this result came from the on-disk response cache, i.e. no new money was
    # spent serving it. The dollar figure still reflects what the call cost when it was
    # actually made, so the two facts are reported separately rather than netted.
    api_cache_hit: bool | None = None
    # Top-k (label, score) for the UI, score = softmax over the tier's own logits.
    # None means THIS TIER DOES NOT PRODUCE ONE, not "empty": the API tier returns a
    # label with no distribution behind it, and a `[]` would read as "no candidates".
    top_k: tuple[tuple[str, float], ...] | None = None
    # Full per-class softmax in label order, for the demo's distribution chart. None means
    # this tier does not produce one (the API tier returns a label, not a distribution).
    distribution: tuple[float, ...] | None = None


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
