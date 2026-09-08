"""FastAPI 3-tier cascade service. STRUCTURE ONLY — real models plug in behind `Tier`.

Returns `tier_used` and `estimated_cost` for every request. Costs come from
configs/costs.yaml through src/serve/pricing.py and from nowhere else (hard rule 5).

What this deliberately does NOT do yet: load a model, call an API, or spend anything.
Every tier is a stub until wired, and the response says so on every request — a stub
that answered plausibly would be indistinguishable from a working cascade, which is the
failure class this project keeps paying for (PREREGISTRATION 3e).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.serve.pricing import CostEstimate, local_usd_per_request
from src.serve.tiers import StubTier, Tier, TierResult


@dataclass
class RouterPolicy:
    """Escalate when the current tier's confidence is below its threshold.

    Thresholds are placeholders until E5 calibrates them on dev_2000 (hard rule 1); they
    are named `*_uncalibrated` so a calibrated value cannot be confused with a guess.
    """
    tier0_uncalibrated: float = 0.99
    tier1_uncalibrated: float = 0.99
    calibrated: bool = False

    def escalate(self, r: TierResult) -> bool:
        if r.label is None:
            return True
        th = self.tier0_uncalibrated if r.tier == "tier0" else self.tier1_uncalibrated
        return (r.confidence or 0.0) < th


@dataclass
class Cascade:
    tier0: Tier
    tier1: Tier
    tier2: Tier
    policy: RouterPolicy = field(default_factory=RouterPolicy)

    def classify(self, text: str) -> dict:
        trace: list[str] = []
        costs: list[CostEstimate] = []
        result: TierResult | None = None

        for tier in (self.tier0, self.tier1, self.tier2):
            result = tier.classify(text)
            trace.append(tier.name)
            costs.append(self._cost(tier, result))
            if tier is self.tier2 or not self.policy.escalate(result):
                break

        assert result is not None
        total = sum(c.usd for c in costs)
        return {
            "label": result.label,
            "confidence": result.confidence,
            "tier_used": result.tier,
            "tiers_invoked": trace,
            "estimated_cost": {
                "usd": total,
                "usd_per_1k": total * 1000.0,
                "per_tier": [{"tier": t, "usd": c.usd, "basis": c.basis,
                              "is_estimate": c.is_estimate} for t, c in zip(trace, costs)],
                "is_estimate": any(c.is_estimate for c in costs),
                "tariff_is_assumed": any(c.tariff_is_assumed for c in costs),
            },
            # Never omitted and never defaulted to False: a stubbed cascade must be
            # impossible to mistake for a working one, in the response itself.
            "stub_tiers": [t for t in trace if getattr(
                {self.tier0.name: self.tier0, self.tier1.name: self.tier1,
                 self.tier2.name: self.tier2}[t], "is_stub", False)],
            "router_calibrated": self.policy.calibrated,
        }

    def _cost(self, tier: Tier, r: TierResult) -> CostEstimate:
        if r.tier in ("tier0", "tier1"):
            return local_usd_per_request()
        if r.input_tokens or r.output_tokens:
            from src.serve.pricing import api_usd_for_result
            return api_usd_for_result("claude-sonnet-5", r, batch=False)
        return CostEstimate(usd=0.0, basis="tier2 stub made no API call",
                            is_estimate=True)


def default_cascade() -> Cascade:
    return Cascade(tier0=StubTier("tier0"), tier1=StubTier("tier1"),
                   tier2=StubTier("tier2", confidence=1.0))


def create_app():
    """Build the FastAPI app. Imported lazily so the cascade is testable without it."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    class ClassifyRequest(BaseModel):
        text: str

    app = FastAPI(title="Reasonable Doubt — 3-tier cascade",
                  description="Structure only; tiers are stubs until models are trained.")
    cascade = default_cascade()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok",
                "stub_tiers": [t.name for t in (cascade.tier0, cascade.tier1, cascade.tier2)
                               if t.is_stub],
                "router_calibrated": cascade.policy.calibrated}

    @app.post("/classify")
    def classify(req: ClassifyRequest) -> dict:
        return cascade.classify(req.text)

    return app
