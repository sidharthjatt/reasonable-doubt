"""FastAPI service — the DEPLOYED architecture: Tier 0 INT8 -> Claude Sonnet 5.

E4b-A. There is no Tier 1, and its absence is a MEASUREMENT rather than an omission:
E4 reached macro-F1 0.7254 against a 0.7923 bar, a 0.0669 miss and 12x the test_3000
sigma, and E4b-A records that no Tier 1 experiment remains in the plan. Wiring a middle
tier here would serve a model the evidence rejected.

WHAT THIS SERVICE WILL NOT DO
-----------------------------
* Start with a Tier 0 that cannot reproduce its reference accuracy on this host. INT8
  kernels degrade to chance on a CPU without AVX-512 VNNI and raise nothing; the target
  (HF Spaces) is x86. See src/serve/canary.py.
* Compute a routing threshold at request time. The threshold is calibrated on dev_2000
  and loaded from a file (hard rule 1).
* Claim an escalation it did not make. With no ANTHROPIC_API_KEY the Tier 0 answer is
  returned with `escalation_skipped=true` and `tier_used="tier0"`.
* Report a price that is not derived from configs/costs.yaml (hard rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from src.serve.config import ServiceConfig
from src.serve.pricing import CostEstimate, api_usd_for_result, local_usd_per_request
from src.serve.tier2 import Tier2Unavailable
from src.serve.tiers import TierResult

__all__ = ["Cascade", "ClassifyRequest", "MarginRouter", "build_cascade", "create_app"]


class ClassifyRequest(BaseModel):
    """Defined at MODULE level, deliberately.

    `from __future__ import annotations` turns every annotation into a string that
    FastAPI resolves against the module namespace. A request model nested inside
    `create_app()` is invisible there, so FastAPI silently reinterprets the parameter as
    a QUERY parameter and every POST /classify returns 422 "field required". The
    endpoint looks wired and is not — and no unit test of `Cascade` can see it, because
    the cascade itself works fine.
    """

    text: str


@dataclass(frozen=True)
class MarginRouter:
    """Escalate when the margin signal falls BELOW a dev-calibrated threshold.

    The threshold arrives already loaded from `router_threshold.json`. Nothing here
    derives, adjusts or adapts it: a threshold that moved with live traffic would be
    fitted to a distribution that is not dev, which is hard rule 1 defeated quietly.
    """

    threshold: float
    signal: str = "margin"

    def escalate(self, r: TierResult) -> bool:
        if r.label is None:
            return True          # nothing to be confident about
        if r.confidence is None:
            raise ValueError(
                f"tier {r.tier} returned no {self.signal}; routing cannot be decided "
                f"and will not be guessed")
        return r.confidence < self.threshold


@dataclass
class Cascade:
    """Tier 0, escalating to Tier 2 only when the router says so AND a key exists."""

    tier0: Any
    tier2: Any
    router: MarginRouter
    config: ServiceConfig

    def classify(self, text: str) -> dict:
        t0 = self.tier0.classify(text)
        margin = t0.confidence
        costs: list[tuple[str, CostEstimate]] = [("tier0", local_usd_per_request())]

        wants_escalation = self.router.escalate(t0)
        escalation_skipped = False
        skip_reason = None
        result, tiers = t0, ["tier0"]

        if wants_escalation:
            if self.tier2.available:
                try:
                    result = self.tier2.classify(text)
                except Tier2Unavailable as exc:
                    # available said yes and classify said no: a race on the env var.
                    # Report the Tier 0 answer and say why, rather than 500-ing or
                    # pretending the escalation happened.
                    result, escalation_skipped = t0, True
                    skip_reason = str(exc)
                else:
                    tiers.append("tier2")
                    costs.append(("tier2", api_usd_for_result(
                        self.config.tier2_model, result,
                        batch=self.config.tier2_batch)))
            else:
                escalation_skipped = True
                skip_reason = (
                    "ANTHROPIC_API_KEY is not set. The router selected this clause for "
                    "escalation; the Tier 0 answer is returned unescalated and is NOT "
                    "the cascade's answer for this row.")

        total = sum(c.usd for _, c in costs)
        return {
            "label": result.label,
            "margin": margin,
            "tier_used": result.tier,
            "tiers_invoked": tiers,
            "estimated_cost_usd": total,
            "escalation_selected": wants_escalation,
            "escalation_skipped": escalation_skipped,
            "escalation_skipped_reason": skip_reason,
            "api_cache_hit": getattr(result, "api_cache_hit", None),
            "cost_detail": {
                "usd_per_1k": total * 1000.0,
                "per_tier": [
                    {"tier": t, "usd": c.usd, "basis": c.basis,
                     "is_estimate": c.is_estimate,
                     "tariff_is_assumed": c.tariff_is_assumed}
                    for t, c in costs],
                "is_estimate": any(c.is_estimate for _, c in costs),
                "tariff_is_assumed": any(c.tariff_is_assumed for _, c in costs),
                "source": "configs/costs.yaml",
            },
            "router": {
                "signal": self.router.signal,
                "threshold": self.router.threshold,
                "calibrated_on": self.config.threshold.calibrated_on,
                "calibrated_artefact": self.config.threshold.artefact,
                "computed_at_request_time": False,
            },
        }


def build_cascade(config: ServiceConfig | None = None, *, tier2=None) -> Cascade:
    """Construct the deployed cascade. Loads the ONNX session once."""
    from src.data.labels import load_labels
    from src.serve.tier0 import Tier0Encoder
    from src.serve.tier2 import Tier2Claude

    config = config or ServiceConfig.load()
    labels = load_labels()

    tier0 = Tier0Encoder(model_dir=config.tier0_model_dir, labels=labels,
                         max_length=config.tier0_max_length)
    tier2 = tier2 if tier2 is not None else Tier2Claude(
        model=config.tier2_model, labels=labels,
        max_output_tokens=config.tier2_max_output_tokens,
        temperature=config.tier2_temperature, batch=config.tier2_batch,
        spend_cap_usd=config.tier2_spend_cap_usd, run_id=config.tier2_run_id)

    return Cascade(tier0=tier0, tier2=tier2, config=config,
                   router=MarginRouter(threshold=config.threshold.threshold,
                                       signal=config.threshold.signal))


def run_startup_canary(cascade: Cascade) -> dict[str, Any]:
    """Refuse to start unless Tier 0 reproduces its reference accuracy on THIS host."""
    from src.data.loading import load_ledgar
    from src.serve.canary import CanarySet, hardware_report, run_canary
    import json

    cfg = cascade.config
    if not cfg.canary_enabled:
        # Explicit, reported, and never the default. A disabled canary is a decision
        # the operator has to make and the response says it was made.
        return {"ran": False, "reason": "disabled in configs/serve.yaml",
                "hardware": hardware_report()}

    canary = CanarySet.from_dict(json.loads(cfg.canary_row_set.read_text()))
    result = run_canary(cascade.tier0, load_ledgar(), canary,
                        floor=cfg.canary_min_accuracy)
    return {"ran": True, "reference_accuracy": cfg.canary_measured_accuracy,
            **result.as_dict()}


def create_app(config: ServiceConfig | None = None, *, tier2=None,
               run_canary_on_start: bool = True):
    """Build the FastAPI app. The canary runs HERE, so a failure prevents startup."""
    from fastapi import FastAPI

    cascade = build_cascade(config, tier2=tier2)
    canary = run_startup_canary(cascade) if run_canary_on_start else {
        "ran": False, "reason": "explicitly skipped by the caller"}

    app = FastAPI(
        title="Reasonable Doubt — Tier 0 INT8 -> Claude Sonnet 5",
        description="Deployed two-tier cascade (E4b-A). No Tier 1: see PREREGISTRATION.")

    @app.get("/health")
    def health() -> dict:
        from src.serve.canary import cpu_isa_flags, onnxruntime_version
        return {
            "status": "ok",
            "architecture": "tier0_int8 -> claude (no tier1; E4b-A)",
            "tier0": cascade.tier0.describe(),
            "tier2": cascade.tier2.describe(),
            "router": cascade.config.threshold.as_dict(),
            "canary": canary,
            "runtime": {
                "onnxruntime_version": onnxruntime_version(),
                # null means UNREADABLE, not absent — see src/serve/canary.py.
                "cpu_isa_flags": cpu_isa_flags(),
            },
            "escalation_enabled": cascade.tier2.available,
        }

    @app.post("/classify")
    def classify(req: ClassifyRequest) -> dict:
        return cascade.classify(req.text)

    return app
