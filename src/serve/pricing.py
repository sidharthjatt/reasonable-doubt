"""Per-request cost, from configs/costs.yaml and nowhere else (hard rule 5).

Local tiers are priced as amortised capital plus measured energy, exactly as E6's curve
defines them (PREREGISTRATION 3aa/3ac) — not as zero. A local tier costed at zero would
make every cascade look free at volume, which is the assumption E6 exists to test.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.api.cost import compute_cost
from src.eval.breakeven import (
    device_cost_usd,
    energy_usd_per_1k,
    load_hardware,
    resolve_tariff,
)


@dataclass(frozen=True)
class CostEstimate:
    usd: float
    basis: str
    is_estimate: bool
    tariff_is_assumed: bool = False


def local_usd_per_request(*, hw: dict | None = None, allow_assumed_tariff: bool = True,
                          lifetime_requests: int | None = None) -> CostEstimate:
    """Amortised capital + measured marginal energy, per request."""
    hw = load_hardware() if hw is None else hw
    tariff = resolve_tariff(hw, allow_assumed=allow_assumed_tariff)
    n = int(lifetime_requests or hw["assumed_lifetime_requests"])
    capital = device_cost_usd(hw) / n
    energy = energy_usd_per_1k(float(hw["energy_joules_per_request"]),
                               tariff.usd_per_kwh) / 1000.0
    return CostEstimate(
        usd=capital + energy,
        basis=(f"capital {device_cost_usd(hw):.2f} USD / {n:,} lifetime requests "
               f"+ marginal energy at {tariff.usd_per_kwh} USD/kWh"),
        is_estimate=True,          # lifetime_requests is a projection, never a measurement
        tariff_is_assumed=tariff.is_assumed,
    )


def api_usd_for_result(model: str, r, *, batch: bool = False,
                       cache_ttl: str | None = None) -> CostEstimate:
    """Exact API cost from the three usage fields, kept separate (hard rule 10)."""
    usd = compute_cost(model,
                       input_tokens=r.input_tokens,
                       cache_creation_input_tokens=r.cache_creation_input_tokens,
                       cache_read_input_tokens=r.cache_read_input_tokens,
                       output_tokens=r.output_tokens,
                       batch=batch, cache_ttl=cache_ttl)
    return CostEstimate(usd=usd, basis=f"{model} usage fields via configs/costs.yaml",
                        is_estimate=False)
