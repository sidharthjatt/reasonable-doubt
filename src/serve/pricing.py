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
                       cache_ttl: str = "1h") -> CostEstimate:
    """Exact API cost from the three input usage fields, kept separate (hard rule 10).

    Routes through ``Usage.as_cost_kwargs()`` rather than passing token counts by hand.
    That matters for two reasons the previous implementation got wrong:

    * ``as_cost_kwargs`` emits ``compute_cost``'s ACTUAL parameter names. The earlier
      version passed ``cache_creation_input_tokens=`` / ``cache_read_input_tokens=``,
      which ``compute_cost`` does not accept — so this function raised ``TypeError`` on
      every real result. It never fired because the only caller was a stub reporting
      zero tokens, which took a different branch.
    * It REFUSES to cost usage whose cache fields were never reported, instead of
      quietly treating "unknown" as zero (hard rules 10 and 11).
    """
    if getattr(r, "usage", None) is None:
        raise ValueError(
            f"cannot cost a {model} result that carries no parsed usage block. "
            f"Reconstructing one from the flattened token ints would turn an "
            f"unreported cache field into a zero (hard rule 10).")
    usd = compute_cost(model, **r.usage.as_cost_kwargs(),
                       batch=batch, cache_ttl=cache_ttl)
    return CostEstimate(
        usd=usd,
        basis=(f"{model} usage fields (input / cache_write / cache_read / output kept "
               f"separate) via configs/costs.yaml; batch={batch}, cache_ttl={cache_ttl}"),
        is_estimate=False)
