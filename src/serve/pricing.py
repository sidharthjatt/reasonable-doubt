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


# The per_tier_throughput key for each served precision. The top-level
# `energy_joules_per_request` / `assumed_lifetime_requests` fields are the INT8 row's and
# are NOT precision-neutral, which is exactly how §3bh caught every /classify response
# quoting an INT8 cost while the service ran FP32.
_TIER0_ROW = {"int8": "tier0_encoder_onnx_int8", "fp32": "tier0_encoder_onnx_fp32"}


def _tier0_hw(hw: dict, precision: str) -> dict:
    """The measured row for `precision`, or a refusal. Never a fallback (hard rule 11)."""
    try:
        key = _TIER0_ROW[precision]
    except KeyError:
        raise ValueError(f"unknown Tier 0 precision {precision!r}; "
                         f"expected one of {sorted(_TIER0_ROW)}") from None
    row = hw["per_tier_throughput"][key]
    missing = [f for f in ("energy_joules_per_request", "requests_per_second")
               if row.get(f) is None]
    if missing:
        raise MeasurementUnavailable(
            f"configs/costs.yaml has no measured {', '.join(missing)} for {key}. "
            f"REFUSING to price a request against another precision's figures or an "
            f"estimate: the energy term is the local cost curve's asymptote, and "
            f"substituting one would be exactly the silent degradation hard rule 11 "
            f"forbids. Measure it with `sudo python -m src.serve.bench_local "
            f"--onnx-dir <dir> --tier {key} --batch-size 1 --max-length 512 --run <r>`.")
    return row


class MeasurementUnavailable(RuntimeError):
    """A per-precision measurement costs need is absent. Never substituted."""


def local_usd_per_request(*, hw: dict | None = None, allow_assumed_tariff: bool = True,
                          lifetime_requests: int | None = None,
                          precision: str = "fp32") -> CostEstimate:
    """Amortised capital + measured marginal energy, per request, FOR `precision`.

    PRECISION-KEYED SINCE §3bl. It previously read the top-level hardware fields, which
    are the INT8 row's, so every `/classify` response reported an INT8 cost while §3bg
    served FP32 (§3bh recorded this as live and wrong). The served precision now selects
    its own measured energy and throughput, and a precision with no measurement RAISES
    rather than borrowing the other one's.
    """
    hw = load_hardware() if hw is None else hw
    tariff = resolve_tariff(hw, allow_assumed=allow_assumed_tariff)
    row = _tier0_hw(hw, precision)
    if lifetime_requests is not None:
        n = int(lifetime_requests)
    else:
        n = int(round(float(row["requests_per_second"]) * 3600 * 24 * 365
                      * float(hw["device_lifetime_years"]) * float(hw["duty_cycle"])))
    capital = device_cost_usd(hw) / n
    energy = energy_usd_per_1k(float(row["energy_joules_per_request"]),
                               tariff.usd_per_kwh) / 1000.0
    return CostEstimate(
        usd=capital + energy,
        basis=(f"{precision}: capital {device_cost_usd(hw):.2f} USD / {n:,} lifetime "
               f"requests (at {float(row['requests_per_second']):.4f} req/s) "
               f"+ marginal energy {float(row['energy_joules_per_request']):.6f} J/req "
               f"at {tariff.usd_per_kwh} USD/kWh"),
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
