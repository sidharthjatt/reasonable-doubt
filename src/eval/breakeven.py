"""E6's cost-vs-volume break-even curve.

The deliverable is a CURVE (PREREGISTRATION E6), not a point: USD per 1,000 clauses
against clause volume V, with no amortisation window privileged. Local tiers are
hyperbolic plus a floor; API tiers are flat in V.

    local_usd_per_1k(V) = 1000 * device_cost_usd / V  +  energy_usd_per_1k
    V* = 1000 * device_cost_usd / (api_usd_per_1k - energy_usd_per_1k)

Two values here are NOT measured, and this module exists partly to keep that visible:

* the electricity tariff is an ASSUMPTION supplied in conversation, never read off a
  bill. `resolve_tariff` refuses to use it unless the caller asks by name, and every
  result carries `tariff_is_assumed` (hard rule 11).
* `device_cost_usd` is DERIVED from INR and an FX rate, never stored as USD (hard rule 5
  and the FX-auditability note in configs/costs.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

COSTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "costs.yaml"
JOULES_PER_KWH = 3.6e6


class TariffUnavailableError(RuntimeError):
    """Raised when only an assumed tariff exists and the caller did not ask for it."""


@dataclass(frozen=True)
class Tariff:
    usd_per_kwh: float
    is_assumed: bool
    basis: str


@dataclass(frozen=True)
class BreakevenResult:
    """One point of E6, carrying the provenance of every soft input."""

    v_star: float
    device_cost_usd: float
    api_usd_per_1k: float
    energy_usd_per_1k: float
    v_max: float
    tariff_usd_per_kwh: float
    tariff_is_assumed: bool
    tariff_basis: str
    energy_is_soc_only: bool
    measured_on_trained_artefact: bool

    def local_usd_per_1k(self, volume: float) -> float:
        return 1000.0 * self.device_cost_usd / volume + self.energy_usd_per_1k

    @property
    def dies_before_breakeven(self) -> bool:
        """E6 falsification condition 2: no finite crossover at or below V_max."""
        return self.v_star > self.v_max


def load_hardware(path: Path | None = None) -> dict:
    return yaml.safe_load((path or COSTS_PATH).read_text())["local_hardware"]


def resolve_tariff(hw: dict, *, allow_assumed: bool = False) -> Tariff:
    """Measured tariff if there is one; the assumption only if asked for BY NAME.

    Hard rule 11: a fallback may never substitute an approximation for a measured value.
    The assumed tariff is legitimately useful — E6 needs *some* number to draw the curve —
    so it is available, but only to a caller that has said it wants an estimate, and it
    comes back marked.
    """
    measured = hw.get("electricity_cost_usd_per_kwh_measured")
    if measured is not None:
        return Tariff(float(measured), False, "measured from tariff document")

    assumed = hw.get("electricity_cost_usd_per_kwh_assumed")
    if assumed is None:
        raise TariffUnavailableError(
            "configs/costs.yaml has neither electricity_cost_usd_per_kwh_measured nor "
            "_assumed. E6's energy term cannot be computed without one."
        )
    if not allow_assumed:
        raise TariffUnavailableError(
            f"electricity_cost_usd_per_kwh_measured is null. An assumed value "
            f"({assumed} USD/kWh, a guess at an Indian domestic slab, never read off a "
            f"bill) is available, but will not be substituted silently. Pass "
            f"allow_assumed_tariff=True to use it; the result will carry "
            f"tariff_is_assumed=True and must be reported as such."
        )
    return Tariff(
        float(assumed), True,
        "ASSUMED: ~Rs 8/kWh guess at a Rajasthan domestic slab, supplied in "
        "conversation, never verified against a bill",
    )


def device_cost_usd(hw: dict) -> float:
    """DERIVED from INR and the recorded FX rate. Never a stored USD figure."""
    return float(hw["device_cost_amount"]) / float(hw["fx"]["usd_per_inr_rate"])


def energy_usd_per_1k(joules_per_request: float, tariff_usd_per_kwh: float) -> float:
    return 1000.0 * (joules_per_request / JOULES_PER_KWH) * tariff_usd_per_kwh


def breakeven(
    api_usd_per_1k: float,
    *,
    hw: dict | None = None,
    allow_assumed_tariff: bool = False,
    tariff_multiplier: float = 1.0,
    energy_multiplier: float = 1.0,
) -> BreakevenResult:
    """Crossover volume V* against a flat API cost per 1,000 clauses.

    `tariff_multiplier` and `energy_multiplier` exist for the published sensitivity
    tables (PREREGISTRATION 3aa, 3ac) — scaling the tariff, and scaling energy to model
    an unmeasured SoC-to-wall correction. Both default to 1.0 so the ordinary path cannot
    accidentally report a scaled figure.
    """
    hw = load_hardware() if hw is None else hw
    tariff = resolve_tariff(hw, allow_assumed=allow_assumed_tariff)
    rate = tariff.usd_per_kwh * tariff_multiplier
    joules = float(hw["energy_joules_per_request"]) * energy_multiplier
    e_per_1k = energy_usd_per_1k(joules, rate)

    if e_per_1k >= api_usd_per_1k:
        raise ValueError(
            f"energy alone (${e_per_1k:.6g}/1k) is at or above the API line "
            f"(${api_usd_per_1k:.6g}/1k): local never wins at any volume, so there is no "
            f"finite crossover. That is an E6 result, not an error to swallow — report it."
        )

    dev = device_cost_usd(hw)
    rps = float(hw["measured_throughput_rps"])
    v_max = rps * 3600 * 24 * 365 * float(hw["device_lifetime_years"]) * float(hw["duty_cycle"])

    return BreakevenResult(
        v_star=1000.0 * dev / (api_usd_per_1k - e_per_1k),
        device_cost_usd=dev,
        api_usd_per_1k=api_usd_per_1k,
        energy_usd_per_1k=e_per_1k,
        v_max=v_max,
        tariff_usd_per_kwh=rate,
        tariff_is_assumed=tariff.is_assumed,
        tariff_basis=tariff.basis,
        energy_is_soc_only=hw.get("power_draw_wall_watts") is None,
        measured_on_trained_artefact=False,
    )
