"""The measured local-hardware figures must agree with each other.

`energy_joules_per_request` is DERIVED, not independently measured: it is
marginal SoC power divided by throughput. Nothing enforced that, so the three numbers
could drift apart silently — and worse, a reader comparing the wrong pair could conclude
the measurement was inconsistent when it was not. That happened (PREREGISTRATION 3ab):
LOAD power was divided by throughput, the result missed the reported energy by 1.87%,
and a sound measurement was flagged as shaky.

The lesson is in the shape of these tests: **a derived quantity has to say which baseline
it is defined against, and the check has to use that baseline.** So the marginal identity
is asserted, and the load-based one is asserted to FAIL, pinning the distinction that was
missed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COSTS = ROOT / "configs" / "costs.yaml"

# The harness reports J/request to 3 dp, so agreement is only meaningful to ~1e-3.
# 0.005 leaves room for that rounding and nothing else: the load-vs-marginal error this
# guards against is 0.011 J/req, more than twice the tolerance.
TOL_J = 0.005


@pytest.fixture(scope="module")
def hw() -> dict:
    return yaml.safe_load(COSTS.read_text())["local_hardware"]


def test_energy_per_request_equals_marginal_power_over_throughput(hw: dict) -> None:
    marginal = hw["power_draw_soc_watts_marginal"]
    rps = hw["measured_throughput_rps"]
    reported = hw["energy_joules_per_request"]
    derived = marginal / rps
    assert abs(derived - reported) < TOL_J, (
        f"energy_joules_per_request={reported} does not match "
        f"marginal {marginal} W / {rps} req/s = {derived:.5f} J/req "
        f"(|diff| {abs(derived - reported):.5f} >= {TOL_J}). These are not independent "
        f"measurements: the energy figure IS this quotient, so a mismatch means one of "
        f"the three was edited without the others."
    )


def test_marginal_is_load_minus_idle(hw: dict) -> None:
    load, idle = hw["power_draw_soc_watts_load"], hw["power_draw_soc_watts_idle"]
    marginal = hw["power_draw_soc_watts_marginal"]
    assert abs((load - idle) - marginal) < 0.01, (
        f"marginal {marginal} != load {load} - idle {idle} = {load - idle}"
    )


def test_load_power_is_NOT_the_right_divisor(hw: dict) -> None:
    """Pin the distinction that produced the false-positive flag.

    Dividing LOAD power by throughput gives a number that is close to, but not equal to,
    the reported energy. That near-agreement is what made the mistake plausible. If a
    future edit ever makes these equal, either idle went to zero or someone redefined the
    energy figure against the load baseline — both of which change what E6 is computing,
    and neither should happen silently.
    """
    load = hw["power_draw_soc_watts_load"]
    rps = hw["measured_throughput_rps"]
    reported = hw["energy_joules_per_request"]
    assert abs(load / rps - reported) >= TOL_J, (
        "load/rps now equals energy_joules_per_request. E6's per-clause energy is defined "
        "against MARGINAL power, because idle draw is incurred whether or not Tier 0 "
        "serves anything and is already carried by the capital term. If this is a "
        "deliberate redefinition, change PREREGISTRATION 3aa and this test together."
    )


def test_wall_power_stays_null_until_measured(hw: dict) -> None:
    """`powermetrics` cannot report wall power; only an external meter can.

    A number appearing here means someone put the SoC figure in a field that claims to be
    something else. E6's SoC-only caveat is stated on the basis that this is null.
    """
    assert hw["power_draw_wall_watts"] is None, (
        "power_draw_wall_watts is set. It cannot be obtained from powermetrics; if it "
        "was measured with a meter, update PREREGISTRATION 3aa's SoC-only paragraph and "
        "the sensitivity table, which exist because this value is unknown."
    )


def test_throughput_sd_not_paired_with_a_different_mean(hw: dict) -> None:
    """A spread must describe the mean it sits beside.

    0.24 was the sd of the 28.97 measurement. When throughput was updated to 30.23 the sd
    became unknown, and null says so. A non-null sd here must have arrived with its mean.
    """
    if hw["measured_throughput_rps_sd"] is not None:
        assert hw["measured_throughput_rps"] != 28.97, (
            "sd is set while throughput is back at 28.97 — check these came from the "
            "same measurement rather than being recombined from different ones."
        )


# --------------------------------------------------------------------------------
# The electricity tariff is an ASSUMPTION, not a measurement. These tests exist so it
# cannot quietly become one. See PREREGISTRATION 3ac.
# --------------------------------------------------------------------------------

from src.eval.breakeven import (  # noqa: E402
    TariffUnavailableError,
    breakeven,
    resolve_tariff,
)

SONNET_USD_PER_1K = 0.4483  # §1, batch + cached


def test_tariff_is_not_silently_substituted(hw: dict) -> None:
    """Hard rule 11: an approximation must be requested by name, never defaulted to."""
    if hw["electricity_cost_usd_per_kwh_measured"] is not None:
        pytest.skip("a measured tariff now exists; the fallback path is not in use")
    with pytest.raises(TariffUnavailableError, match="allow_assumed_tariff"):
        resolve_tariff(hw)


def test_any_result_built_on_the_assumed_tariff_says_so(hw: dict) -> None:
    """If _measured is null, EVERY E6 result must carry tariff_is_assumed=True.

    This is the property that stops an assumed input reaching the report unlabelled.
    """
    if hw["electricity_cost_usd_per_kwh_measured"] is not None:
        pytest.skip("a measured tariff now exists")
    for mult in (0.5, 1.0, 2.0, 10.0):
        r = breakeven(SONNET_USD_PER_1K, hw=hw, allow_assumed_tariff=True,
                      tariff_multiplier=mult)
        assert r.tariff_is_assumed is True, (
            f"result at tariff x{mult} does not carry tariff_is_assumed while "
            f"electricity_cost_usd_per_kwh_measured is null"
        )
        assert "ASSUMED" in r.tariff_basis


def test_measured_tariff_would_clear_the_flag(hw: dict) -> None:
    """The flag tracks the data, not a hardcoded constant.

    Without this, `tariff_is_assumed = True` could be permanently wired on and the test
    above would still pass — verifying a constant rather than a property.
    """
    measured = dict(hw, electricity_cost_usd_per_kwh_measured=0.0847)
    r = breakeven(SONNET_USD_PER_1K, hw=measured, allow_assumed_tariff=False)
    assert r.tariff_is_assumed is False
    assert resolve_tariff(measured).basis.startswith("measured")


def test_tariff_sensitivity_matches_the_published_table(hw: dict) -> None:
    """Pin PREREGISTRATION 3ac's numbers so the doc cannot drift from the code."""
    ref = breakeven(SONNET_USD_PER_1K, hw=hw, allow_assumed_tariff=True,
                    tariff_multiplier=0.0).v_star          # energy-free reference
    # UPDATED 2026-09-09 with the trained-artefact energy figure (0.6269 J/req, was
    # 0.591). The shifts are 1-27 clauses on a V* of ~1.41M; 3ac's conclusion that the
    # tariff is not load-bearing is unaffected, and the test still asserts that.
    expected = {0.5: 1413948, 1.0: 1413971, 2.0: 1414018, 10.0: 1414390}
    for mult, v in expected.items():
        got = breakeven(SONNET_USD_PER_1K, hw=hw, allow_assumed_tariff=True,
                        tariff_multiplier=mult).v_star
        assert round(got) == v, f"tariff x{mult}: V*={got:,.0f}, table says {v:,}"
    worst = breakeven(SONNET_USD_PER_1K, hw=hw, allow_assumed_tariff=True,
                      tariff_multiplier=10.0).v_star
    assert (worst / ref - 1) < 0.0005, (
        f"a 10x tariff now moves V* by {(worst/ref-1)*100:.4f}%, over the 0.05% at which "
        f"3ac calls the assumption non-load-bearing. Re-open that conclusion."
    )


def test_device_cost_is_derived_from_inr_not_stored_as_usd(hw: dict) -> None:
    """Hard rule 5 / the FX-auditability note: no hardcoded USD device cost."""
    assert "device_cost_usd" not in hw, (
        "a USD device cost has been stored. It must stay derived from "
        "device_cost_amount and fx.usd_per_inr_rate so the report can be audited "
        "against the rate and its date."
    )
