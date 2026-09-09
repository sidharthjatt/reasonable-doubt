"""The §3ae band must be inseparable from E2's result — asserted, not trusted."""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
from pathlib import Path

import pytest

from src.eval.e2_result import _CHI2_95, E2Band, E2Result, margin_for

SD = 0.0031954361          # C3-measured, train_holdout_3000, 3 seeds
ROOT = Path(__file__).resolve().parents[1]


def _c3_module():
    spec = importlib.util.spec_from_file_location("c3", ROOT / "scripts" / "c3_substitute.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def band() -> E2Band:
    return E2Band(seed_sd=SD, n_seeds=3)


def result(delta: float = 0.02) -> E2Result:
    return E2Result(arm="sqrt_inv_freq", arm_macro_f1=0.7570 + delta,
                    e1_macro_f1=0.7570, band=band())


# --- the substituted rule and the reported result must use ONE band ------------------
def test_chi2_multipliers_match_c3_substitute():
    c3 = _c3_module()
    assert _CHI2_95[3] == (c3.SIGMA_OVER_S_LO, c3.SIGMA_OVER_S_HI)


def test_margin_formula_matches_c3_substitute():
    c3 = _c3_module()
    assert margin_for(SD) == pytest.approx(c3.margin(SD), abs=1e-15)


def test_margin_is_the_registered_formula():
    assert margin_for(SD) == pytest.approx(math.sqrt(2) * 1.96 * SD, abs=1e-15)


# --- the enforcement property --------------------------------------------------------
def _public_string_renderings(r: E2Result) -> dict[str, str]:
    """Every public method/property whose output a human could paste into a report."""
    out = {}
    for name in dir(r):
        if name.startswith("_") and name != "__str__":
            continue
        attr = getattr(type(r), name, None)
        try:
            val = getattr(r, name)
            val = val() if callable(val) and not inspect.isclass(val) else val
        except Exception:
            continue
        if isinstance(val, str):
            out[name] = val
        elif isinstance(val, dict):
            out[name] = json.dumps(val)
    return out


def test_every_rendering_that_exposes_the_margin_carries_the_band():
    """The precise property: a rendering may name the arm or the split freely, but any
    rendering that exposes the MARGIN or the VERDICT must carry the band with it —
    those are the quantities §3ae says overstate their precision when quoted alone."""
    r = result()
    b = band()
    rendered = _public_string_renderings(r)
    assert rendered, "no public renderings found — the guard would be vacuous"
    exposes_margin = {
        n: t for n, t in rendered.items()
        if f"{b.margin:.4f}" in t or "ACCEPTED" in t or '"margin"' in t
    }
    assert exposes_margin, (
        "no rendering exposes the margin — the guard is vacuous, which means either "
        "statement() or as_dict() has stopped reporting E2's result")
    for name, text in exposes_margin.items():
        assert "95%" in text or "margin_band_95" in text, (
            f"{name} exposes E2's margin/verdict without the §3ae band: {text[:200]}")


def test_the_guard_catches_a_rendering_that_forgets_the_band():
    """So a passing suite means 'the band is there', not 'the guard is broken'."""
    class Forgetful(E2Result):
        def summary(self) -> str:                      # a plausible future addition
            v = "ACCEPTED" if self.accepted else "NOT ACCEPTED"
            return f"E2 [{self.arm}] margin {self.band.margin:.4f} — {v}"

    bad = Forgetful(arm="x", arm_macro_f1=0.78, e1_macro_f1=0.7570, band=band())
    offending = [
        n for n, t in _public_string_renderings(bad).items()
        if (f"{bad.band.margin:.4f}" in t or "ACCEPTED" in t or '"margin"' in t)
        and not ("95%" in t or "margin_band_95" in t)
    ]
    assert "summary" in offending


def test_statement_contains_every_band_number():
    b = band()
    s = result().statement()
    for v in (b.margin, b.sd_lo, b.sd_hi, b.margin_lo, b.margin_hi):
        assert f"{v:.4f}" in s, f"{v:.4f} missing from statement()"


def test_as_dict_always_has_the_statement_and_band():
    d = result().as_dict()
    assert d["mandatory_band_statement"]
    assert d["margin_band_95"] == [pytest.approx(band().margin_lo),
                                   pytest.approx(band().margin_hi)]
    assert "95%" in d["mandatory_band_statement"]


def test_no_bare_verdict_string_accessor():
    """`accepted` is a bool by design: a bool cannot masquerade as a precise number."""
    assert isinstance(result().accepted, bool)
    for name in dir(result()):
        if name.startswith("_"):
            continue
        val = getattr(result(), name)
        val = val() if callable(val) else val
        if isinstance(val, str):
            assert val.strip() not in {"ACCEPTED", "NOT ACCEPTED"}, (
                f"{name} returns a bare verdict with no band attached")


# --- the measured C3 values land where §3ae said they would --------------------------
def test_measured_c3_is_regime_A_with_a_band_reaching_regime_C():
    b = band()
    assert b.margin == pytest.approx(0.0088573, abs=1e-6)
    assert b.regime(b.margin) == "A"
    assert b.regime(b.margin_hi) == "C", (
        "§3ae's warning is that a regime-A point estimate can have an interval reaching "
        "regime C; at the measured seed_sd it does, and the sentence must say so")
    assert b.margin_hi == pytest.approx(0.0557, abs=5e-4)
    assert b.sd_hi < 0.083, "C3's own falsification line is not crossed"


def test_accept_decision_uses_the_point_estimate_not_the_band():
    """§3ae: the point estimate is what the formula registers and is what is used."""
    b = band()
    assert result(delta=b.margin + 1e-6).accepted
    assert not result(delta=b.margin - 1e-6).accepted
    # a delta inside the band's upper end but above the point estimate still accepts
    assert result(delta=0.03).accepted


# --- refusals ------------------------------------------------------------------------
def test_unregistered_seed_count_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="amendment"):
        E2Band(seed_sd=SD, n_seeds=5)


def test_nonpositive_seed_sd_raises():
    with pytest.raises(ValueError):
        E2Band(seed_sd=0.0, n_seeds=3)
