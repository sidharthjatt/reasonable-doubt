"""Cascade structure: routing, cost provenance, and that a stub can never pass as real."""

from __future__ import annotations

import pytest

from src.serve.app import Cascade, RouterPolicy, default_cascade
from src.serve.tiers import StubTier, TierResult


class FixedTier:
    def __init__(self, name, label, conf, **usage):
        self.name, self.is_stub = name, False
        self._r = TierResult(label=label, confidence=conf, tier=name, is_stub=False, **usage)

    def classify(self, text): return self._r


def test_confident_tier0_stops_the_cascade():
    c = Cascade(tier0=FixedTier("tier0", "Notices", 0.999),
                tier1=StubTier("tier1"), tier2=StubTier("tier2"))
    out = c.classify("x")
    assert out["tier_used"] == "tier0"
    assert out["tiers_invoked"] == ["tier0"]


def test_unconfident_tier0_escalates_all_the_way():
    out = default_cascade().classify("x")
    assert out["tiers_invoked"] == ["tier0", "tier1", "tier2"]
    assert out["tier_used"] == "tier2"


def test_an_unparseable_label_always_escalates():
    c = Cascade(tier0=FixedTier("tier0", None, 1.0),
                tier1=FixedTier("tier1", "Notices", 1.0), tier2=StubTier("tier2"))
    assert c.classify("x")["tier_used"] == "tier1"


def test_stub_tiers_are_named_in_every_response():
    """A stubbed cascade must be impossible to mistake for a working one."""
    out = default_cascade().classify("x")
    assert out["stub_tiers"] == ["tier0", "tier1", "tier2"]
    assert out["router_calibrated"] is False


def test_cost_is_reported_per_tier_with_its_basis():
    out = default_cascade().classify("x")
    ec = out["estimated_cost"]
    assert ec["usd"] == pytest.approx(sum(p["usd"] for p in ec["per_tier"]))
    assert ec["usd_per_1k"] == pytest.approx(ec["usd"] * 1000)
    assert ec["is_estimate"] is True
    assert ec["tariff_is_assumed"] is True, "the assumed tariff must surface in the API"
    for p in ec["per_tier"]:
        assert p["basis"], "every cost component states where it came from"


def test_local_cost_is_not_zero():
    """A local tier costed at zero makes every cascade look free at volume, which is the
    assumption E6 exists to test rather than to assume."""
    from src.serve.pricing import local_usd_per_request
    assert local_usd_per_request().usd > 0


def test_no_price_is_hardcoded_in_the_service():
    """Hard rule 5. Any literal price here would silently diverge from costs.yaml."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for f in ("src/serve/app.py", "src/serve/pricing.py", "src/serve/tiers.py"):
        src = (root / f).read_text()
        body = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#") and "usd_per_kwh" not in l)
        assert not re.search(r"\b\d+\.\d{2,}\s*(?:/|\*)?\s*1[_,]?000\b", body), f
        assert "0.4483" not in body and "3.00" not in body, f"hardcoded price in {f}"


def test_router_thresholds_are_named_uncalibrated():
    """E5 has not run, so no threshold here is calibrated. The field names say so, and
    `calibrated` is False, so a guess cannot be reported as a calibration."""
    p = RouterPolicy()
    assert hasattr(p, "tier0_uncalibrated") and hasattr(p, "tier1_uncalibrated")
    assert p.calibrated is False


def test_app_builds_and_health_reports_stubs():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src.serve.app import create_app
    client = TestClient(create_app())
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["stub_tiers"] == ["tier0", "tier1", "tier2"]
    r = client.post("/classify", json={"text": "This Agreement is governed by NY law."})
    body = r.json()
    assert body["tier_used"] == "tier2"
    assert "estimated_cost" in body and body["stub_tiers"]
