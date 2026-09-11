"""The DEPLOYED service: Tier 0 INT8 -> Claude Sonnet 5 (E4b-A, no Tier 1).

These tests use fake tiers wherever a real one would only add a 250 MB ONNX load. What
they do NOT fake is the machinery under test: the router threshold comes from the real
committed `configs/router_threshold.json`, and every cost assertion is computed from the
real `configs/costs.yaml` rather than from a literal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.api.usage import parse_usage
from src.serve.app import Cascade, MarginRouter, run_startup_canary
from src.serve.canary import (
    CanaryFailure,
    CanaryResult,
    CanarySet,
    cpu_isa_flags,
    run_canary,
)
from src.serve.config import RouterThreshold, ServiceConfig, ThresholdArtefactMismatch
from src.serve.pricing import api_usd_for_result
from src.serve.tiers import TierResult

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- fakes


class FakeTier0:
    name, is_stub = "tier0", False

    def __init__(self, label="Notices", margin=0.99, by_text=None):
        self._label, self._margin, self._by_text = label, margin, by_text or {}

    def classify(self, text):
        label, margin = self._by_text.get(text, (self._label, self._margin))
        return TierResult(label=label, confidence=margin, tier="tier0", is_stub=False)

    def describe(self):
        return {"model_dir": "fake"}


class FakeTier2:
    name, is_stub = "tier2", False

    def __init__(self, *, available=True, label="Governing Laws",
                 input_tokens=1000, output_tokens=20,
                 cache_read=500, cache_write=0, cache_hit=False):
        self.available = available
        self._label, self._hit = label, cache_hit
        self._usage = parse_usage(
            {"input_tokens": input_tokens, "output_tokens": output_tokens,
             "cache_creation_input_tokens": cache_write,
             "cache_read_input_tokens": cache_read},
            model="claude-sonnet-5", provider="anthropic", cache_fields_reported=True)

    def classify(self, text):
        return TierResult(label=self._label, confidence=0.9, tier="tier2",
                          is_stub=False, usage=self._usage,
                          input_tokens=self._usage.input_tokens,
                          output_tokens=self._usage.output_tokens,
                          cache_read_input_tokens=self._usage.cache_read_input_tokens,
                          cache_creation_input_tokens=(
                              self._usage.cache_creation_input_tokens),
                          api_cache_hit=self._hit)

    def describe(self):
        return {"model": "claude-sonnet-5", "available": self.available}


@pytest.fixture
def config():
    return ServiceConfig.load()


def make_cascade(config, *, tier0=None, tier2=None):
    return Cascade(
        tier0=tier0 or FakeTier0(), tier2=tier2 or FakeTier2(), config=config,
        router=MarginRouter(threshold=config.threshold.threshold,
                            signal=config.threshold.signal))


# ------------------------------------------------------------- 5. canary refusal


def test_canary_refuses_to_start_when_accuracy_is_below_floor(ledgar):
    """The Kaggle failure mode: INT8 at chance, no error raised. This must REFUSE.

    A wrong-label tier scores 0 on every row, which is what a broken INT8 kernel
    produces — a confident, plausible, wrong answer on every request.
    """
    canary = CanarySet.from_dict(
        json.loads((ROOT / "configs" / "canary_200.json").read_text()))
    broken = FakeTier0(label="Adjustments", margin=0.97)

    with pytest.raises(CanaryFailure) as exc:
        run_canary(broken, ledgar, canary, floor=0.80)

    msg = str(exc.value)
    assert "REFUSING TO START" in msg
    assert "AVX-512 VNNI" in msg, "the message must name the mechanism, not just fail"
    assert "0.0000" in msg or "accuracy 0.0" in msg


def test_canary_passes_when_the_tier_is_correct(ledgar):
    """A tier that answers correctly clears the floor and reports its hardware."""
    from src.data.loading import get_split, label_names

    canary = CanarySet.from_dict(
        json.loads((ROOT / "configs" / "canary_200.json").read_text()))
    rows = get_split(ledgar, canary.split).select(canary.indices)
    names = label_names(ledgar)
    gold = {t: (names[int(l)], 0.99) for t, l in zip(rows["text"], rows["label"])}

    result = run_canary(FakeTier0(by_text=gold), ledgar, canary, floor=0.80)
    assert isinstance(result, CanaryResult)
    assert result.passed and result.accuracy == 1.0 and result.n == 200
    assert "cpu_isa_flags" in result.hardware


def test_canary_rows_touch_no_evaluation_role(ledgar):
    """Hard rule 1 adjacency: the canary must not consume dev, test or the selection
    set. It draws from TRAIN and excludes the train manifests that already have roles."""
    from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest

    canary = CanarySet.from_dict(
        json.loads((ROOT / "configs" / "canary_200.json").read_text()))
    assert canary.split == "train"

    for name in ("train_holdout_3000", "exemplars_8"):
        reserved = set(load_manifest(name, DEFAULT_MANIFEST_DIR).indices)
        assert not (set(canary.indices) & reserved), f"canary overlaps {name}"

    for name in ("dev_2000", "test_3000"):
        m = load_manifest(name, DEFAULT_MANIFEST_DIR)
        assert m.split != canary.split


def test_disabled_canary_is_reported_not_silent(config, monkeypatch):
    """A skipped check must show up in /health as skipped, never as absent."""
    disabled = ServiceConfig(**{**config.__dict__, "canary_enabled": False})
    out = run_startup_canary(make_cascade(disabled))
    assert out["ran"] is False and "disabled" in out["reason"]
    assert "hardware" in out


def test_cpu_flags_report_unreadable_as_none_not_false():
    """None means 'we could not tell'; False means 'this CPU lacks it'. Collapsing the
    two would turn an unknown into a measurement (hard rule 11)."""
    flags = cpu_isa_flags()
    assert set(flags) == {"avx512_vnni", "avx512f", "avx2"}
    for v in flags.values():
        assert v is None or isinstance(v, bool)


# ------------------------------------------------- 3. the escalation_skipped path


def test_no_api_key_returns_tier0_and_marks_escalation_skipped(config):
    """Never pretend to escalate. The router selected the row; the key is absent; the
    response says both, and reports Tier 0 as the tier that actually answered."""
    below = config.threshold.threshold - 0.05
    cascade = make_cascade(config, tier0=FakeTier0(margin=below),
                           tier2=FakeTier2(available=False))
    out = cascade.classify("ambiguous clause")

    assert out["escalation_selected"] is True
    assert out["escalation_skipped"] is True
    assert out["tier_used"] == "tier0"
    assert out["tiers_invoked"] == ["tier0"]
    assert "ANTHROPIC_API_KEY" in out["escalation_skipped_reason"]
    assert [p["tier"] for p in out["cost_detail"]["per_tier"]] == ["tier0"], \
        "a skipped escalation must not be billed"


def test_confident_row_does_not_escalate_even_with_a_key(config):
    above = config.threshold.threshold + 0.05
    out = make_cascade(config, tier0=FakeTier0(margin=above)).classify("clear clause")
    assert out["escalation_selected"] is False
    assert out["escalation_skipped"] is False
    assert out["tier_used"] == "tier0"


def test_escalation_happens_when_the_key_is_present(config):
    below = config.threshold.threshold - 0.05
    out = make_cascade(config, tier0=FakeTier0(margin=below)).classify("ambiguous")
    assert out["tier_used"] == "tier2"
    assert out["tiers_invoked"] == ["tier0", "tier2"]
    assert out["escalation_skipped"] is False
    assert out["margin"] == pytest.approx(below), \
        "margin is Tier 0's, even when Tier 2 answered"


def test_tier2_unavailable_mid_request_does_not_fabricate(config):
    """`available` said yes and `classify` then raised — a race on the env var. The
    Tier 0 answer is returned and marked; no exception escapes as a fake escalation."""
    from src.serve.tier2 import Tier2Unavailable

    class Racy(FakeTier2):
        def classify(self, text):
            raise Tier2Unavailable("ANTHROPIC_API_KEY vanished mid-request")

    below = config.threshold.threshold - 0.05
    out = make_cascade(config, tier0=FakeTier0(margin=below),
                       tier2=Racy(available=True)).classify("x")
    assert out["tier_used"] == "tier0" and out["escalation_skipped"] is True


def test_a_missing_margin_is_refused_rather_than_guessed(config):
    router = MarginRouter(threshold=0.5)
    with pytest.raises(ValueError, match="routing cannot be decided"):
        router.escalate(TierResult(label="Notices", confidence=None,
                                   tier="tier0", is_stub=False))


# ------------------------------------------------------ 4. cost from costs.yaml only


def _rates(model: str) -> dict:
    card = yaml.safe_load((ROOT / "configs" / "costs.yaml").read_text())
    return card["providers"]["anthropic"]["models"][model]


def test_tier2_cost_is_derived_from_costs_yaml(config):
    """Recompute the escalated cost straight from the YAML and require a match."""
    r = _rates("claude-sonnet-5")
    card = yaml.safe_load((ROOT / "configs" / "costs.yaml").read_text())
    cache_read_mult = card["modifiers"]["cache_read_multiplier"]

    below = config.threshold.threshold - 0.05
    tier2 = FakeTier2(input_tokens=1000, output_tokens=20, cache_read=500, cache_write=0)
    out = make_cascade(config, tier0=FakeTier0(margin=below), tier2=tier2).classify("x")

    expected_tier2 = (1000 * r["input"]
                      + 500 * r["input"] * cache_read_mult
                      + 20 * r["output"]) / 1_000_000
    per_tier = {p["tier"]: p["usd"] for p in out["cost_detail"]["per_tier"]}
    assert per_tier["tier2"] == pytest.approx(expected_tier2)
    assert out["estimated_cost_usd"] == pytest.approx(sum(per_tier.values()))


def test_the_three_input_usage_fields_are_priced_separately(config):
    """Hard rule 10. Summing them into one input figure changes the answer, so a test
    that only checked the total could not tell the two implementations apart."""
    r = _rates("claude-sonnet-5")
    card = yaml.safe_load((ROOT / "configs" / "costs.yaml").read_text())
    read_mult = card["modifiers"]["cache_read_multiplier"]
    write_mult = card["modifiers"]["cache_write_multipliers"]["1h"]

    tier2 = FakeTier2(input_tokens=1000, output_tokens=0,
                      cache_read=1000, cache_write=1000)
    got = api_usd_for_result("claude-sonnet-5", tier2.classify("x"), batch=False)

    separate = (1000 * r["input"]
                + 1000 * r["input"] * read_mult
                + 1000 * r["input"] * write_mult) / 1_000_000
    summed_as_one = (3000 * r["input"]) / 1_000_000
    assert got.usd == pytest.approx(separate)
    assert got.usd != pytest.approx(summed_as_one)


def test_costing_a_result_with_no_usage_block_raises(config):
    """Hard rule 11: reconstructing usage from flattened ints would turn an unreported
    cache field into a zero. Refuse instead."""
    bare = TierResult(label="x", confidence=0.5, tier="tier2", is_stub=False,
                      input_tokens=100, output_tokens=10)
    with pytest.raises(ValueError, match="no parsed usage block"):
        api_usd_for_result("claude-sonnet-5", bare)


def test_api_usd_for_result_accepts_compute_cost_kwargs():
    """REGRESSION. This function passed `cache_creation_input_tokens=` /
    `cache_read_input_tokens=` to `compute_cost`, whose parameters are
    `cache_write_tokens` / `cache_read_tokens`. Every real escalation raised TypeError;
    it never fired only because the sole caller was a zero-token stub."""
    est = api_usd_for_result("claude-sonnet-5", FakeTier2().classify("x"))
    assert est.usd > 0 and est.is_estimate is False


def test_interactive_serving_does_not_claim_the_batch_discount(config):
    """The 50% Batch discount applies to the Batch API, not a live request."""
    assert config.tier2_batch is False
    full = api_usd_for_result("claude-sonnet-5", FakeTier2().classify("x"), batch=False)
    batched = api_usd_for_result("claude-sonnet-5", FakeTier2().classify("x"), batch=True)
    assert batched.usd == pytest.approx(full.usd * 0.5)


def test_local_tier_cost_is_not_zero_and_declares_its_assumption(config):
    out = make_cascade(config).classify("x")
    tier0 = next(p for p in out["cost_detail"]["per_tier"] if p["tier"] == "tier0")
    assert tier0["usd"] > 0
    assert tier0["is_estimate"] is True
    assert tier0["tariff_is_assumed"] is True
    assert out["cost_detail"]["source"] == "configs/costs.yaml"


def test_no_price_is_hardcoded_in_the_service():
    """Hard rule 5. Any literal price would silently diverge from costs.yaml."""
    import re

    for f in ("src/serve/app.py", "src/serve/pricing.py", "src/serve/tiers.py",
              "src/serve/tier0.py", "src/serve/tier2.py", "src/serve/config.py",
              "src/serve/canary.py"):
        src = (ROOT / f).read_text()
        body = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#") and "usd_per_kwh" not in l)
        assert not re.search(r"\b\d+\.\d{2,}\s*(?:/|\*)?\s*1[_,]?000\b", body), f
        for literal in ("0.4483", "3.00", "2.0 /", "10.0 /"):
            assert literal not in body, f"hardcoded price in {f}"


# ------------------------------------ 1 & 2. configured artefact, file-loaded threshold


def test_tier0_model_dir_comes_from_config_not_code(config):
    """E1b may replace int8_ce_1; a baked-in path would keep serving old weights."""
    assert config.tier0_model_dir.name == "int8_ce_1"
    src = (ROOT / "src" / "serve" / "app.py").read_text()
    assert "int8_ce" not in src, "the artefact name must not appear in code"


def test_threshold_is_loaded_from_a_dev_calibrated_file(config):
    d = json.loads((ROOT / "configs" / "router_threshold.json").read_text())
    assert config.threshold.threshold == pytest.approx(d["threshold"])
    assert d["calibrated_on"] == "dev_2000"          # hard rule 1
    assert d["signal"] == "margin"
    assert config.threshold.artefact.endswith("int8_ce_1")


def test_the_router_never_computes_a_threshold_at_request_time(config):
    out = make_cascade(config).classify("x")
    assert out["router"]["computed_at_request_time"] is False
    assert out["router"]["threshold"] == pytest.approx(config.threshold.threshold)
    assert out["router"]["calibrated_on"] == "dev_2000"

    # AST, not a substring scan: the word "calibrated" appears in this module's prose
    # and a substring check would pass or fail on documentation rather than on code.
    # What must be true is that the calibration machinery is not IMPORTED or CALLED here.
    import ast

    tree = ast.parse((ROOT / "src" / "serve" / "app.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    assert not any("router.calibrate" in m for m in imported), \
        "the calibration module must not be importable from the request path"

    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    called |= {node.func.attr for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    for forbidden in ("sweep_thresholds", "compare_signals", "quantile"):
        assert forbidden not in called, f"{forbidden}() must not run per-request"


def test_a_threshold_calibrated_on_test_is_refused(tmp_path):
    """Hard rule 1, enforced at load time rather than trusted."""
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps({
        "signal": "margin", "threshold": 0.5, "calibrated_on": "test_3000",
        "artefact": "models/int8_ce_1", "seed": 1, "target_escalation_rate": 0.1,
        "achieved_escalation_rate_on_dev": 0.1,
        "retained_tier0_accuracy_on_dev": 0.9}))
    with pytest.raises(ValueError, match="Hard rule 1"):
        RouterThreshold.load(bad)


def test_serving_weights_the_threshold_was_not_calibrated_on_is_refused(tmp_path):
    """The margin distribution is a property of the weights. Swapping in E1b's 10-epoch
    artefact without recalibrating must fail loudly, not shift the escalation rate."""
    serve = yaml.safe_load((ROOT / "configs" / "serve.yaml").read_text())
    serve["tier0"]["model_dir"] = "models/int8_ce10ep_1"
    cfg = tmp_path / "serve.yaml"
    cfg.write_text(yaml.safe_dump(serve))

    with pytest.raises(ThresholdArtefactMismatch, match="int8_ce10ep_1"):
        ServiceConfig.load(cfg, root=ROOT)


def test_there_is_no_tier1_in_the_deployed_cascade(config):
    """E4b-A: Tier 1 is absent by measurement, not by oversight."""
    cascade = make_cascade(config)
    assert not hasattr(cascade, "tier1")
    out = cascade.classify("x")
    assert "tier1" not in out["tiers_invoked"]


# ------------------------------------------------------- the app: /health and startup


@pytest.fixture
def fastapi_client(config):
    """TestClient, but only when the real Tier 0 artefact is on disk.

    `create_app()` goes through `build_cascade()`, which constructs a REAL Tier0Encoder
    and loads a 244 MB ONNX model. models/ is gitignored in full, so on a clean checkout
    these tests would ERROR with FileNotFoundError rather than skip — which reads as a
    broken build instead of an absent artefact.
    """
    pytest.importorskip("fastapi")
    from tests.conftest import require_artifact

    require_artifact(config.tier0_model_dir,
                     "the served INT8 encoder; models/ is gitignored (`/models/`) so it "
                     "is never present in CI. Fetch or mount the artefact to run these")
    from fastapi.testclient import TestClient

    return TestClient


def test_health_exposes_ort_version_and_isa_flags(config, fastapi_client):
    """Requirement 5's disclosure half. Kaggle's non-VNNI x86 scored these weights at
    chance with no error, and HF Spaces is x86 — so the runtime facts that explain a
    canary result must be readable without shelling into the box."""
    from src.serve.app import create_app

    app = create_app(config, tier2=FakeTier2(available=False),
                     run_canary_on_start=False)
    body = fastapi_client(app).get("/health").json()

    runtime = body["runtime"]
    assert "onnxruntime_version" in runtime
    flags = runtime["cpu_isa_flags"]
    assert set(flags) == {"avx512_vnni", "avx512f", "avx2"}
    for v in flags.values():
        assert v is None or isinstance(v, bool)   # null == unreadable, never False

    assert body["architecture"].startswith("tier0_int8 -> claude")
    assert body["escalation_enabled"] is False
    assert body["router"]["calibrated_on"] == "dev_2000"


def test_health_says_when_the_canary_was_skipped(config, fastapi_client):
    from src.serve.app import create_app

    app = create_app(config, tier2=FakeTier2(available=False),
                     run_canary_on_start=False)
    canary = fastapi_client(app).get("/health").json()["canary"]
    assert canary["ran"] is False, "a skipped canary must never look like a passed one"


def test_a_failing_canary_prevents_the_app_from_being_created(config, fastapi_client,
                                                              monkeypatch, ledgar):
    """The refusal is at STARTUP, not per-request: an app that started and then failed
    every request would still be serving chance-level labels."""
    from src.serve import app as app_mod

    monkeypatch.setattr(app_mod, "build_cascade",
                        lambda cfg, tier2=None: Cascade(
                            tier0=FakeTier0(label="Adjustments", margin=0.97),
                            tier2=FakeTier2(available=False), config=config,
                            router=MarginRouter(threshold=config.threshold.threshold)))

    with pytest.raises(CanaryFailure, match="REFUSING TO START"):
        app_mod.create_app(config, run_canary_on_start=True)


def test_classify_endpoint_returns_the_required_fields(config, fastapi_client):
    """label, margin, tier_used, estimated_cost_usd — requirement 4's response shape."""
    from src.serve.app import create_app

    app = create_app(config, tier2=FakeTier2(available=False),
                     run_canary_on_start=False)
    body = fastapi_client(app).post("/classify", json={"text": "a clause"}).json()

    for field in ("label", "margin", "tier_used", "estimated_cost_usd"):
        assert field in body, field
    assert body["tier_used"] == "tier0"
    assert isinstance(body["estimated_cost_usd"], float)


# ------------------------------------------------- Tier2Claude itself (stubbed client)


class _FakeUsage:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text, usage):
        self.content, self.usage = [_FakeBlock(text)], usage


class FakeAnthropicClient:
    """Stands in for anthropic.Anthropic. Counts calls so re-sends are detectable."""

    def __init__(self, text='{"label": "Notices", "confidence": 0.71}', **usage):
        self.calls = []
        self._text = text
        self._usage = usage or dict(input_tokens=1200, output_tokens=18,
                                    cache_creation_input_tokens=0,
                                    cache_read_input_tokens=1084)
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return _FakeResponse(self._text, _FakeUsage(**self._usage))


def make_tier2(tmp_path, client=None, labels=None):
    """Build a Tier2Claude wired to a THROWAWAY ledger.

    Tier2Claude defaults to the real results/spend_ledger.jsonl, which is correct in
    production and catastrophic in a test: it is the version-controlled record of real
    money (hard rule 12), and a suite that appends fake entries corrupts it. An earlier
    version of this helper omitted the ledger and wrote 24 fabricated entries into it.
    The session guard in conftest.py now makes that impossible to miss.
    """
    from src.api.cache import ResponseCache
    from src.api.ledger import SpendLedger
    from src.data.labels import load_labels
    from src.serve.tier2 import Tier2Claude

    # temperature deliberately NOT passed: the default is None (omit), matching
    # Stage 1. Pinning 0.0 here would hide the very regression this suite checks for.
    ledger_path = tmp_path / "spend_ledger.jsonl"
    ledger_path.write_text("")
    t = Tier2Claude(model="claude-sonnet-5", labels=labels or load_labels(),
                    max_output_tokens=40, batch=False,
                    cache=ResponseCache(tmp_path),
                    ledger=SpendLedger(ledger_path), spend_cap_usd=None)
    t._client = client or FakeAnthropicClient()
    return t


def test_tier2_refuses_without_a_key(tmp_path, monkeypatch):
    from src.serve.tier2 import Tier2Unavailable

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    t = make_tier2(tmp_path)
    assert t.available is False
    with pytest.raises(Tier2Unavailable, match="ANTHROPIC_API_KEY"):
        t.classify("a clause")


def test_tier2_treats_a_blank_key_as_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert make_tier2(tmp_path).available is False


def test_tier2_caches_the_response_before_use(tmp_path, monkeypatch):
    """Hard rule 3. The second call must not reach the API at all."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    client = FakeAnthropicClient()
    t = make_tier2(tmp_path, client)

    first = t.classify("This notice shall be delivered by hand.")
    assert len(client.calls) == 1
    assert first.label == "Notices" and first.api_cache_hit is False

    second = t.classify("This notice shall be delivered by hand.")
    assert len(client.calls) == 1, "a cached (model, prompt) pair was re-sent"
    assert second.api_cache_hit is True
    assert second.label == first.label
    assert list(tmp_path.rglob("*.json")), "nothing was written to disk"


def test_tier2_cache_key_separates_models(tmp_path, monkeypatch):
    """Hard rule 9. A prompt-only key would serve Sonnet's answer for a Haiku request,
    along with Sonnet's token counts — cheap, plausible and wrong."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    t = make_tier2(tmp_path)
    k_sonnet = t.provider.cache_key("claude-sonnet-5", system=t._system.text,
                                    messages=t._messages("x"), params=t._params())
    k_haiku = t.provider.cache_key("claude-haiku-4-5-20251001", system=t._system.text,
                                   messages=t._messages("x"), params=t._params())
    assert k_sonnet.prompt_sha256 == k_haiku.prompt_sha256
    assert k_sonnet.digest != k_haiku.digest


def test_tier2_result_carries_all_three_input_usage_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    t = make_tier2(tmp_path, FakeAnthropicClient(
        input_tokens=100, output_tokens=10,
        cache_creation_input_tokens=2168, cache_read_input_tokens=0))
    r = t.classify("clause")

    assert r.usage is not None
    assert r.usage.input_tokens == 100
    assert r.usage.cache_creation_input_tokens == 2168
    assert r.usage.cache_read_input_tokens == 0
    # and it is costable end to end, keeping the three apart
    assert api_usd_for_result("claude-sonnet-5", r).usd > 0


def test_tier2_uses_a_1h_cache_ttl_not_the_5m_default(tmp_path, monkeypatch):
    """CLAUDE.md: 1h TTL, because batch requests process out of order."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    client = FakeAnthropicClient()
    make_tier2(tmp_path, client).classify("clause")
    system = client.calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_tier2_unparseable_output_yields_no_label_rather_than_a_guess(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    t = make_tier2(tmp_path, FakeAnthropicClient(text="I cannot classify this."))
    r = t.classify("clause")
    assert r.label is None, "an unparseable response must not become a plausible label"
    assert r.usage is not None, "it still cost money and must still be costable"


# ------------------- item 3: the deployed request must match Stage 1's, field by field


def test_deployed_prompt_is_byte_identical_to_stage_1(ledgar):
    """Stage 1 renders from label_names(ds); the service renders from configs/labels.json.
    Two sources for one cacheable prefix — if they ever diverge, the deployed tier is
    measured against a baseline it no longer shares a prompt with."""
    from src.data.labels import load_labels
    from src.data.loading import label_names
    from src.data.prompts import render_zeroshot

    assert render_zeroshot(label_names(ledgar)).sha256 == \
        render_zeroshot(load_labels()).sha256


def test_temperature_is_omitted_not_sent_as_zero(tmp_path, monkeypatch):
    """Sonnet 5 REJECTS `temperature`. Stage 1 passes temperature=None for exactly this
    reason; sending 0.0 would have been a 400 on every escalation."""
    from src.serve.tier2 import TEMPERATURE

    assert TEMPERATURE is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    client = FakeAnthropicClient()
    make_tier2(tmp_path, client).classify("clause")
    assert "temperature" not in client.calls[0], \
        "temperature must be OMITTED, not sent (null is still a rejected field)"


def test_thinking_is_explicitly_disabled_like_stage_1(tmp_path, monkeypatch):
    """Unset lets the model default apply, changing output shape and the token bill."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    client = FakeAnthropicClient()
    make_tier2(tmp_path, client).classify("clause")
    assert client.calls[0]["thinking"] == {"type": "disabled"}


def test_max_tokens_and_model_match_stage_1(tmp_path, monkeypatch, config):
    """40 is Stage 1's amended Sonnet output budget (costs.yaml, §3i)."""
    card = yaml.safe_load((ROOT / "configs" / "costs.yaml").read_text())
    stage1_out = card["budget"]["stages"]["stage_1"]["max_output_tokens"]
    assert config.tier2_max_output_tokens == stage1_out["claude-sonnet-5"]
    assert config.tier2_model == "claude-sonnet-5"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    client = FakeAnthropicClient()
    make_tier2(tmp_path, client).classify("clause")
    assert client.calls[0]["max_tokens"] == 40
    assert client.calls[0]["model"] == "claude-sonnet-5"


def test_the_cache_key_describes_the_request_actually_sent(tmp_path, monkeypatch):
    """The key's params and the API kwargs come from one `_params()`. If they could
    drift, a cached entry would answer for a request that was never made."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    client = FakeAnthropicClient()
    t = make_tier2(tmp_path, client)
    t.classify("clause")
    sent = client.calls[0]
    for k, v in t._params().items():
        assert sent[k] == v, f"{k} differs between the cache key and the request"


# ----------------------------------------- item 4: spend ledger and the escalation cap


@pytest.fixture
def ledger(tmp_path):
    from src.api.ledger import SpendLedger

    path = tmp_path / "spend_ledger.jsonl"
    path.write_text("")
    return SpendLedger(path)


def test_a_real_escalation_appends_to_the_spend_ledger(tmp_path, monkeypatch, ledger):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    t = make_tier2(tmp_path)
    t.ledger, t.spend_cap_usd = ledger, 10.0

    assert ledger.cumulative_usd() == 0.0
    t.classify("a clause")

    entries = [json.loads(l) for l in ledger.path.read_text().splitlines() if l.strip()]
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "actual" and e["cost_usd"] > 0
    assert e["run_id"] == "serve_tier0_sonnet5"
    # hard rule 10: the three input fields recorded separately, never summed
    assert e["input_tokens"] == 1200
    assert e["cache_read_input_tokens"] == 1084
    assert e["cache_creation_input_tokens"] == 0
    assert ledger.cumulative_usd() == pytest.approx(e["cost_usd"])


def test_a_cache_hit_spends_nothing_and_writes_no_ledger_entry(tmp_path, monkeypatch,
                                                               ledger):
    """Billing a free answer a second time would inflate the record of real money."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    t = make_tier2(tmp_path)
    t.ledger, t.spend_cap_usd = ledger, 10.0

    t.classify("same clause")
    after_first = ledger.cumulative_usd()
    second = t.classify("same clause")

    assert second.api_cache_hit is True
    assert ledger.cumulative_usd() == pytest.approx(after_first)
    assert len([l for l in ledger.path.read_text().splitlines() if l.strip()]) == 1


def test_escalation_is_refused_once_the_cap_is_reached(tmp_path, monkeypatch, ledger):
    from src.serve.tier2 import SpendCapReached

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    client = FakeAnthropicClient()
    t = make_tier2(tmp_path, client)
    t.ledger = ledger
    t.spend_cap_usd = 0.001          # below one request's cost

    t.classify("first clause")       # cap not yet reached: this one goes through
    assert len(client.calls) == 1
    assert ledger.cumulative_usd() > t.spend_cap_usd

    with pytest.raises(SpendCapReached, match="SPEND CAP REACHED"):
        t.classify("second clause")
    assert len(client.calls) == 1, "an API call was made after the cap was reached"


def test_a_cache_hit_still_serves_after_the_cap_is_reached(tmp_path, monkeypatch,
                                                           ledger):
    """The cap governs SPENDING. A free answer is not spending."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    t = make_tier2(tmp_path)
    t.ledger, t.spend_cap_usd = ledger, 10.0
    t.classify("cached clause")

    t.spend_cap_usd = 0.0            # cap now reached
    assert t.cap_reached() is True
    assert t.classify("cached clause").api_cache_hit is True


def test_the_cascade_falls_back_to_escalation_skipped_at_the_cap(config, tmp_path,
                                                                 monkeypatch, ledger):
    """The required end-to-end behaviour: cap reached -> Tier 0 answer, marked."""
    from src.serve.tier2 import Tier2Claude
    from src.api.cache import ResponseCache
    from src.data.labels import load_labels

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    client = FakeAnthropicClient()
    t2 = Tier2Claude(model="claude-sonnet-5", labels=load_labels(),
                     max_output_tokens=40, cache=ResponseCache(tmp_path),
                     ledger=ledger, spend_cap_usd=0.0)
    t2._client = client

    below = config.threshold.threshold - 0.01
    out = make_cascade(config, tier0=FakeTier0(margin=below),
                       tier2=t2).classify("ambiguous clause")

    assert out["escalation_selected"] is True
    assert out["escalation_skipped"] is True
    assert out["tier_used"] == "tier0"
    assert "SPEND CAP REACHED" in out["escalation_skipped_reason"]
    assert len(client.calls) == 0, "no API call may be made at the cap"
    assert [p["tier"] for p in out["cost_detail"]["per_tier"]] == ["tier0"]


def test_a_cap_of_none_is_an_explicit_choice_not_a_default(tmp_path, monkeypatch,
                                                           ledger):
    """Unlimited spending must be something the operator asked for."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    t = make_tier2(tmp_path)
    t.ledger, t.spend_cap_usd = ledger, None
    assert t.cap_reached() is False
    t.classify("clause")
    assert ledger.cumulative_usd() > 0

    serve = yaml.safe_load((ROOT / "configs" / "serve.yaml").read_text())
    assert serve["tier2"]["spend_cap_usd"] is not None, \
        "the committed config must ship WITH a cap"


def test_spend_cap_is_configurable(config):
    assert config.tier2_spend_cap_usd == 1.00
    card = yaml.safe_load((ROOT / "configs" / "costs.yaml").read_text())
    assert config.tier2_spend_cap_usd < card["budget"]["hard_stop_usd"], \
        "a serving cap at or above the project budget could consume the experiments"


def test_the_real_ledger_holds_exactly_the_recorded_live_run():
    """results/spend_ledger.jsonl is the record of REAL money (hard rule 12).

    The live smoke test made the service's first three real Claude calls on 2026-09-11:
    input 145/221/479, the 1,080-token prefix written once then read twice, output
    22/29/26, $0.007212 total. Pinned here so an accidental append or rewrite by this
    suite shows up as a content failure, not just as the sha256 tamper guard in
    conftest.py firing.
    """
    from src.api.ledger import SpendLedger

    led = SpendLedger()
    serving = [e for e in led.entries()
               if e.run_id == "serve_tier0_sonnet5" and e.kind == "actual"]
    assert len(serving) == 3
    assert sum(e.cost_usd for e in serving) == pytest.approx(0.007212, abs=1e-9)

    # hard rule 10: the three input fields are recorded separately, never summed
    assert [e.input_tokens for e in serving] == [145, 221, 479]
    assert [e.cache_creation_input_tokens for e in serving] == [1080, 0, 0]
    assert [e.cache_read_input_tokens for e in serving] == [0, 1080, 1080]
    assert [e.output_tokens for e in serving] == [22, 29, 26]

    # and the serve cap sees only these, not the ~$3.58 of experiment spend
    assert led.cumulative_usd(run_id="serve_tier0_sonnet5") == pytest.approx(0.007212)
    assert led.cumulative_usd() > 3.5


# ----------------------------------------------------- the recalibrated threshold


def test_threshold_is_percentile_calibrated_at_the_registered_rate(config):
    d = json.loads((ROOT / "configs" / "router_threshold.json").read_text())
    assert d["calibration_mode"] == "percentile"
    assert d["target_escalation_rate"] == pytest.approx(0.040556)
    assert d["source_npz"].endswith("dev_logits_int8_local_ce_seed1.npz")
    assert d["precision"] == "int8" and d["isa"] == "arm64_local"
    assert "NOT AN ACCURACY CLAIM" in d["verdict_note"]
    assert d["achieved_escalation_rate_on_dev"] == pytest.approx(0.0406, abs=0.005)


def test_an_absolute_threshold_file_is_refused(tmp_path):
    """§3aq: absolute thresholds span 1.49x across seeds and do not transfer."""
    d = json.loads((ROOT / "configs" / "router_threshold.json").read_text())
    d["calibration_mode"] = "absolute"
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="PERCENTILE"):
        RouterThreshold.load(bad)


# ---------------------------------- the serve cap and the project hard stop are separate


def _actual_row(run_id, usd):
    return {"timestamp_utc": "2026-09-11T00:00:00+00:00", "run_id": run_id,
            "provider": "anthropic", "model": "claude-sonnet-5", "batch_id": None,
            "input_tokens": 1, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0, "output_tokens": 1, "cost_usd": usd,
            "cumulative_usd": usd, "kind": "actual", "estimated_usd": None,
            "n_requests": 1, "notes": None}


def _ledger_with(tmp_path, rows):
    from src.api.ledger import SpendLedger

    # NOT "spend_ledger.jsonl": make_tier2() creates a throwaway ledger at exactly that
    # name under the same tmp_path and truncates it, which silently emptied these rows.
    path = tmp_path / "seeded_ledger.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return SpendLedger(path)


def test_experiment_spend_does_not_consume_the_serve_cap(tmp_path, monkeypatch):
    """THE OUTAGE THIS FIXES. The serve cap read the WHOLE ledger, so $1.00 was compared
    against $3.58 of experiment spend: the cap was permanently reached, every escalation
    refused, and every response came back escalation_skipped=true looking perfectly
    normal. Experiment rows must not count against a limit on SERVING."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    ledger = _ledger_with(tmp_path, [
        _actual_row("stage1_claude-sonnet-5", 3.20),
        _actual_row("rung2_claude-sonnet-5", 0.381478),
    ])
    t = make_tier2(tmp_path)
    t.ledger, t.spend_cap_usd = ledger, 1.00

    assert ledger.cumulative_usd() == pytest.approx(3.581478)
    assert t.spent_usd() == 0.0, "no serving spend has been recorded"
    assert t.cap_reached() is False
    t.classify("a clause")          # must not raise SpendCapReached


def test_the_serve_cap_counts_only_this_services_own_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    ledger = _ledger_with(tmp_path, [
        _actual_row("stage1_claude-sonnet-5", 9.00),      # huge, but not serving
        _actual_row("serve_tier0_sonnet5", 0.40),
        _actual_row("some_other_service", 5.00),          # also not ours
    ])
    t = make_tier2(tmp_path)
    t.ledger, t.spend_cap_usd = ledger, 1.00

    assert t.spent_usd() == pytest.approx(0.40)
    assert t.cap_reached() is False


def test_the_serve_cap_fires_on_this_services_own_spend(tmp_path, monkeypatch):
    from src.serve.tier2 import SpendCapReached

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    ledger = _ledger_with(tmp_path, [_actual_row("serve_tier0_sonnet5", 1.50)])
    client = FakeAnthropicClient()
    t = make_tier2(tmp_path, client)
    t.ledger, t.spend_cap_usd = ledger, 1.00

    assert t.spent_usd() == pytest.approx(1.50)
    with pytest.raises(SpendCapReached, match="run_id"):
        t.classify("a clause")
    assert len(client.calls) == 0


def test_changing_the_run_id_rescopes_the_cap(tmp_path, monkeypatch):
    """Documented footgun: a new run id is a new accounting bucket, so prior serving
    spend stops counting. Pinned so the behaviour is deliberate, not discovered."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    ledger = _ledger_with(tmp_path, [_actual_row("serve_tier0_sonnet5", 1.50)])
    t = make_tier2(tmp_path)
    t.ledger, t.spend_cap_usd = ledger, 1.00

    assert t.cap_reached() is True
    t.run_id = "serve_tier0_sonnet5_v2"
    assert t.spent_usd() == 0.0 and t.cap_reached() is False


def test_the_hard_stop_still_reads_every_run_id(tmp_path):
    """The $15 project limit governs ALL spend and must NOT be scoped to one producer."""
    from src.api.ledger import BudgetExceeded

    ledger = _ledger_with(tmp_path, [
        _actual_row("stage1_claude-sonnet-5", 9.00),
        _actual_row("serve_tier0_sonnet5", 5.00),
    ])
    assert ledger.cumulative_usd() == pytest.approx(14.00)
    assert ledger.cumulative_usd(run_id="serve_tier0_sonnet5") == pytest.approx(5.00)
    assert ledger.remaining_usd() == pytest.approx(1.00)

    ledger.assert_within_budget(0.50)              # inside the hard stop
    with pytest.raises(BudgetExceeded, match="hard stop"):
        ledger.assert_within_budget(2.00)          # 14.00 + 2.00 > 15.00


def test_serving_spend_does_count_toward_the_hard_stop(tmp_path):
    """The scoping is one-directional: experiment spend is invisible to the serve cap,
    but serving spend is NOT invisible to the project budget."""
    ledger = _ledger_with(tmp_path, [_actual_row("serve_tier0_sonnet5", 14.60)])
    from src.api.ledger import BudgetExceeded

    with pytest.raises(BudgetExceeded):
        ledger.assert_within_budget(0.50)


def test_estimate_rows_are_money_for_neither_limit(tmp_path):
    est = dict(_actual_row("serve_tier0_sonnet5", 7.00), kind="estimate",
               cost_usd=0.0, estimated_usd=7.00)
    ledger = _ledger_with(tmp_path, [est, _actual_row("serve_tier0_sonnet5", 0.25)])
    assert ledger.cumulative_usd() == pytest.approx(0.25)
    assert ledger.cumulative_usd(run_id="serve_tier0_sonnet5") == pytest.approx(0.25)


def test_every_refusal_check_precedes_the_confirmed_banner():
    """`require_confirmation` prints "CONFIRMED — projected cumulative: $X" on its way
    out, so any refusal placed after it prints a refusal UNDER a line saying the run was
    confirmed. That is how the smoke test first read: CONFIRMED, then the cap refusal.
    Order is asserted by AST rather than by reading, so it cannot quietly regress."""
    import ast

    src = (ROOT / "scripts" / "smoke_live_tier2.py").read_text()
    tree = ast.parse(src)
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")

    cap_lines, confirm_lines = [], []
    for node in ast.walk(main):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cap_reached":
            cap_lines.append(node.lineno)
        if isinstance(node.func, ast.Name) and node.func.id == "require_confirmation":
            confirm_lines.append(node.lineno)

    assert cap_lines, "the smoke test must check the serve cap"
    assert confirm_lines, "the smoke test must gate on require_confirmation"
    assert min(cap_lines) < min(confirm_lines), (
        f"the serve-cap refusal (line {min(cap_lines)}) must run BEFORE "
        f"require_confirmation (line {min(confirm_lines)}), which prints CONFIRMED")


def test_the_smoke_test_gates_on_the_serve_cap_as_well_as_the_hard_stop():
    """Two limits, both checked. The hard stop comes via require_confirmation; the serve
    cap has to be checked explicitly because require_confirmation knows nothing about it."""
    src = (ROOT / "scripts" / "smoke_live_tier2.py").read_text()
    assert "cumulative_usd(run_id=" in src, \
        "the serve cap must read the ledger SCOPED to this service's run id"
    assert "tier2_spend_cap_usd" in src and "require_confirmation" in src
