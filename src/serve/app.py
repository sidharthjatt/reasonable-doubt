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
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.serve.canary_gate import CanaryGate, CanaryStatus, Tier0NotQualified
from src.serve.config import ServiceConfig
from src.serve.pricing import CostEstimate, api_usd_for_result, local_usd_per_request
from src.serve.tier2 import Tier2Unavailable
from src.serve.tiers import TierResult

__all__ = ["Cascade", "ClassifyRequest", "MarginRouter", "Tier0NotQualified",
           "build_cascade", "create_app"]


# INPUT CAP. A clause is a paragraph; 512 tokens is roughly 2-4k characters and anything
# past that is truncated by the tokenizer anyway. The cap exists so a public endpoint
# cannot be handed a novel and made to tokenise it: the work is bounded BEFORE the
# tokenizer runs, not after. Exceeding it is a 422 naming the limit, never a silent trim.
MAX_INPUT_CHARS = 20_000

# The one-page demo UI. Served from disk so the page is editable without touching Python.
INDEX_HTML_PATH = Path(__file__).resolve().parent / "static" / "index.html"

# What the warming-up UI tells a visitor to expect. A rough figure from the measured cold
# start, used for copy only — nothing branches on it.
CANARY_ESTIMATED_SECONDS = 90


class ClassifyRequest(BaseModel):
    """Defined at MODULE level, deliberately.

    `from __future__ import annotations` turns every annotation into a string that
    FastAPI resolves against the module namespace. A request model nested inside
    `create_app()` is invisible there, so FastAPI silently reinterprets the parameter as
    a QUERY parameter and every POST /classify returns 422 "field required". The
    endpoint looks wired and is not — and no unit test of `Cascade` can see it, because
    the cascade itself works fine.
    """

    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


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
    # THE QUALIFICATION GATE. Default-open only because a cascade built without one is a
    # unit under test, never the served object: build_cascade() always supplies a real
    # gate, and create_app() is what settles it.
    gate: "CanaryGate | None" = None

    def classify(self, text: str) -> dict:
        # CHECKED HERE, ON EVERY PREDICTION, and deliberately not in the endpoint. An
        # HTTP-level check is bypassed by any other caller of this object, and "the
        # endpoint forgot" is the failure being guarded against (§3bp).
        if self.gate is not None:
            self.gate.check_may_serve()
        t0 = self.tier0.classify(text)
        margin = t0.confidence
        # PRECISION-KEYED (§3bl). Passing the served precision is what stops this
        # response quoting the INT8 row's cost while the service runs FP32 (§3bh).
        costs: list[tuple[str, CostEstimate]] = [
            ("tier0", local_usd_per_request(precision=self.config.tier0_precision))]

        # THE ROUTER ALWAYS RUNS. Its selection is information about the clause and is
        # reported whether or not anything is done with it (§3bm).
        low_confidence = self.router.escalate(t0)
        escalation_enabled = self.config.tier2_enabled
        # ESCALATION IS A SEPARATE QUESTION FROM LOW CONFIDENCE, and conflating them is
        # what `escalation_skipped` did. With Tier 2 disabled by config a flagged clause
        # is NOT a skipped escalation — nothing was selected for escalation at all, so
        # `escalation_selected` is false and `escalation_skipped` stays false. A reader
        # counting escalation_skipped as "escalations we failed to make" would otherwise
        # read a deliberate architecture as a degraded one.
        wants_escalation = low_confidence and escalation_enabled
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
            # The router's own verdict on this clause, independent of what is done with
            # it. Tier 0 answers these at 0.3488 accuracy against 0.8733 overall (§3bm),
            # so the flag is the useful part even with escalation off.
            "low_confidence": low_confidence,
            "needs_review": low_confidence,
            "needs_review_meaning": (
                "low confidence — human review recommended"
                if low_confidence else None),
            # Tier 0's own distribution, always — not the answering tier's. With
            # escalation off these are the same; with it on, the API tier returns a
            # label and no distribution, and showing an empty list would read as "no
            # candidates" rather than "this tier does not produce one".
            "top_3": ([{"label": lbl, "score": sc} for lbl, sc in t0.top_k]
                      if t0.top_k is not None else None),
            "escalation_enabled": escalation_enabled,
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
                "escalation_enabled": escalation_enabled,
                "threshold": self.router.threshold,
                "calibrated_on": self.config.threshold.calibrated_on,
                "calibrated_artefact": self.config.threshold.artefact,
                "computed_at_request_time": False,
            },
        }


def build_cascade(config: ServiceConfig | None = None, *, tier0=None,
                  tier2=None) -> Cascade:
    """Construct the deployed cascade. Loads the ONNX session once.

    `tier0` is an injection point that mirrors `tier2`. It exists for tests of the HTTP
    surface — request validation, the response shape, the demo page, `/health` — none of
    which care which encoder answers. `models/` is gitignored, so without it those tests
    could only skip in CI, and the service's HTTP layer would have no CI coverage at all.
    Production passes nothing and gets the real encoder.
    """
    from src.data.labels import load_labels
    from src.serve.startup_timing import phase
    from src.serve.tier0 import Tier0Encoder
    from src.serve.tier2 import Tier2Claude

    with phase("config_load"):
        config = config or ServiceConfig.load()
    with phase("labels_load"):
        labels = load_labels()

    tier0 = tier0 if tier0 is not None else Tier0Encoder(
        model_dir=config.tier0_model_dir, labels=labels,
        max_length=config.tier0_max_length)
    tier2 = tier2 if tier2 is not None else Tier2Claude(
        model=config.tier2_model, labels=labels,
        max_output_tokens=config.tier2_max_output_tokens,
        temperature=config.tier2_temperature, batch=config.tier2_batch,
        spend_cap_usd=config.tier2_spend_cap_usd, run_id=config.tier2_run_id)

    return Cascade(tier0=tier0, tier2=tier2, config=config,
                   router=MarginRouter(threshold=config.threshold.threshold,
                                       signal=config.threshold.signal),
                   gate=CanaryGate())


def run_startup_canary(cascade: Cascade) -> dict[str, Any]:
    """Refuse to start unless Tier 0 reproduces its reference accuracy on THIS host."""
    from src.data.loading import load_ledgar
    from src.serve.canary import CanarySet, hardware_report, run_canary
    import json
    import time

    cfg = cascade.config
    if not cfg.canary_enabled:
        # Explicit, reported, and never the default. A disabled canary is a decision
        # the operator has to make and the response says it was made.
        return {"ran": False, "reason": "disabled in configs/serve.yaml",
                "hardware": hardware_report()}

    from src.serve.startup_timing import phase, phases_summary

    with phase("canary_rowset_read"):
        canary = CanarySet.from_dict(json.loads(cfg.canary_row_set.read_text()))
    # load_ledgar() USED TO SIT INSIDE THE CANARY'S TIMER as an argument expression, which
    # is why the 133.5s figure in the Cloud Run logs is dataset load AND inference added
    # together. They are separated here because they have nothing to do with each other.
    with phase("dataset_load"):
        ds = load_ledgar()
    started = time.monotonic()
    result = run_canary(cascade.tier0, ds, canary, floor=cfg.canary_min_accuracy,
                        on_progress=(cascade.gate.note_progress
                                     if cascade.gate is not None else None))
    print(f"STARTUP PHASE canary_run: {time.monotonic() - started:.2f}s "
          f"n={result.n} per_row_ms={1000 * (time.monotonic() - started) / result.n:.1f}",
          flush=True)
    # LOG IT. The number that decides whether this process is allowed to serve belongs in
    # the logs, not only behind /health: on a managed platform the logs are what you have
    # when a revision fails to come up, and /health is exactly what you cannot reach then.
    # A failure already raises with its own detail; this covers the passing case, which is
    # otherwise invisible.
    hw = result.hardware
    print(f"TIER 0 CANARY PASSED: {result.n_correct}/{result.n} = {result.accuracy:.4f} "
          f"(floor {cfg.canary_min_accuracy:.4f}, reference "
          f"{cfg.canary_measured_accuracy:.4f}) in {time.monotonic() - started:.1f}s "
          f"| precision={cfg.tier0_precision} artefact={cfg.tier0_model_dir.name} "
          f"| {hw.get('system')}/{hw.get('machine')} ort {hw.get('onnxruntime_version')} "
          f"isa={hw.get('cpu_isa_flags')}", flush=True)
    print(phases_summary(), flush=True)
    return {"ran": True, "reference_accuracy": cfg.canary_measured_accuracy,
            **result.as_dict()}


def create_app(config: ServiceConfig | None = None, *, tier0=None, tier2=None,
               run_canary_on_start: bool = True, canary_background: bool = True):
    """Build the FastAPI app.

    THE CANARY NO LONGER BLOCKS THE PORT (§3bp). It runs on a background thread while
    uvicorn binds immediately, so a cold request gets an instant 503 that says what is
    happening instead of hanging for two minutes. What did NOT change: no prediction is
    returned until the canary passes, and a failure refuses permanently. That property
    lives in `Cascade.gate`, which `classify` checks on every call.

    `canary_background=False` runs it inline, which is what the tests use when they need
    the outcome to be settled before the first assertion.
    """
    import threading

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    from src.serve.canary import CanaryFailure

    cascade = build_cascade(config, tier0=tier0, tier2=tier2)
    gate = cascade.gate
    assert gate is not None, "build_cascade must supply a gate"

    def _run_canary_into_gate(*, reraise: bool) -> None:
        """Run the canary and settle the gate.

        `reraise` is the difference between the two modes, and it is a deliberate one.
        BACKGROUND: the port is already open, so a failure is recorded and every
        prediction is refused; the process stays up because a crash-loop on a managed
        platform leaves nothing to inspect, while a live /health says exactly what went
        wrong. BLOCKING: nothing is serving yet, so the failure propagates and the
        process exits non-zero, which is what `docker run` and CI expect.
        """
        try:
            result = run_startup_canary(cascade)
        except CanaryFailure as exc:
            print(f"TIER 0 CANARY FAILED — THIS SERVICE WILL NOT PREDICT: {exc}",
                  flush=True)
            gate.mark_failed(str(exc))
            if reraise:
                raise
            return
        except Exception as exc:                       # noqa: BLE001
            # Any other error also means unqualified. It is recorded as a failure with
            # its type rather than swallowed: an exception here must never leave the gate
            # PENDING forever, because "pending" would look like "still warming up".
            print(f"TIER 0 CANARY ERRORED — THIS SERVICE WILL NOT PREDICT: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            gate.mark_failed(f"{type(exc).__name__}: {exc}")
            if reraise:
                raise
            return
        if result.get("ran"):
            gate.mark_passed(result)
        else:
            gate.mark_skipped(str(result.get("reason", "canary disabled")))

    if not run_canary_on_start:
        gate.mark_skipped("explicitly skipped by the caller")
        canary = {"ran": False, "reason": "explicitly skipped by the caller"}
    elif canary_background:
        canary = None
        threading.Thread(target=_run_canary_into_gate, name="canary",
                         kwargs={"reraise": False}, daemon=True).start()
    else:
        _run_canary_into_gate(reraise=True)
        canary = gate.as_dict()

    app = FastAPI(
        title="Reasonable Doubt — Tier 0 -> Claude Sonnet 5",
        description="Deployed two-tier cascade (E4b-A). No Tier 1: see PREREGISTRATION.")

    @app.get("/health")
    def health() -> dict:
        from src.ort_runtime import telemetry_disabled
        from src.serve.canary import cpu_isa_flags, onnxruntime_version
        return {
            "status": "ok",
            # Reads the SERVED precision. It was hardcoded "tier0_int8" and kept
            # saying so while the service served FP32 — a health endpoint describing a
            # different system than the one answering.
            "architecture": (f"tier0_{cascade.config.tier0_precision} -> "
                             f"{cascade.config.tier2_model} (no tier1; E4b-A)"),
            "tier0": {**cascade.tier0.describe(),
                      "precision": cascade.config.tier0_precision,
                      "model_dir_from_env": cascade.config.tier0_model_dir_from_env},
            "tier2": cascade.tier2.describe(),
            "router": cascade.config.threshold.as_dict(),
            # The gate is the source of truth. `canary` is only the inline-mode copy.
            "canary": {**(canary or {}), **gate.as_dict()},
            "runtime": {
                "onnxruntime_version": onnxruntime_version(),
                # null means UNREADABLE, not absent — see src/serve/canary.py.
                "cpu_isa_flags": cpu_isa_flags(),
                # ORT telemetry is off: it crashes at teardown on macOS and the service
                # should not phone home regardless (src/ort_runtime.py).
                "ort_telemetry_disabled": telemetry_disabled(),
                "machine": __import__("platform").machine(),
                # Where this process is running, when the platform tells us. Set at
                # deploy time; ABSENT rather than guessed, because the page prints it as
                # provenance and a hardcoded region would have claimed "asia-south1"
                # while running on a laptop.
                "region": __import__("os").environ.get("SERVICE_REGION") or None,
            },
            # TWO DIFFERENT FACTS, kept apart. `configured` is the operator's
            # decision (§3bm turned it off); `tier2_available` is whether a key is
            # present at all. Reporting only the latter, as this did, said
            # "escalation_enabled: false" for a missing key and for a deliberate
            # architecture alike.
            "escalation": {
                "configured": cascade.config.tier2_enabled,
                "tier2_available": cascade.tier2.available,
                "effective": cascade.config.tier2_enabled and cascade.tier2.available,
                "note": ("escalation is OFF by configuration (§3bm): at the served "
                         "operating point it did not improve accuracy "
                         "(+0.0008 macro-F1, 95% CI [-0.0060, +0.0072]). Low-confidence "
                         "rows are FLAGGED instead."
                         if not cascade.config.tier2_enabled else None),
            },
            "escalation_enabled": cascade.config.tier2_enabled and cascade.tier2.available,
        }

    @app.post("/classify")
    def classify(req: ClassifyRequest) -> dict:
        try:
            return cascade.classify(req.text)
        except Tier0NotQualified as exc:
            status = gate.status
            # 503 for both, and the distinction is in the body rather than the code:
            # both mean "this service is not answering", and a client that only reads
            # the status code must not treat a permanent refusal as a retryable one.
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "tier0_not_qualified",
                    "canary_status": status.value,
                    "retryable": status is CanaryStatus.PENDING,
                    "message": str(exc),
                    "estimated_wait_seconds": (
                        CANARY_ESTIMATED_SECONDS
                        if status is CanaryStatus.PENDING else None),
                },
            ) from exc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Read per request, not cached at import: the page is a few KB, and a module
        # constant would mean a missing file fails at IMPORT time — i.e. before the
        # canary — turning a cosmetic problem into a startup failure of the API.
        return INDEX_HTML_PATH.read_text(encoding="utf-8")

    return app
