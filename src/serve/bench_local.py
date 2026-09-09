"""Mac Mini measurement harness for E6's cost-vs-volume curve.

    python -m src.serve.bench_local --onnx-dir models/tier0/int8 --n 500

Measures the three quantities §1b lists as null, and that E6 needs:

  * **throughput (req/s)** — sets `V_max`, the volume beyond which the local curve
    steps to a second device rather than continuing to fall.
  * **power draw (watts)** — MANDATORY under the curve framing. It is the asymptote of
    the local cost curve; without it the curve tends to zero and local always wins at
    volume, which is false.
  * latency p50/p95 — reported, but NOT on E6's critical path (it does not enter the
    cost curve). E7 is what tests it.

No GPU, no API calls, no network. Power comes from `powermetrics`, which needs sudo;
if it is unavailable the harness **raises** rather than reporting a throughput-only
result that would silently produce a wrong asymptote (hard rule 11).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchResult:
    tier: str
    artefact: str
    n_requests: int
    batch_size: int
    wall_seconds: float
    throughput_rps: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    idle_soc_watts: float | None
    load_soc_watts: float | None
    marginal_soc_watts: float | None
    joules_per_request: float | None
    power_scope: str = "soc_package_cpu_gpu_ane_excludes_ram_ssd_psu_fans"

    def as_dict(self) -> dict:
        return asdict(self)


def measure_power(seconds: int = 20) -> float:
    """Mean **SoC package** power in watts via `powermetrics`. Raises if unavailable.

    IMPORTANT — what this number is and is not. `--samplers cpu_power` reports
    "Combined Power (CPU + GPU + ANE)": the power drawn by the SoC package only. It
    EXCLUDES RAM, SSD, PSU losses, networking and fans, so it is NOT wall power. A Mac
    Mini that reports ~0.2 W idle here draws several watts at the wall. For a dedicated
    device, E6 wants wall power, which cannot be obtained from powermetrics at all —
    it needs an external meter (a smart plug). Fields are named `*_soc_watts`
    throughout so the report cannot claim more than was measured.

    Hard rule 11: no fallback. A missing power figure makes E6's asymptote wrong in
    exactly the region the hypothesis is about, so a throughput-only result must not
    silently stand in for a complete one.
    """
    if shutil.which("powermetrics") is None:
        raise RuntimeError(
            "powermetrics not found. It is required: power draw is the asymptote of "
            "the local cost curve (E6 condition 3). Refusing to report a "
            "throughput-only benchmark."
        )
    proc = subprocess.run(
        ["sudo", "-n", "powermetrics", "--samplers", "cpu_power",
         "-i", "1000", "-n", str(seconds)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "powermetrics failed (it needs sudo). Run with passwordless sudo for "
            f"powermetrics, or run the harness under sudo. stderr: {proc.stderr[:300]}"
        )
    watts = [
        float(line.split(":")[1].strip().split()[0]) / 1000.0
        for line in proc.stdout.splitlines()
        if "Combined Power" in line and "mW" in line
    ]
    if not watts:
        raise RuntimeError("powermetrics produced no 'Combined Power' samples to parse")
    return statistics.mean(watts)


def load_runtime(onnx_dir: Path):
    """Load tokenizer and ORT session ONCE.

    Loading was previously inside bench_onnx, so the power-measurement spin loop
    re-initialised a 245 MB session on every pass and measured model loading rather
    than inference. Timing was unaffected (loading sat outside the timer), but the
    watt figure was not.
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(onnx_dir))
    sess = ort.InferenceSession(str(next(onnx_dir.glob("*.onnx"))),
                                providers=["CPUExecutionProvider"])
    return tok, sess, {i.name for i in sess.get_inputs()}


def bench_onnx(runtime, texts: list[str], batch_size: int, max_length: int):
    """Time ONNX-INT8 inference. Returns (per-request latencies ms, wall seconds).

    Tokenisation is INSIDE the timed region — it is part of serving cost. Model and
    tokenizer loading is outside it, done once by load_runtime.
    """
    import numpy as np

    tok, sess, inputs = runtime
    latencies: list[float] = []
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, truncation=True, max_length=max_length,
                  padding=True, return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in inputs}
        t1 = time.perf_counter()
        sess.run(None, feed)
        dt = (time.perf_counter() - t1) * 1000.0
        latencies.extend([dt / len(chunk)] * len(chunk))
    return latencies, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--tier", default="tier0_encoder_onnx_int8")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--skip-power", action="store_true",
                    help="EXPLICIT opt-out. Result is marked incomplete and cannot "
                         "be used for E6.")
    # PER-RUN FILENAME, AND REFUSAL TO OVERWRITE. Two rounds of the same defect got
    # this far: a fixed path (cost E5 two seeds), then a per-ARTEFACT path, which still
    # collides the moment the same artefact is measured more than once -- which is
    # exactly what the 9-run replication does. Naming is now per (artefact, run), and
    # the write REFUSES an existing file rather than replacing it.
    #
    # The refusal is the part that generalises. A naming scheme only defends against
    # the collisions its author thought of; refusing to overwrite defends against the
    # ones they did not, and turns a silent discard into a stop.
    ap.add_argument("--run", type=int, default=1,
                    help="replicate index; part of the default filename")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result file (default: refuse)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/bench_<tier>_<artefact>_run<run>.json")
    args = ap.parse_args()
    if args.out is None:
        args.out = Path(
            f"results/bench_{args.tier}_{Path(args.onnx_dir).name}_run{args.run}.json")
    if args.out.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {args.out}. A benchmark result already exists for "
            f"this (artefact, run) pair. Use --run with a fresh index, or --force if "
            f"you genuinely mean to discard the earlier measurement.")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.loading import get_split, load_ledgar
    from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest, verify_manifest

    ds = load_ledgar()
    man = load_manifest("test_3000", DEFAULT_MANIFEST_DIR)
    verify_manifest(man, ds, DEFAULT_MANIFEST_DIR)
    split = get_split(ds, man.split)
    texts = list(split.select(man.indices[: args.n])["text"])

    print(f"loading runtime…")
    runtime = load_runtime(args.onnx_dir)
    print(f"warming up ({args.warmup} rows)…")
    bench_onnx(runtime, texts[: args.warmup], args.batch_size, args.max_length)

    idle = load_w = marginal = joules = None
    if not args.skip_power:
        print("measuring idle power (20s)…")
        idle = measure_power(20)

    print(f"benchmarking {len(texts)} rows, batch {args.batch_size}…")
    lat, wall = bench_onnx(runtime, texts, args.batch_size, args.max_length)
    rps = len(texts) / wall

    if not args.skip_power:
        print("measuring power under sustained load…")
        import threading
        stop = threading.Event()

        def spin():
            while not stop.is_set():
                bench_onnx(runtime, texts[:64], args.batch_size, args.max_length)

        t = threading.Thread(target=spin, daemon=True)
        t.start()
        load_w = measure_power(20)
        stop.set()
        marginal = load_w - idle
        joules = marginal / rps  # watts / (req/s) = joules per request

    lat.sort()
    res = BenchResult(
        tier=args.tier, artefact=str(args.onnx_dir), n_requests=len(texts),
        batch_size=args.batch_size, wall_seconds=wall, throughput_rps=rps,
        p50_latency_ms=lat[len(lat) // 2],
        p95_latency_ms=lat[int(0.95 * len(lat))],
        p99_latency_ms=lat[int(0.99 * len(lat))],
        idle_soc_watts=idle, load_soc_watts=load_w, marginal_soc_watts=marginal,
        joules_per_request=joules,
    )
    print("\n" + "=" * 60)
    print(f"  throughput      : {rps:.2f} req/s")
    print(f"  latency p50/p95 : {res.p50_latency_ms:.1f} / {res.p95_latency_ms:.1f} ms")
    if joules is not None:
        print(f"  SoC power idle/load : {idle:.2f} / {load_w:.2f} W "
              f"(marginal {marginal:.2f} W)")
        print( "    NOTE: SoC package only (CPU+GPU+ANE). Excludes RAM, SSD, PSU, fans.")
        print( "    Wall power needs an external meter; powermetrics cannot report it.")
        print(f"  energy          : {joules:.3f} J/request")
        print("\n  -> paste into configs/costs.yaml local_hardware:")
        print(f"       measured_throughput_rps: {rps:.2f}")
        print(f"       power_draw_soc_watts: {load_w:.2f}   # NOT wall power")
        print(f"     and per_tier_throughput.{args.tier}.requests_per_second: {rps:.2f}")
    else:
        print("  POWER NOT MEASURED — this result is INCOMPLETE and must not be used "
              "for E6 (see §1b).")
    print("=" * 60)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = res.as_dict()
    payload["complete_for_e6"] = joules is not None
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
