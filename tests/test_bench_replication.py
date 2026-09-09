"""Per-run bench filenames, the overwrite refusal, and the within/between split."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "scripts" / "bench_aggregate.py"


DEFAULT_SHA = "c" * 40


def _bench(d: Path, artefact: str, run: int, rps: float, complete=True,
           sha: str | None = DEFAULT_SHA, dirty: bool = False) -> Path:
    p = d / f"bench_tier0_encoder_onnx_int8_{artefact}_run{run}.json"
    p.write_text(json.dumps({
        "tier": "tier0_encoder_onnx_int8", "artefact": f"models/{artefact}",
        "n_requests": 300, "batch_size": 1, "wall_seconds": 300 / rps,
        "throughput_rps": rps, "p50_latency_ms": 24.0, "p95_latency_ms": 96.0,
        "p99_latency_ms": 150.0, "idle_soc_watts": 0.32, "load_soc_watts": 18.2,
        "marginal_soc_watts": 17.9, "joules_per_request": 17.9 / rps,
        "complete_for_e6": complete,
        "code_commit": sha, "code_dirty": dirty, "run_index": run}))
    return p


def _run(*args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(AGG), *args],
                          capture_output=True, text=True, cwd=cwd)


def test_default_filename_is_unique_per_artefact_and_run():
    """The defect: a per-ARTEFACT path still collides when one artefact is measured
    more than once, which is exactly what the 9-run replication does."""
    from src.serve import bench_local  # noqa: F401  (import guards the module parses)
    names = {f"results/bench_t_{a}_run{r}.json" for a in ("int8_ce_1", "int8_ce_2")
             for r in (1, 2, 3)}
    assert len(names) == 6


def test_aggregate_splits_within_from_between(tmp_path):
    (tmp_path / "results").mkdir()
    d = tmp_path / "results"
    # between-artefact spread ~0, within-artefact spread real
    for a in ("int8_ce_1", "int8_ce_2", "int8_ce_3"):
        for r, v in zip((1, 2, 3), (29.0, 30.0, 29.5)):
            _bench(d, a, r, v)
    r = _run("--results-dir", str(d), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads((d / "bench_tier0_encoder_onnx_int8_aggregate.json").read_text())
    t = out["fields"]["throughput_rps"]
    assert t["n_runs"] == 9
    assert t["between_artefact_sd"] < 1e-9      # identical per-artefact means
    assert t["within_artefact_sd_pooled"] > 0.3
    assert t["between_over_within"] < 0.1
    assert "does not depend on the weights" in r.stdout


def test_aggregate_flags_a_real_between_artefact_effect(tmp_path):
    """Negative control: if one artefact really is slower, the check must SAY so,
    rather than the passing case above meaning 'the check is broken'."""
    (tmp_path / "results").mkdir()
    d = tmp_path / "results"
    for a, base in (("int8_ce_1", 30.0), ("int8_ce_2", 22.0), ("int8_ce_3", 30.0)):
        for r, off in zip((1, 2, 3), (0.0, 0.1, -0.1)):
            _bench(d, a, r, base + off)
    r = _run("--results-dir", str(d), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "weight-independence claim is contradicted" in r.stdout
    assert "do not update costs.yaml" in r.stdout


def test_exclusion_requires_a_stated_reason(tmp_path):
    (tmp_path / "results").mkdir()
    d = tmp_path / "results"
    for a in ("int8_ce_1",):
        for r in (1, 2, 3):
            _bench(d, a, r, 29.0)
    bad = _run("--results-dir", str(d), "--exclude",
               "bench_tier0_encoder_onnx_int8_int8_ce_1_run2.json", cwd=tmp_path)
    assert bad.returncode != 0
    assert "outlier-hunting" in (bad.stdout + bad.stderr)
    ok = _run("--results-dir", str(d), "--exclude",
              "bench_tier0_encoder_onnx_int8_int8_ce_1_run2.json",
              "--exclude-reason", "thermal throttle logged", cwd=tmp_path)
    assert ok.returncode == 0, ok.stderr
    out = json.loads((d / "bench_tier0_encoder_onnx_int8_aggregate.json").read_text())
    assert out["exclude_reason"] == "thermal throttle logged"
    assert out["fields"]["throughput_rps"]["n_runs"] == 2


def test_incomplete_power_run_is_refused(tmp_path):
    (tmp_path / "results").mkdir()
    d = tmp_path / "results"
    _bench(d, "int8_ce_1", 1, 29.0, complete=False)
    r = _run("--results-dir", str(d), cwd=tmp_path)
    assert r.returncode != 0
    assert "complete_for_e6" in (r.stdout + r.stderr)


def test_aggregate_refuses_mixed_code_versions(tmp_path):
    """The exact failure the six-file set demonstrated: two code versions, separable
    only by mtime. A mean over them describes neither."""
    (tmp_path / "results").mkdir()
    d = tmp_path / "results"
    for r, sha in zip((1, 2, 3), ("a" * 40, "a" * 40, "b" * 40)):
        _bench(d, "int8_ce_1", r, 29.0, sha=sha)
    r = _run("--results-dir", str(d), cwd=tmp_path)
    assert r.returncode != 0
    assert "two code versions" in (r.stdout + r.stderr)


def test_aggregate_refuses_unversioned_runs(tmp_path):
    """Files written before artefacts carried a code version cannot be salvaged."""
    (tmp_path / "results").mkdir()
    d = tmp_path / "results"
    _bench(d, "int8_ce_1", 1, 29.0, sha=None)   # code_commit present but null
    r = _run("--results-dir", str(d), cwd=tmp_path)
    assert r.returncode != 0
    assert "NO code_commit" in (r.stdout + r.stderr)


def test_aggregate_accepts_one_consistent_version(tmp_path):
    (tmp_path / "results").mkdir()
    d = tmp_path / "results"
    for a in ("int8_ce_1", "int8_ce_2", "int8_ce_3"):
        for r, v in zip((1, 2, 3), (29.0, 30.0, 29.5)):
            _bench(d, a, r, v)
    r = _run("--results-dir", str(d), cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads((d / "bench_tier0_encoder_onnx_int8_aggregate.json").read_text())
    assert out["code_commit"] == "c" * 40 and out["code_dirty"] is False
