"""The per-seed filename and the aggregate's refusal — the bug class that ate seeds 1-2."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "e5_sweep.py"


def _npz(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    lb = rng.integers(0, 100, 300)
    lg = rng.normal(size=(300, 100))
    lg[np.arange(300), lb] += rng.normal(2.0, 1.0, 300)
    np.savez_compressed(path, dev_logits=lg, dev_labels=lb,
                        dev_2000_indices=np.arange(300),
                        dev_absent_classes=np.array([14]))
    return path


def _run(*args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=cwd)


def test_default_output_path_carries_the_seed(tmp_path):
    """Two seeds must not write the same file. This is the defect being regression-tested:
    a seedless default silently discarded every seed but the last."""
    (tmp_path / "results").mkdir()
    written = []
    for s in (1, 2):
        npz = _npz(tmp_path / f"dev_logits_ce_seed{s}.npz", s)
        r = _run("--npz", str(npz), "--seed", str(s), cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        written.append(tmp_path / "results" / f"e5_sweep_fp32_seed{s}.json")
    assert all(p.exists() for p in written), [str(p) for p in written]
    assert len({p.name for p in written}) == 2
    # and they are genuinely different runs, not one file read twice
    a, b = (json.loads(p.read_text()) for p in written)
    assert a["seed"] == 1 and b["seed"] == 2
    assert a["aurocs"] != b["aurocs"]


def test_aggregate_refuses_on_fewer_than_three_seeds(tmp_path):
    """A mean over 2 seeds is not the quantity hard rule 2 names; refusing beats
    averaging whatever happens to be on disk."""
    (tmp_path / "results").mkdir()
    for s in (1, 2):
        npz = _npz(tmp_path / f"dev_logits_ce_seed{s}.npz", s)
        assert _run("--npz", str(npz), "--seed", str(s), cwd=tmp_path).returncode == 0
    r = _run("--aggregate", cwd=tmp_path)
    assert r.returncode != 0
    assert "seed3" in (r.stdout + r.stderr) and "hard rule 2" in (r.stdout + r.stderr)


def test_aggregate_reports_rank_stability(tmp_path):
    (tmp_path / "results").mkdir()
    for s in (1, 2, 3):
        npz = _npz(tmp_path / f"dev_logits_ce_seed{s}.npz", s)
        assert _run("--npz", str(npz), "--seed", str(s), cwd=tmp_path).returncode == 0
    r = _run("--aggregate", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    d = json.loads((tmp_path / "results" / "e5_sweep_fp32_aggregate.json").read_text())
    assert d["n_seeds"] == 3
    assert set(d["rank_stable"]) == set(d["aurocs"])
    for sig, v in d["aurocs"].items():
        assert len(v["per_seed"]) == 3 and len(v["per_seed_rank"]) == 3
    assert "rule_spread_ge_0.05" in d and "rule_auroc_ge_0.75" in d


def test_int8_file_is_refused_by_substring_not_prefix(tmp_path):
    (tmp_path / "results").mkdir()
    npz = _npz(tmp_path / "dev_logits_int8_ce_seed1.npz", 1)
    r = _run("--npz", str(npz), "--seed", "1", cwd=tmp_path)
    assert r.returncode != 0
    assert "refusing" in (r.stdout + r.stderr)
