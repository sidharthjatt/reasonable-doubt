#!/usr/bin/env python3
"""C2 (confidence anchoring) and C4 (exemplar class over-prediction), on disk, at $0.00.

Both accept rules were registered 2026-09-07 and adopted VERBATIM 2026-09-08 (§3ad); this
script only executes them. C4's rate-fallback switch was fixed from the ZERO-SHOT rate
alone in §3am, committed BEFORE this script reads a single few-shot prediction:
`Governing Laws` at 5.80% zero-shot is above the 1% trigger, so the RATIO form (>= 2x)
applies and the +2pp fallback does not. That decision is hard-coded here, not recomputed,
so it cannot drift once the few-shot numbers are visible.

C2 SCOPE: Sonnet 5 only (§3b). C4's second clause guards against general few-shot drift
being misread as exemplar-specific repetition.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "claude-sonnet-5"
THREE_X = "Governing Laws"
ONE_X = ["Compliance With Laws", "Organizations", "Remedies", "Solvency", "Waivers"]
# Fixed in §3am from the zero-shot rate alone, before any few-shot prediction was read.
C4_FORM = "ratio"
C4_RATIO_BAR = 2.0
C2_SD_RATIO_BAR = 0.5
C2_MODE_TARGET, C2_MODE_TOL = 0.9, 0.02


def leg(path: Path, ids: list[str]) -> tuple[list[str | None], list[float | None]]:
    """Labels and verbalized confidences for the given rows, canonical parser."""
    from src.data.schema import ParseFailure, parse_response
    rows = json.loads(path.read_text())["models"][MODEL]["rows"]
    labs: list[str | None] = []
    confs: list[float | None] = []
    for i in ids:
        try:
            r = parse_response(rows[i].get("text", ""))
            labs.append(r.label)
            confs.append(float(r.confidence) if r.confidence is not None else None)
        except ParseFailure:
            labs.append(None)
            confs.append(None)
    return labs, confs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-json", type=Path, default=Path("results/c2_c4.json"))
    a = ap.parse_args()

    man = json.loads(Path("configs/manifests/test_3000.json").read_text())
    ids = [str(i) for i in man["indices"][:1000]]
    zs_lab, zs_conf = leg(a.results_dir / "stage1_results.json", ids)
    fs_lab, fs_conf = leg(a.results_dir / "fewshot_results.json", ids)
    n = len(ids)
    print(f"  {n} shared rows, {MODEL}\n")

    # ---------------- C2 ----------------------------------------------------------------
    zc = np.array([c for c in zs_conf if c is not None])
    fc = np.array([c for c in fs_conf if c is not None])
    if len(zc) != n or len(fc) != n:
        print(f"  NOTE: confidences present on {len(zc)}/{n} zero-shot, {len(fc)}/{n} "
              f"few-shot rows; C2 is computed on the rows that carry one.")
    zsd, fsd = float(zc.std(ddof=1)), float(fc.std(ddof=1))
    ratio = fsd / zsd
    zmode = float(Counter(zc.tolist()).most_common(1)[0][0])
    fmode = float(Counter(fc.tolist()).most_common(1)[0][0])
    c2_sd_ok = ratio <= C2_SD_RATIO_BAR
    c2_mode_ok = abs(fmode - C2_MODE_TARGET) <= C2_MODE_TOL
    c2_pass = c2_sd_ok and c2_mode_ok

    print("C2 — few-shot confidence anchoring (Sonnet 5 only)")
    print(f"  zero-shot: mean {zc.mean():.4f}  sd {zsd:.4f}  mode {zmode:.2f}  "
          f"{len(set(zc.tolist()))} distinct values")
    print(f"  few-shot : mean {fc.mean():.4f}  sd {fsd:.4f}  mode {fmode:.2f}  "
          f"{len(set(fc.tolist()))} distinct values")
    print(f"  sd ratio  {ratio:.4f}  (bar <= {C2_SD_RATIO_BAR})            "
          f"{'PASS' if c2_sd_ok else 'FAIL'}")
    print(f"  mode      {fmode:.2f}   (bar within {C2_MODE_TOL} of "
          f"{C2_MODE_TARGET})   {'PASS' if c2_mode_ok else 'FAIL'}")
    print(f"  C2: {'PASS — anchoring detected' if c2_pass else 'FAIL — accept rule not met'}")
    top_z = Counter(zc.tolist()).most_common(5)
    top_f = Counter(fc.tolist()).most_common(5)
    print(f"  top zero-shot values: {top_z}")
    print(f"  top few-shot  values: {top_f}")

    # ---------------- C4 ----------------------------------------------------------------
    print(f"\nC4 — exemplar class over-prediction (form fixed in §3am: {C4_FORM.upper()},"
          f" bar >= {C4_RATIO_BAR:g}x)")
    zr = zs_lab.count(THREE_X) / n
    fr = fs_lab.count(THREE_X) / n
    obs = fr / zr if zr else float("nan")
    shift_3x = fr - zr
    print(f"  {'class':<24}{'zero-shot':>11}{'few-shot':>11}{'ratio':>9}{'shift':>10}")
    print(f"  {THREE_X + ' (3x)':<24}{zr:>11.2%}{fr:>11.2%}{obs:>8.2f}x"
          f"{shift_3x:>+10.2%}")
    one_x = {}
    for c in ONE_X:
        z, f = zs_lab.count(c) / n, fs_lab.count(c) / n
        one_x[c] = {"zero_shot": z, "few_shot": f, "shift": f - z}
        print(f"  {c + ' (1x)':<24}{z:>11.2%}{f:>11.2%}"
              f"{(f / z if z else float('nan')):>8.2f}x{f - z:>+10.2%}")
    max_1x = max(v["shift"] for v in one_x.values())
    c4_ratio_ok = obs >= C4_RATIO_BAR
    c4_exceeds_ok = shift_3x > max_1x
    c4_pass = c4_ratio_ok and c4_exceeds_ok
    print(f"\n  ratio {obs:.2f}x vs bar {C4_RATIO_BAR:g}x                 "
          f"{'PASS' if c4_ratio_ok else 'FAIL'}")
    print(f"  3x shift {shift_3x:+.2%} vs largest 1x shift {max_1x:+.2%}   "
          f"{'PASS' if c4_exceeds_ok else 'FAIL'}")
    print(f"  C4: {'PASS — exemplar repetition biased predictions' if c4_pass else 'FAIL — accept rule not met'}")

    a.out_json.write_text(json.dumps({
        "cost_usd": 0.0, "model": MODEL, "n_rows": n,
        "rows": "test_3000 manifest.indices[:1000] — POSITIONAL",
        "C2": {"registered": "sd ratio <= 0.5 AND mode within 0.02 of 0.9",
               "zero_shot_sd": zsd, "few_shot_sd": fsd, "sd_ratio": ratio,
               "zero_shot_mean": float(zc.mean()), "few_shot_mean": float(fc.mean()),
               "zero_shot_mode": zmode, "few_shot_mode": fmode,
               "zero_shot_distinct": len(set(zc.tolist())),
               "few_shot_distinct": len(set(fc.tolist())),
               "sd_clause_pass": c2_sd_ok, "mode_clause_pass": c2_mode_ok,
               "pass": c2_pass},
        "C4": {"form": C4_FORM, "form_fixed_in": "§3am, before few-shot was read",
               "three_x_class": THREE_X, "zero_shot_rate": zr, "few_shot_rate": fr,
               "ratio": obs, "shift": shift_3x, "one_x": one_x,
               "max_one_x_shift": max_1x,
               "ratio_clause_pass": c4_ratio_ok, "exceeds_clause_pass": c4_exceeds_ok,
               "pass": c4_pass},
    }, indent=2))
    print(f"\n  wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
