#!/usr/bin/env python
"""E5 exploratory: routing-signal comparison + threshold sweep on dev_2000.

PRECISION PROVENANCE IS NOT OPTIONAL HERE. E5 calibrates thresholds for the DEPLOYED
model, and the deployed precision is INT8 (E1, E3). Kaggle's INT8 is currently at chance,
so this runs on the FP32 dev logits and every output it writes is stamped
FP32-calibrated, INT8 deployment pending E3 resolution.

That stamp is applied by this script, not by whoever reads it: a caller cannot ask for
an unlabelled result. Nothing here is the router result.

Calibration is dev-only (hard rule 1); `assert_threshold_split` enforces it upstream.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.router.load_logits import load_split
from src.router.calibrate import compare_signals, sweep_thresholds

LABEL = "FP32-calibrated, INT8 deployment pending E3 resolution"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path,
                    help="logits_<arm>_seed<n>.npz — the FP32 file, NOT int8_logits_*")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", type=Path, default=Path("results/e5_sweep_fp32.json"))
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    a = ap.parse_args()

    # Substring, not prefix: the dev-inference commit writes dev_logits_int8_*.npz, which
    # a prefix check would wave through — and it uses the SAME `dev_logits` key as the
    # FP32 file, so nothing downstream would notice.
    if "int8" in a.npz.name.lower():
        raise SystemExit(
            f"refusing {a.npz.name}: INT8 dev logits are at chance pending E3 (3ah). "
            "Calibrating on them would fit thresholds to noise and report it as a router.")

    d = load_split(a.npz, "dev_2000", for_calibration=True)   # loader vocabulary, not the npz prefix
    cmp_ = compare_signals(d.logits, d.labels, split="dev_2000")

    print(f"\n=== E5 signal comparison — {LABEL} ===")
    print(f"seed {a.seed}  n={cmp_.n}  tier0 correct={cmp_.n_correct} "
          f"({cmp_.n_correct/cmp_.n:.4f})  classes absent={len(d.absent_classes)}")
    print(f"random-escalation null AUROC {cmp_.random_null:.4f} "
          f"95% [{cmp_.random_null_ci95[0]:.4f}, {cmp_.random_null_ci95[1]:.4f}]")
    for name, v in sorted(cmp_.aurocs.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} AUROC {v:.4f}")

    best = max(cmp_.aurocs, key=cmp_.aurocs.get)
    worst = min(cmp_.aurocs, key=cmp_.aurocs.get)
    spread = cmp_.aurocs[best] - cmp_.aurocs[worst]
    print(f"\nE5 rule is AUROC >= 0.75 AND >= 0.05 above worst. "
          f"best={best} {cmp_.aurocs[best]:.4f}, spread {spread:.4f} "
          f"-> NOT ADJUDICATED: this is FP32, E5 attaches to the deployed precision.")

    sweeps = {}
    for sig in ("margin", "max_softmax", "neg_entropy"):
        sw = sweep_thresholds(d.logits, d.labels, signal=sig, split="dev_2000")
        pts = []
        for t in a.targets:
            th = sw.threshold_for_escalation(t)
            i = int(np.argmin(np.abs(sw.thresholds - th)))
            pts.append({"target_escalation": t, "threshold": float(th),
                        "escalation_rate": float(sw.escalation_rate[i]),
                        "retained_accuracy": float(sw.retained_accuracy[i])})
        sweeps[sig] = pts
        print(f"\n  {sig}: escalation -> retained tier-0 accuracy")
        for p in pts:
            print(f"    esc {p['escalation_rate']:.3f} @ th {p['threshold']:+.4f} "
                  f"-> retained acc {p['retained_accuracy']:.4f}")

    out = {"experiment": "E5 (exploratory)", "status": "NOT the router result",
           "precision": "fp32", "precision_label": LABEL,
           "blocked_by": "E3 falsified (3ah); INT8 dev logits at chance",
           "seed": a.seed, "npz": a.npz.name, "split": "dev_2000",
           "n": cmp_.n, "n_tier0_correct": cmp_.n_correct,
           "dev_absent_classes": d.absent_classes.tolist(),
           "aurocs": cmp_.aurocs, "random_null": cmp_.random_null,
           "random_null_ci95": list(cmp_.random_null_ci95), "sweeps": sweeps}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}  [{LABEL}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
