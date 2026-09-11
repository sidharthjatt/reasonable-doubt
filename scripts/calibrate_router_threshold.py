"""Produce the DEPLOYED router threshold from dev_2000 logits (hard rule 1).

    python scripts/calibrate_router_threshold.py --seed 1 --target-escalation 0.040556

PERCENTILE, NOT ABSOLUTE — §3aq's one actionable engineering result.
The per-seed dev-calibrated margin thresholds across E6's three seeds are 0.1154 /
0.1177 / 0.1718, a 1.49x spread, and that spread is the CAUSE of the 1.53x spread in
realized escalation rate. The margin signal is not on a comparable scale across
retrainings, so an absolute threshold does not transfer between seeds of the same model
on the same data. A percentile is invariant to the monotone rescaling retraining applies,
so it holds the escalation rate — and therefore the API bill — fixed.

WHAT PERCENTILE CALIBRATION DOES AND DOES NOT BUY (§3au, which TESTED it).
It stabilises the escalation RATE ONLY. Mean escalation-set Jaccard moves 0.2370 ->
0.2419, f = +0.0065, i.e. 0.65% of the achievable improvement. WHICH clauses get
escalated is NOT stabilised, so no SLA, audit or reproducibility claim may rest on set
stability. Cost predictability is the whole of the benefit.

HOW THE PERCENTILE REACHES A PER-REQUEST DECISION.
A percentile cannot be evaluated against a single clause — it is a property of a
distribution. So the percentile is applied HERE, offline, to the served artefact's own
margin distribution on dev_2000, and the resulting absolute value is what the service
compares each request against. The percentile is the transferable quantity and is
recorded; the absolute value is model-specific, is re-derived whenever the weights
change, and is never computed from live traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.router.calibrate import assert_threshold_split  # noqa: E402
from src.router.signals import CLOSED_FORM  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[1] / "configs" / "router_threshold.json"

# The deployed precision and ISA. Recorded in the npz by scripts/dev_logits_int8_local.py
# and ASSERTED here rather than merely copied: an FP32 or Kaggle-x86 logits file would
# calibrate a threshold for a margin distribution the service never sees. The FP32 npz
# happens to carry no `provenance` key at all, so pointing this script at it would crash
# — but that is luck, not a guard, and luck is not a control.
REQUIRED_ISA = "arm64_local"

VERDICT_NOTE = (
    "THE ESCALATION RATE FIXES THE BILL. IT IS NOT AN ACCURACY CLAIM. §3av's registered "
    "verdict (E6-A) is that cascading adds NO MEASURABLE ACCURACY: test_3000 macro-F1 "
    "delta +0.0063, 95% CI [-0.0004, +0.0103], sign-consistency 2/3; under percentile "
    "selection +0.0065, 2/3. Routing works — the margin signal drops accuracy from ~0.85 "
    "to ~0.37 on its own selection (AUROC 0.86) — but the escalation target has no "
    "marginal value on precisely the rows routing correctly identifies as hard. This "
    "threshold therefore buys a bounded, predictable API bill and a defined "
    "worst-case handling path. Any accuracy improvement read off it is unsupported."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--precision", required=True, choices=("fp32", "int8"),
                    help="THE SERVED precision. The threshold must be calibrated on dev "
                         "logits of the precision that will actually answer requests: "
                         "FP32 and INT8 have different margin distributions, so an INT8 "
                         "threshold applied to FP32 gives an escalation rate nobody chose.")
    ap.add_argument("--signal", default="margin")
    ap.add_argument("--target-escalation", type=float, required=True,
                    help="fraction, e.g. 0.040556 for §3aq's 4.056%%")
    ap.add_argument("--logits", type=Path, default=None,
                    help="defaults to results/dev_logits_int8_local_ce_seed<N>.npz")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to configs/router_threshold_<precision>.json")
    args = ap.parse_args()
    if args.out is None:
        args.out = (Path(__file__).resolve().parents[1] / "configs"
                    / f"router_threshold_{args.precision}.json")

    if not 0.0 < args.target_escalation < 1.0:
        raise SystemExit(f"--target-escalation must be a fraction in (0,1), "
                         f"got {args.target_escalation}")

    default_npz = {"int8": f"results/dev_logits_int8_local_ce_seed{args.seed}.npz",
                   "fp32": f"results/dev_logits_fp32_local_ce_seed{args.seed}.npz"}
    npz_path = args.logits or Path(default_npz[args.precision])
    d = np.load(npz_path, allow_pickle=True)
    if "provenance" not in d.files:
        raise SystemExit(
            f"{npz_path} carries no provenance block. Refusing to calibrate against "
            f"logits whose precision and ISA are unknown.")
    prov = json.loads(str(d["provenance"]))

    split = prov["manifest"]
    assert_threshold_split(split)          # hard rule 1, enforced not remembered

    # Item 2's guard, made explicit rather than inherited from luck.
    if prov["precision"] != args.precision or prov["isa"] != REQUIRED_ISA:
        raise SystemExit(
            f"REFUSING: logits are precision={prov['precision']!r} isa={prov['isa']!r}, "
            f"but --precision says {args.precision!r} on {REQUIRED_ISA!r}. Calibrating "
            f"on the wrong precision, or on Kaggle's x86 logits, would fit a threshold "
            f"to a margin distribution the service never sees.")
    if prov["seed"] != args.seed:
        raise SystemExit(f"npz is seed {prov['seed']}, --seed says {args.seed}")

    logits, labels = d["dev_logits"], d["dev_labels"]
    score = CLOSED_FORM[args.signal](logits)

    # The percentile, evaluated EXACTLY rather than snapped to a sweep grid. The 201-point
    # grid in sweep_thresholds() has 0.5% spacing, which cannot represent 4.056%.
    threshold = float(np.quantile(score, args.target_escalation))
    escalate = score < threshold
    achieved = float(escalate.mean())
    correct = np.asarray(labels) == np.asarray(logits).argmax(-1)
    retained = float(correct[~escalate].mean())

    payload = {
        "signal": args.signal,
        "calibration_mode": "percentile",
        "threshold": threshold,
        "rule": "escalate when signal < threshold",
        "percentile": args.target_escalation * 100.0,
        "target_escalation_rate": args.target_escalation,
        "achieved_escalation_rate_on_dev": achieved,
        "retained_tier0_accuracy_on_dev": retained,
        "calibrated_on": split,
        "calibrated_on_manifest_sha256": prov["manifest_sha256"],
        "artefact": prov["artefact"],
        "seed": prov["seed"],
        "precision": prov["precision"],
        "isa": prov["isa"],
        "max_length": prov["max_length"],
        "n_dev_rows": int(len(labels)),
        "source_npz": str(npz_path),
        "percentile_basis": (
            "§3aq: absolute margin thresholds span 1.49x across E6's three seeds "
            "(0.1154 / 0.1177 / 0.1718), so an absolute value does not transfer between "
            "retrainings. A percentile is invariant to that rescaling and holds the "
            "escalation rate fixed. §3au NARROWS this: it stabilises the RATE ONLY "
            "(set Jaccard 0.2370 -> 0.2419), never WHICH rows are escalated."),
        "verdict_note": VERDICT_NOTE,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {args.out}")
    print(f"  mode=percentile signal={args.signal}")
    print(f"  percentile={payload['percentile']:.4f}%  ->  threshold={threshold:.6f}")
    print(f"  calibrated on {split} (seed {prov['seed']}, {prov['artefact']}, "
          f"{prov['precision']}/{prov['isa']})")
    print(f"  source npz: {npz_path}")
    print(f"  escalation on dev: target {args.target_escalation:.6f} -> "
          f"achieved {achieved:.6f}")
    print(f"  retained Tier 0 accuracy on dev: {retained:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
