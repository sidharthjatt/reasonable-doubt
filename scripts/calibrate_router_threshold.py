"""Produce the DEPLOYED router threshold from dev_2000 logits (hard rule 1).

    python scripts/calibrate_router_threshold.py --seed 1 --target-escalation 0.10

Writes configs/router_threshold.json. The service LOADS this file; it never computes a
threshold at request time. A threshold derived per-request would be derived from traffic,
and traffic is not the dev split — that is hard rule 1 violated by the back door, and it
would drift silently as traffic changed.

The file records the artefact it was calibrated against. `src/serve/config.py` refuses to
serve a Tier 0 model directory that does not match, because E1b may replace int8_ce_1
with 10-epoch weights whose margin distribution is NOT the one calibrated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.router.calibrate import assert_threshold_split, sweep_thresholds  # noqa: E402

OUT_PATH = Path(__file__).resolve().parents[1] / "configs" / "router_threshold.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--signal", default="margin")
    ap.add_argument("--target-escalation", type=float, required=True)
    ap.add_argument("--logits", type=Path, default=None,
                    help="defaults to results/dev_logits_int8_local_ce_seed<N>.npz")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    npz_path = args.logits or Path(
        f"results/dev_logits_int8_local_ce_seed{args.seed}.npz")
    d = np.load(npz_path, allow_pickle=True)
    prov = json.loads(str(d["provenance"]))

    split = prov["manifest"]
    assert_threshold_split(split)          # hard rule 1, enforced not remembered
    if prov["seed"] != args.seed:
        raise SystemExit(f"npz is seed {prov['seed']}, --seed says {args.seed}")

    logits, labels = d["dev_logits"], d["dev_labels"]
    sweep = sweep_thresholds(logits, labels, signal=args.signal, split=split)

    i = int(np.argmin(np.abs(sweep.escalation_rate - args.target_escalation)))
    threshold = float(sweep.thresholds[i])
    achieved = float(sweep.escalation_rate[i])
    retained = float(sweep.retained_accuracy[i])

    payload = {
        "signal": args.signal,
        "threshold": threshold,
        "rule": "escalate when signal < threshold",
        "calibrated_on": split,
        "calibrated_on_manifest_sha256": prov["manifest_sha256"],
        "artefact": prov["artefact"],
        "seed": prov["seed"],
        "precision": prov["precision"],
        "isa": prov["isa"],
        "max_length": prov["max_length"],
        "target_escalation_rate": args.target_escalation,
        "achieved_escalation_rate_on_dev": achieved,
        "retained_tier0_accuracy_on_dev": retained,
        "n_dev_rows": int(len(labels)),
        "source_npz": str(npz_path),
        "operating_point_note": (
            "The escalation rate is a POLICY CHOICE, not an accuracy optimum. E6-A "
            "found the cascade is not established as better than Tier 0 alone "
            "(test_3000 macro-F1 delta +0.0063, 95% CI [-0.0004, +0.0103], "
            "sign-consistency 2/3). This threshold buys a defined worst-case handling "
            "path, not measured accuracy."),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {args.out}")
    print(f"  signal={args.signal} threshold={threshold:.6f}")
    print(f"  calibrated on {split} (seed {prov['seed']}, {prov['artefact']})")
    print(f"  escalation on dev: target {args.target_escalation:.3f} -> "
          f"achieved {achieved:.4f}")
    print(f"  retained Tier 0 accuracy on dev: {retained:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
