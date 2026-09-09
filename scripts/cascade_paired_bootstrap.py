#!/usr/bin/env python3
"""Is the cascade's gain over Tier 0 alone real? PAIRED bootstrap over test rows.

Corrects a stated error. The E6 frontier's best Pareto point gains +0.0063 macro-F1 over
Tier 0 alone, and that was called "inside one seed sd (0.0034-0.0056)". It is not inside:
it is 1.1-1.9x that sd. But the sd was the wrong yardstick anyway. Tier 0 alone and Tier 0
plus escalation are evaluated on THE SAME ROWS with THE SAME SEEDS, differing only in
whether a row was escalated. The across-seed sd measures how much RETRAINING moves the
score; the question here is how much ESCALATION moves it on THESE clauses. The right test
is a paired bootstrap over rows, and a conservative wrong test is still a wrong test.

Hard rule 1 is preserved: thresholds come from dev_2000 and are held FIXED while test rows
are resampled. Resampling test rows must not be allowed to move a dev-calibrated quantity,
or the "calibrated on dev, applied to test" separation would leak.

Costs $0.00 — every input is on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.bootstrap import fast_macro_f1, paired_bootstrap  # noqa: E402
from src.eval.metrics import score  # noqa: E402
from src.router.load_logits import load_split  # noqa: E402
from src.router.signals import CLOSED_FORM  # noqa: E402

SEEDS = (1, 2, 3)
SIGNAL = "margin"
BOOTSTRAP_SEED = 20260909
MODEL = "claude-sonnet-5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("results/e6_cascade_paired.json"))
    ap.add_argument("--n-resamples", type=int, default=10_000)
    a = ap.parse_args()

    frontier = json.loads((a.results_dir / "e6_frontier.json").read_text())
    pareto = frontier["pareto"]["sunk"]
    best = max(pareto, key=lambda p: p["macro_f1"])
    target_rate = best["escalation_rate"]
    print(f"  best Pareto point: escalation {target_rate:.4f}, "
          f"macro-F1 {best['macro_f1']:.4f}")

    names: list[str] = json.loads(Path("configs/labels.json").read_text())["labels"]
    name_to_id = {n: i for i, n in enumerate(names)}

    from src.data.schema import ParseFailure, parse_response
    rows = json.loads((a.results_dir / "stage1_results.json").read_text())
    rows = rows["models"][MODEL]["rows"]
    api_pred: dict[int, str | None] = {}
    for k, body in rows.items():
        try:
            api_pred[int(k)] = parse_response(body.get("text", "")).label
        except ParseFailure:
            api_pred[int(k)] = None

    gold_ref: np.ndarray | None = None
    t0_c: dict[int, np.ndarray] = {}
    casc_c: dict[int, np.ndarray] = {}
    esc_rates = []

    for s in SEEDS:
        dev = load_split(a.results_dir / f"dev_logits_int8_local_ce_seed{s}.npz",
                         "dev_2000", for_calibration=True)
        npz = np.load(a.results_dir / f"test_logits_int8_local_ce_seed{s}.npz")
        logits, labels_i, index = (npz["test_3000_logits"], npz["test_3000_labels"],
                                   npz["test_3000_indices"])
        missing = [int(i) for i in index if int(i) not in api_pred]
        if missing:
            raise SystemExit(f"{len(missing)} test rows have no {MODEL} result "
                             f"(first {missing[:5]}). Refusing a partial join.")

        # CALIBRATE ON DEV -> a single fixed threshold, then APPLY to test. Held fixed
        # across every resample below.
        dev_sig = CLOSED_FORM[SIGNAL](dev.logits)
        thr = float(np.quantile(dev_sig, target_rate))
        test_sig = CLOSED_FORM[SIGNAL](logits)
        escalate = test_sig <= thr
        esc_rates.append(float(escalate.mean()))

        g = labels_i.astype(int)
        if gold_ref is None:
            gold_ref = g
        elif not np.array_equal(gold_ref, g):
            raise SystemExit(f"seed {s} disagrees on gold labels; artefacts inconsistent")

        t0 = logits.argmax(-1).astype(int)
        api = np.array([name_to_id.get(api_pred[int(i)] or "", -1) for i in index])
        t0_c[s] = t0
        casc_c[s] = np.where(escalate, api, t0)

    assert gold_ref is not None
    n_lab = len(names)
    print(f"  applied escalation rate per seed: "
          f"{', '.join(f'{r:.4f}' for r in esc_rates)}")

    # Reference scores through the project's scorer; averaging over ALL 100 classes,
    # which test_3000 covers (classes_represented: 100).
    ref = {}
    for tag, arms in (("tier0", t0_c), ("cascade", casc_c)):
        per = {}
        for s in SEEDS:
            preds = [names[int(j)] if j >= 0 else None for j in arms[s]]
            r = score([names[int(g)] for g in gold_ref], preds, labels=names)
            per[s] = r.macro_f1
            fast = fast_macro_f1(gold_ref, arms[s], n_lab)
            if abs(fast - r.macro_f1) > 1e-12:
                raise SystemExit(f"fast scorer disagrees on {tag} seed {s}: "
                                 f"{fast!r} vs {r.macro_f1!r}")
        ref[tag] = per
        m, sd = float(np.mean(list(per.values()))), float(np.std(list(per.values()), ddof=1))
        print(f"  {tag:<8}: macro-F1 {m:.4f} +/- {sd:.4f} "
              f"[{', '.join(f'{per[s]:.4f}' for s in SEEDS)}]")

    across_seed_sd = float(np.std(list(ref["tier0"].values()), ddof=1))
    delta = (float(np.mean(list(ref["cascade"].values())))
             - float(np.mean(list(ref["tier0"].values()))))

    bs = paired_bootstrap(
        len(gold_ref),
        lambda i: float(np.mean([fast_macro_f1(gold_ref[i], t0_c[s][i], n_lab)
                                 for s in SEEDS])),
        lambda i: float(np.mean([fast_macro_f1(gold_ref[i], casc_c[s][i], n_lab)
                                 for s in SEEDS])),
        n_resamples=a.n_resamples, seed=BOOTSTRAP_SEED)

    print("\n  PAIRED BOOTSTRAP — cascade minus Tier 0 alone, same rows, same seeds")
    print(bs.render("cascade - Tier 0"))
    print(f"\n  For contrast, the yardstick that was WRONG to use:")
    print(f"    across-seed sd of Tier 0        : {across_seed_sd:.4f} "
          f"(= {abs(delta) / across_seed_sd:.1f}x the delta — NOT 'inside one sd')")
    print(f"    it measures retraining noise, not the effect of escalating these rows")

    verdict = ("the cascade's gain over Tier 0 alone is NOT distinguishable from zero"
               if not bs.excludes_zero else
               "the cascade's gain over Tier 0 alone is distinguishable from zero")
    print(f"\n  VERDICT: {verdict} on the correct (paired) test.")

    a.out_json.write_text(json.dumps({
        "experiment": "E6 best-Pareto-point paired test",
        "cost_usd": 0.0,
        "escalation_rate_target": target_rate,
        "escalation_rate_applied_per_seed": esc_rates,
        "signal": SIGNAL,
        "calibrated_on": "dev_2000",
        "applied_to": "test_3000",
        "threshold_held_fixed_under_resampling": True,
        "macro_f1_per_seed": ref,
        "delta": delta,
        "across_seed_sd_tier0": across_seed_sd,
        "across_seed_sd_is_the_wrong_yardstick": (
            "paired comparison: same rows, same seeds, differing only in escalation"),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap": bs.__dict__,
        "verdict": verdict,
    }, indent=2))
    print(f"\n  wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
