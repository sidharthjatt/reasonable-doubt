#!/usr/bin/env python3
"""3au — does PERCENTILE calibration stabilise the escalation SET, or only the RATE?

Registered in PREREGISTRATION 3au BEFORE this ran, with the band mechanical:
  f = (J_pct - 0.2370) / (1 - 0.2370);  >=0.50 STRONG, 0.20-0.50 PARTIAL, <0.20 WEAK.

3an's Jaccard of 0.213-0.258 was measured under ABSOLUTE thresholds and confounds threshold
SCALE (which percentile calibration removes by construction) with margin RANKING (which it
does not). Selecting the top 4.056% by margin per seed removes the scale term, so whatever
non-overlap survives is ranking disagreement.

Costs $0.00.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.bootstrap import fast_macro_f1, paired_bootstrap  # noqa: E402
from src.router.signals import CLOSED_FORM  # noqa: E402

SEEDS, SIGNAL, MODEL = (1, 2, 3), "margin", "claude-sonnet-5"
J_ABS, TARGET = 0.2370, 0.040555555555555556
J_ABS_PAIRS = {"1&2": 0.240, "1&3": 0.258, "2&3": 0.213}   # 3an, absolute thresholds
J_CHANCE = 0.0208   # two 122-row sets drawn at random from 3,000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-json", type=Path, default=Path("results/e6_percentile.json"))
    a = ap.parse_args()

    names = json.loads(Path("configs/labels.json").read_text())["labels"]
    nid = {n: i for i, n in enumerate(names)}
    n_lab = len(names)

    from src.data.schema import ParseFailure, parse_response
    rows = json.loads((a.results_dir / "stage1_results.json").read_text())["models"][MODEL]["rows"]
    api_pred = {}
    for k, b in rows.items():
        try:
            api_pred[int(k)] = parse_response(b.get("text", "")).label
        except ParseFailure:
            api_pred[int(k)] = None

    per, sets = {}, {}
    for s in SEEDS:
        npz = np.load(a.results_dir / f"test_logits_int8_local_ce_seed{s}.npz")
        lg, y, ix = npz["test_3000_logits"], npz["test_3000_labels"].astype(int), npz["test_3000_indices"]
        sig = CLOSED_FORM[SIGNAL](lg)
        # RANK-based: the k lowest-margin rows. Scale is removed by construction; no
        # dev threshold is read, so hard rule 1 is not engaged for this measurement.
        k = int(round(TARGET * len(sig)))
        esc = np.zeros(len(sig), bool)
        esc[np.argsort(sig, kind="stable")[:k]] = True
        t0 = lg.argmax(-1).astype(int)
        api = np.array([nid.get(api_pred[int(i)] or "", -1) for i in ix])
        casc = np.where(esc, api, t0)
        sets[s] = set(int(i) for i in ix[esc])
        per[s] = {"k": int(esc.sum()), "rate": float(esc.mean()),
                  "tier0": fast_macro_f1(y, t0, n_lab),
                  "cascade": fast_macro_f1(y, casc, n_lab)}
        per[s]["delta"] = per[s]["cascade"] - per[s]["tier0"]

    print("=" * 74)
    print(f"3au — PERCENTILE selection: top {TARGET:.3%} by {SIGNAL} per seed")
    print("=" * 74)
    print(f"\n  {'seed':<6}{'k':>6}{'rate':>9}{'tier0':>10}{'cascade':>10}{'delta':>10}")
    for s in SEEDS:
        p_ = per[s]
        print(f"  {s:<6}{p_['k']:>6}{p_['rate']:>9.3%}{p_['tier0']:>10.4f}"
              f"{p_['cascade']:>10.4f}{p_['delta']:>+10.4f}")
    d = np.array([per[s]["delta"] for s in SEEDS])
    n_pos = int((d > 0).sum())
    print(f"  mean delta {d.mean():+.4f}, sd {d.std(ddof=1):.4f}, "
          f"sign-consistency {n_pos}/{len(SEEDS)}")

    jac = {f"{x}&{y}": len(sets[x] & sets[y]) / len(sets[x] | sets[y])
           for x, y in ((1, 2), (1, 3), (2, 3))}
    j_mean = float(np.mean(list(jac.values())))
    union = set().union(*sets.values())
    mult = Counter()
    for s in SEEDS:
        for i in sets[s]:
            mult[i] += 1
    by_k = Counter(mult.values())
    f = (j_mean - J_ABS) / (1 - J_ABS)

    # PER-PAIR, not just the mean. The mean is stable while the pairs are not, and an
    # aggregate that hides opposite-sign movement is the defect this project keeps
    # rediscovering (3an; 3ap D). j_chance is PRINTED, not left in the JSON: a floor that
    # exists only in an artefact is 3e instance 9's shape.
    print(f"\n  {'pair':<8}{'absolute':>10}{'percentile':>12}{'move':>9}")
    for k_, v in jac.items():
        print(f"  {k_:<8}{J_ABS_PAIRS[k_]:>10.3f}{v:>12.3f}{v - J_ABS_PAIRS[k_]:>+9.3f}")
    print(f"  {'mean':<8}{J_ABS:>10.4f}{j_mean:>12.4f}{j_mean - J_ABS:>+9.4f}")
    print(f"  {'chance':<8}{J_CHANCE:>10.4f}{J_CHANCE:>12.4f}"
          f"{'':>9}   <- floor for two {per[1]['k']}-row sets from 3000")
    print(f"\n  observed / chance = {j_mean / J_CHANCE:.1f}x  "
          f"-> agreement is FAR above chance, and FAR below 1.0")
    print(f"  distinct rows {len(union)}  |  by 1 seed: {by_k[1]}, 2: {by_k[2]}, "
          f"3: {by_k[3]} ({by_k[3]/len(union):.1%})")
    print(f"  f = ({j_mean:.4f} - {J_ABS:.4f}) / (1 - {J_ABS:.4f}) = {f:+.4f}")

    if j_mean >= 0.6185:
        disp = ("STRONG SUPPORT — percentile calibration stabilises the escalation SET; "
                "3aq stands as worded (rate AND set).")
    elif j_mean >= 0.3896:
        disp = ("PARTIAL — set stability improves but is not achieved; 3aq must be "
                "reworded to rate-stabilisation with set improvement quantified.")
    else:
        disp = ("WEAK / NULL — non-overlap is dominated by RANKING disagreement, which "
                "percentile calibration does not touch. 3aq must claim ONLY that it "
                "stabilises the escalation RATE.")
    print(f"\n  3au DISPOSITION: {disp}")

    a.out_json.write_text(json.dumps({
        "experiment": "3au", "cost_usd": 0.0, "selection": "rank-based top 4.056%",
        "per_seed": per, "pairwise_jaccard": jac, "j_pct_mean": j_mean,
        "j_abs_baseline": J_ABS, "j_chance": J_CHANCE, "j_abs_pairs": J_ABS_PAIRS,
        "observed_over_chance": j_mean / J_CHANCE,
        "per_pair_move": {k_: v - J_ABS_PAIRS[k_] for k_, v in jac.items()}, "fraction_closed": f,
        "distinct_rows": len(union),
        "by_n_seeds": {str(k_): v for k_, v in sorted(by_k.items())},
        "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
        "sign_consistency": f"{n_pos}/{len(SEEDS)}",
        "disposition": disp,
    }, indent=2))
    print(f"\n  wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
