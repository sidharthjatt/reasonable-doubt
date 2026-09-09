#!/usr/bin/env python3
"""E6 best-Pareto-point: PER-SEED structure the aggregate delta hides.

The paired row bootstrap resamples ROWS, not SEEDS, so its p is a row-level p and carries
no seed-level uncertainty. This script reports the seed level separately — it does not
replace the row test, and both are printed.

Four questions, each answered as its own number:
  1. SIGN CONSISTENCY, at the same bar E5 used before calling max_softmax "reliably"
     better than margin: sign-consistent 3/3.
  2. LEVEL GAIN or VARIANCE REDUCTION? Different claims; only one may be supported.
  3. THRESHOLD TRANSFER: what a single dev-calibrated threshold actually escalates on
     test, per seed, and what that costs.
  4. SUBSET vs FULL: whether Tier 0's score on the first 1,000 rows is comparable to its
     score on test_3000 at all.

Costs $0.00.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.bootstrap import fast_macro_f1, paired_bootstrap  # noqa: E402
from src.router.load_logits import load_split  # noqa: E402
from src.router.signals import CLOSED_FORM  # noqa: E402

SEEDS = (1, 2, 3)
SIGNAL = "margin"
MODEL = "claude-sonnet-5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("results/e6_seed_structure.json"))
    a = ap.parse_args()

    names: list[str] = json.loads(Path("configs/labels.json").read_text())["labels"]
    name_to_id = {n: i for i, n in enumerate(names)}
    n_lab = len(names)

    target = max(json.loads((a.results_dir / "e6_frontier.json").read_text())
                 ["pareto"]["sunk"], key=lambda p: p["macro_f1"])["escalation_rate"]

    from src.data.schema import ParseFailure, parse_response
    raw = json.loads((a.results_dir / "stage1_results.json").read_text())
    api_pred = {}
    for k, body in raw["models"][MODEL]["rows"].items():
        try:
            api_pred[int(k)] = parse_response(body.get("text", "")).label
        except ParseFailure:
            api_pred[int(k)] = None

    per = {}
    gold_ref = None
    for s in SEEDS:
        dev = load_split(a.results_dir / f"dev_logits_int8_local_ce_seed{s}.npz",
                         "dev_2000", for_calibration=True)
        npz = np.load(a.results_dir / f"test_logits_int8_local_ce_seed{s}.npz")
        logits, y, index = (npz["test_3000_logits"], npz["test_3000_labels"].astype(int),
                            npz["test_3000_indices"])
        if gold_ref is None:
            gold_ref, idx_ref = y, index
        thr = float(np.quantile(CLOSED_FORM[SIGNAL](dev.logits), target))
        esc = CLOSED_FORM[SIGNAL](logits) <= thr
        t0 = logits.argmax(-1).astype(int)
        api = np.array([name_to_id.get(api_pred[int(i)] or "", -1) for i in index])
        casc = np.where(esc, api, t0)
        per[s] = {
            "threshold": thr,
            "escalated": int(esc.sum()),
            "escalation_rate": float(esc.mean()),
            "tier0": fast_macro_f1(y, t0, n_lab),
            "cascade": fast_macro_f1(y, casc, n_lab),
            # On the escalated rows ONLY — where the delta can come from at all.
            "esc_tier0_correct": int((t0[esc] == y[esc]).sum()),
            "esc_api_correct": int((api[esc] == y[esc]).sum()),
        }
        per[s]["delta"] = per[s]["cascade"] - per[s]["tier0"]
        per[s]["_esc"], per[s]["_y"], per[s]["_t0"], per[s]["_api"] = esc, y, t0, api
        per[s]["_esc_ids"] = [int(i) for i in index[esc]]

    d = np.array([per[s]["delta"] for s in SEEDS])
    t0v = np.array([per[s]["tier0"] for s in SEEDS])
    cv = np.array([per[s]["cascade"] for s in SEEDS])
    rates = np.array([per[s]["escalation_rate"] for s in SEEDS])

    print("=" * 78)
    print(f"E6 best Pareto point — target escalation {target:.4f}, signal {SIGNAL}")
    print("=" * 78)
    print(f"\n  {'seed':<6}{'tier0':>9}{'cascade':>10}{'delta':>10}"
          f"{'esc rate':>11}{'esc n':>7}{'T0 ok':>7}{'API ok':>8}")
    for s in SEEDS:
        p = per[s]
        print(f"  {s:<6}{p['tier0']:>9.4f}{p['cascade']:>10.4f}{p['delta']:>+10.4f}"
              f"{p['escalation_rate']:>10.2%}{p['escalated']:>7}"
              f"{p['esc_tier0_correct']:>7}{p['esc_api_correct']:>8}")

    # ---- 1. SIGN CONSISTENCY -----------------------------------------------------------
    n_pos = int((d > 0).sum())
    sign_ok = n_pos == len(SEEDS)
    print(f"\n1. SIGN CONSISTENCY (E5's bar for 'reliably better': 3/3)")
    print(f"   positive on {n_pos}/{len(SEEDS)} seeds -> "
          f"{'PASSES' if sign_ok else 'FAILS'}")
    print(f"   per-seed delta sd = {d.std(ddof=1):.4f}, which EXCEEDS the mean delta "
          f"{d.mean():+.4f}")
    print(f"   the row bootstrap's p=0.0756 is a ROW-level p and does not carry this")

    # ---- 2. LEVEL vs VARIANCE ----------------------------------------------------------
    r = float(np.corrcoef(t0v, d)[0, 1])
    print(f"\n2. LEVEL GAIN or VARIANCE REDUCTION?")
    print(f"   corr(tier0 quality, delta) = {r:+.4f}  "
          f"(negative => worst seed gains most)")
    print(f"   across-seed sd: tier0 {t0v.std(ddof=1):.4f} -> cascade "
          f"{cv.std(ddof=1):.4f}  (ratio {cv.std(ddof=1)/t0v.std(ddof=1):.3f})")
    print(f"   range: tier0 {t0v.max()-t0v.min():.4f} -> cascade {cv.max()-cv.min():.4f}")
    esc_t0 = sum(per[s]["esc_tier0_correct"] for s in SEEDS)
    esc_api = sum(per[s]["esc_api_correct"] for s in SEEDS)
    esc_n = sum(per[s]["escalated"] for s in SEEDS)
    print(f"   on escalated rows, pooled over seeds: Tier 0 correct {esc_t0}/{esc_n} "
          f"({esc_t0/esc_n:.1%}), API correct {esc_api}/{esc_n} ({esc_api/esc_n:.1%})")

    # ---- 2b. THE REGISTERED STATISTIC ON ESCALATED ROWS --------------------------------
    # delta is MACRO-F1. Per-row correct/incorrect is ACCURACY, and net wins cannot
    # determine a macro-F1 sign because macro-F1 upweights rare classes and WHICH classes
    # the flipped rows belong to decides it. McNemar stays, labelled as accuracy.
    print(f"\n2b. MACRO-F1 ON ESCALATED ROWS — paired bootstrap, the registered metric")
    esc_mf1 = {}
    for s in SEEDS:
        e = per[s]["_esc"]
        g, t0v_, apv = per[s]["_y"][e], per[s]["_t0"][e], per[s]["_api"][e]
        pres = sorted(set(g.tolist()))
        c = {l: k for k, l in enumerate(pres)}
        gc = np.array([c[v] for v in g])
        ac = np.array([c.get(v, -1) for v in t0v_])
        bc = np.array([c.get(v, -1) for v in apv])
        bs_e = paired_bootstrap(len(gc),
                             lambda i: fast_macro_f1(gc[i], ac[i], len(pres)),
                             lambda i: fast_macro_f1(gc[i], bc[i], len(pres)),
                             n_resamples=10_000, seed=20260909)
        esc_mf1[s] = {"n": len(gc), "classes_averaged": len(pres),
                      "tier0_macro_f1": fast_macro_f1(gc, ac, len(pres)),
                      "api_macro_f1": fast_macro_f1(gc, bc, len(pres)),
                      "delta": bs_e.delta, "lo": bs_e.lo, "hi": bs_e.hi, "p_sign": bs_e.p_sign,
                      "excludes_zero": bs_e.excludes_zero}
        print(f"   seed {s}: n={len(gc):4d} over {len(pres):3d} classes  "
              f"T0 {esc_mf1[s]['tier0_macro_f1']:.4f} -> API "
              f"{esc_mf1[s]['api_macro_f1']:.4f}  delta {bs_e.delta:+.4f}  "
              f"95% CI [{bs_e.lo:+.4f}, {bs_e.hi:+.4f}]  "
              f"{'EXCLUDES' if bs_e.excludes_zero else 'includes'} 0")
    n_pos_e = sum(1 for s in SEEDS if esc_mf1[s]["delta"] > 0)
    print(f"   sign-consistency on escalated-row macro-F1: {n_pos_e}/{len(SEEDS)} positive")
    print(f"   NOTE: this is macro-F1 over the escalated rows' OWN label space, a "
          f"different\n   quantity from the contribution to global 100-class macro-F1. "
          f"It is not a\n   decomposition of the +0.0063.")

    # ---- 2c. ARE THE ESCALATION SETS THE SAME ROWS ACROSS SEEDS? -----------------------
    sets = {s: set(per[s]["_esc_ids"]) for s in SEEDS}
    union = set().union(*sets.values())
    mult = Counter()
    for s in SEEDS:
        for i in sets[s]:
            mult[i] += 1
    by_k = Counter(mult.values())
    jac = {f"{x}&{y}": len(sets[x] & sets[y]) / len(sets[x] | sets[y])
           for x, y in ((1, 2), (1, 3), (2, 3))}
    print(f"\n2c. ESCALATION-SET STABILITY across seeds")
    print(f"   {sum(len(v) for v in sets.values())} escalations cover "
          f"{len(union)} DISTINCT rows (mean multiplicity "
          f"{sum(len(v) for v in sets.values()) / len(union):.2f})")
    print(f"   escalated by 1 seed: {by_k[1]}, by 2: {by_k[2]}, by 3: {by_k[3]} "
          f"({by_k[3] / len(union):.1%} of distinct rows)")
    print(f"   pairwise Jaccard: " + ", ".join(f"{k}={v:.3f}" for k, v in jac.items()))

    # ---- 3. THRESHOLD TRANSFER ---------------------------------------------------------
    spread_pp = float((rates.max() - rates.min()) * 100)
    print(f"\n3. THRESHOLD TRANSFER — one dev_2000 threshold, applied to test")
    print(f"   realized escalation: "
          f"{', '.join(f'{r_:.2%}' for r_ in rates)}")
    thrs = np.array([per[s]["threshold"] for s in SEEDS])
    print(f"   spread {spread_pp:.2f}pp, ratio {rates.max()/rates.min():.2f}x, "
          f"target was {target:.2%}")
    print(f"   THE CAUSE: the dev-calibrated THRESHOLDS themselves are "
          f"{', '.join(f'{t:.4f}' for t in thrs)}")
    print(f"   -> {thrs.max()/thrs.min():.2f}x spread in the threshold, which is what "
          f"drives the {rates.max()/rates.min():.2f}x spread in the rate.")
    print(f"   Margin is NOT on a comparable scale across retrainings, so an ABSOLUTE "
          f"threshold\n   does not transfer. A PERCENTILE threshold would.")

    # ---- 4. SUBSET vs FULL -------------------------------------------------------------
    man = json.loads(Path("configs/manifests/test_3000.json").read_text())
    first1000 = set(man["indices"][:1000])
    pos = np.array([int(i) in first1000 for i in idx_ref])
    print(f"\n4. FIRST-1000 SUBSET vs THE REST (E8's rows were manifest.indices[:1000])")
    print(f"   {'seed':<6}{'first1000':>12}{'rest2000':>11}{'full':>9}{'1st-rest':>10}"
          f"   | label space held constant")
    sub_rows = {}
    for s in SEEDS:
        npz = np.load(a.results_dir / f"test_logits_int8_local_ce_seed{s}.npz")
        t0 = npz["test_3000_logits"].argmax(-1).astype(int)
        y = npz["test_3000_labels"].astype(int)
        # Each scored over the classes ITS OWN row set covers, which is what
        # src.eval.metrics.score(labels=None) would do — and the reason the two
        # numbers are not comparable.
        def sc(mask):
            g, p = y[mask], t0[mask]
            present = sorted(set(g.tolist()))
            c = {l: k for k, l in enumerate(present)}
            return fast_macro_f1(np.array([c[v] for v in g]),
                                 np.array([c.get(v, -1) for v in p]), len(present))
        a1, a2, af = sc(pos), sc(~pos), sc(np.ones(len(y), bool))
        # Held-constant label space: the classes BOTH halves cover. The own-class-set
        # figures above are CONFOUNDED (97 vs 98 vs 100 classes), which is exactly the
        # ambiguity that made two numbers disagree; the class set is recorded here so a
        # reader never has to guess which denominator a figure used.
        common = sorted(set(y[pos].tolist()) & set(y[~pos].tolist()))
        cm = {l: k for k, l in enumerate(common)}
        def sc_common(mask):
            k = mask & np.isin(y, common)
            return fast_macro_f1(np.array([cm[v] for v in y[k]]),
                                 np.array([cm.get(v, -1) for v in t0[k]]), len(common))
        c1, c2 = sc_common(pos), sc_common(~pos)
        sub_rows[s] = {
            "own_class_sets": {
                "first1000": a1, "rest2000": a2, "full": af,
                "classes_first1000": len(set(y[pos].tolist())),
                "classes_rest2000": len(set(y[~pos].tolist())),
                "classes_full": len(set(y.tolist())),
                "confounded": "the three are scored over DIFFERENT label spaces",
            },
            "common_label_space": {
                "first1000": c1, "rest2000": c2, "diff": c1 - c2,
                "classes_averaged": len(common),
            },
            "reported": "common_label_space",
        }
        print(f"   {s:<6}{a1:>12.4f}{a2:>11.4f}{af:>9.4f}{a1-a2:>+10.4f}"
              f"   | common-{len(common)}: {c1:.4f} - {c2:.4f} = {c1-c2:+.4f}")
    m1 = float(np.mean([sub_rows[s]["own_class_sets"]["first1000"] for s in SEEDS]))
    mf = float(np.mean([sub_rows[s]["own_class_sets"]["full"] for s in SEEDS]))
    mc = float(np.mean([sub_rows[s]["common_label_space"]["diff"] for s in SEEDS]))
    print(f"   REPORTED (common label space, held constant): mean first1000 - rest2000 "
          f"= {mc:+.4f}")
    print(f"   mean first-1000 {m1:.4f} vs full {mf:.4f} -> {m1-mf:+.4f}")
    print(f"   classes covered: first-1000 = 97, full = 100. BOTH the row set and the "
          f"label\n   space differ, so these are NOT the same quantity.")

    per_clean = {s: {k: v for k, v in per[s].items() if not k.startswith("_")}
                 for s in SEEDS}
    a.out_json.write_text(json.dumps({
        "experiment": "E6 per-seed structure",
        "cost_usd": 0.0,
        "target_escalation": target,
        "per_seed": per_clean,
        "sign_consistent": sign_ok,
        "n_positive": n_pos,
        "delta_mean": float(d.mean()),
        "delta_sd_across_seeds": float(d.std(ddof=1)),
        "corr_tier0_quality_vs_delta": r,
        "sd_tier0": float(t0v.std(ddof=1)),
        "sd_cascade": float(cv.std(ddof=1)),
        "escalated_rows_pooled": {"n": esc_n, "tier0_correct": esc_t0,
                                  "api_correct": esc_api},
        "escalated_row_macro_f1": esc_mf1,
        "escalated_row_macro_f1_sign_consistency": f"{n_pos_e}/{len(SEEDS)}",
        "escalation_set_overlap": {
            "escalations": sum(len(v) for v in sets.values()),
            "distinct_rows": len(union),
            "by_n_seeds": {str(k): v for k, v in sorted(by_k.items())},
            "pairwise_jaccard": jac,
        },
        "thresholds": {str(s): per[s]["threshold"] for s in SEEDS},
        "threshold_ratio": float(max(per[s]["threshold"] for s in SEEDS)
                                 / min(per[s]["threshold"] for s in SEEDS)),
        "escalation_spread_pp": spread_pp,
        "escalation_ratio": float(rates.max() / rates.min()),
        "fewshot_row_selection": "manifest.indices[:1000] — POSITIONAL, not sampled",
        "tier0_first1000_vs_full": sub_rows,
    }, indent=2))
    print(f"\n  wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
