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


def read_provenance(npz_path: Path) -> dict:
    """Decide whether these logits may be calibrated on — from PROVENANCE, not the name.

    The original guard refused any filename containing "int8", written when every INT8
    number in the project came from Kaggle's non-VNNI x86 kernel and was at chance
    (§3ah). That is no longer the discriminating fact. INT8 re-measured on arm64 tracks
    FP32 to 0.0048 macro-F1 and is the DEPLOYED precision, so refusing it by name would
    now block the only calibration that matches deployment.

    The question became "was this measured on a kernel that works?", which a filename
    cannot answer and an embedded provenance record can. Files written before provenance
    existed carry none; those are treated as FP32 only if the name does not say int8, and
    refused otherwise — an unlabelled INT8 file is exactly the Kaggle output.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        raw = str(z["provenance"]) if "provenance" in z.files else None
    if raw:
        p = json.loads(raw)
        precision, isa = p.get("precision", "?"), p.get("isa", "?")
        if precision == "int8" and isa != "arm64_local":
            raise SystemExit(
                f"refusing {npz_path.name}: precision=int8 on isa={isa!r}. Only INT8 "
                f"measured on arm64 (the deployed ISA) may be calibrated on; the x86 "
                f"kernel scored test_3000 at chance (§3ah), so those logits are noise.")
        return {"precision": precision, "isa": isa, "provenance": p}
    if "int8" in npz_path.name.lower():
        raise SystemExit(
            f"refusing {npz_path.name}: the name says INT8 but the file carries NO "
            f"provenance, so the ISA it was measured on is unknown. The Kaggle dev run "
            f"wrote exactly such files from the broken x86 kernel (§3ah). Regenerate "
            f"with scripts/dev_logits_int8_local.py, which records provenance.")
    return {"precision": "fp32", "isa": "unrecorded_assumed_gpu_fp32", "provenance": None}


def label_for(prov: dict) -> str:
    if prov["precision"] == "int8":
        return "INT8-calibrated on arm64 — the DEPLOYED precision (E3 met, max |delta| 0.0083)"
    return LABEL


SEEDS = (1, 2, 3)


def aggregate(out_path: Path, prefix: str = "e5_sweep_fp32") -> int:
    """Mean +/- sd across seeds, and E5's rule adjudicated on the AGGREGATE.

    This lives in the script rather than in a person's terminal because hard rule 2
    makes the aggregate the reportable quantity and the per-seed numbers intermediate.
    Hand-aggregation is what left seeds 1 and 2 in scrollback.

    It REFUSES on fewer than 3 seeds rather than averaging what it finds: a mean over 2
    seeds is not the quantity hard rule 2 names, and silently reporting one would be the
    same defect as the seedless filename — a plausible number standing in for the
    registered one.
    """
    import statistics as stats

    per = {}
    for s in SEEDS:
        p = Path(f"results/{prefix}_seed{s}.json")
        if not p.exists():
            raise SystemExit(
                f"missing {p}. E5's aggregate needs all {len(SEEDS)} seeds — hard rule 2 "
                f"requires mean +/- std over >=3 seeds, and a mean over fewer is not that "
                f"quantity. Run the per-seed sweeps first.")
        per[s] = json.loads(p.read_text())
    precisions = {per[s].get("precision") for s in SEEDS}
    isas = {per[s].get("isa") for s in SEEDS}
    if len(precisions) != 1 or len(isas) != 1:
        raise SystemExit(
            f"seeds disagree on provenance: precision={precisions}, isa={isas}. "
            f"Averaging AUROCs across precisions would report a number measured on "
            f"neither system.")

    signals = sorted(per[SEEDS[0]]["aurocs"])
    ranks = {s: {sig: i + 1 for i, sig in
                 enumerate(sorted(signals, key=lambda x: -per[s]["aurocs"][x]))}
             for s in SEEDS}
    agg = {}
    for sig in signals:
        v = [per[s]["aurocs"][sig] for s in SEEDS]
        agg[sig] = {"mean": stats.mean(v), "sd": stats.stdev(v), "per_seed": v,
                    "per_seed_rank": [ranks[s][sig] for s in SEEDS]}

    means = {k: v["mean"] for k, v in agg.items()}
    best, worst = max(means, key=means.get), min(means, key=means.get)
    spread = means[best] - means[worst]
    # RANK STABILITY is reported because the means alone cannot show it: four signals
    # within 0.005 can still reorder between seeds, and a "best signal" that is not the
    # best on every seed is not a selection, it is a coin flip with error bars.
    stable = {sig: len(set(agg[sig]["per_seed_rank"])) == 1 for sig in signals}
    rank1_every_seed = [sig for sig in signals if agg[sig]["per_seed_rank"] == [1] * len(SEEDS)]

    prec, isa = precisions.pop(), isas.pop()
    print(f"\n=== E5 aggregate over {len(SEEDS)} seeds — precision={prec} isa={isa} ===")
    print(f"{'signal':22s} {'mean':>8s} {'sd':>8s}   ranks")
    for sig in sorted(signals, key=lambda x: -means[x]):
        print(f"  {sig:20s} {means[sig]:8.4f} {agg[sig]['sd']:8.4f}   "
              f"{agg[sig]['per_seed_rank']}{'' if stable[sig] else '  <- UNSTABLE'}")
    print(f"\nbest mean AUROC {means[best]:.4f} ({best}) vs required >= 0.75: "
          f"{'MET' if means[best] >= 0.75 else 'NOT MET'}")
    print(f"spread of means {spread:.4f} vs required >= 0.05: "
          f"{'MET' if spread >= 0.05 else 'NOT MET'}")
    print(f"holds rank 1 on every seed: {rank1_every_seed or 'NONE'}")

    sweeps = {}
    for sig in ("margin", "max_softmax", "neg_entropy"):
        by_target = {}
        for i, t in enumerate(per[SEEDS[0]]["sweeps"][sig]):
            v = [per[s]["sweeps"][sig][i]["retained_accuracy"] for s in SEEDS]
            e = [per[s]["sweeps"][sig][i]["escalation_rate"] for s in SEEDS]
            by_target[f"{t['target_escalation']:.2f}"] = {
                "escalation_mean": stats.mean(e),
                "retained_accuracy_mean": stats.mean(v),
                "retained_accuracy_sd": stats.stdev(v)}
        sweeps[sig] = by_target

    out = {"experiment": "E5 (exploratory, AGGREGATE)", "status": "NOT the router result",
           "precision": prec, "isa": isa,
           "n_seeds": len(SEEDS), "seeds": list(SEEDS),
           "aurocs": agg,
           "best_signal_by_mean": best, "spread_of_means": spread,
           "rule_auroc_ge_0.75": means[best] >= 0.75,
           "rule_spread_ge_0.05": spread >= 0.05,
           "rank_stable": stable, "holds_rank_1_every_seed": rank1_every_seed,
           "sweeps_mean_over_seeds": sweeps}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}  [precision={prec} isa={isa}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path,
                    help="dev_logits_<arm>_seed<n>.npz — the FP32 file, NOT *int8*")
    ap.add_argument("--seed", type=int)
    # PER-SEED FILENAME. The previous default was one seedless path, so running seeds
    # 1..3 in sequence overwrote it twice and left only the last on disk — seeds 1 and 2
    # survived nowhere but terminal scrollback. Same class as the custom_id collision
    # caught before Stage 1: a key that is not unique per unit of work, silently
    # discarding results instead of failing.
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/e5_sweep_fp32_seed<seed>.json")
    ap.add_argument("--aggregate", action="store_true",
                    help="read the per-seed files and write mean +/- sd across seeds")
    ap.add_argument("--prefix", default="e5_sweep_fp32",
                    help="per-seed filename stem for --aggregate (e.g. e5_sweep_int8)")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    a = ap.parse_args()

    if a.aggregate:
        return aggregate(a.out or Path(f"results/{a.prefix}_aggregate.json"), a.prefix)
    if a.npz is None or a.seed is None:
        raise SystemExit("--npz and --seed are required unless --aggregate is passed")
    out_path = a.out or Path(f"results/e5_sweep_fp32_seed{a.seed}.json")

    prov = read_provenance(a.npz)

    active_label = label_for(prov)
    d = load_split(a.npz, "dev_2000", for_calibration=True)   # loader vocabulary, not the npz prefix
    cmp_ = compare_signals(d.logits, d.labels, split="dev_2000")

    print(f"\n=== E5 signal comparison — {active_label} ===")
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

    out = {"experiment": "E5", "status": ("router result on the deployed precision"
             if prov["precision"] == "int8" else "NOT the router result"),
           "precision": prov["precision"], "isa": prov["isa"],
           "precision_label": active_label, "provenance": prov["provenance"],
           "blocked_by": (None if prov["precision"] == "int8"
                          else "reported on FP32; INT8 is the deployed precision"),
           "seed": a.seed, "npz": a.npz.name, "split": "dev_2000",
           "n": cmp_.n, "n_tier0_correct": cmp_.n_correct,
           "dev_absent_classes": d.absent_classes.tolist(),
           "aurocs": cmp_.aurocs, "random_null": cmp_.random_null,
           "random_null_ci95": list(cmp_.random_null_ci95), "sweeps": sweeps}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}  [{active_label}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
