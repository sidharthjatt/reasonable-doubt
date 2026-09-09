"""E6 — cost-vs-accuracy Pareto frontier for the 3-tier cascade. COST AXIS ONLY.

    python scripts/build_frontier.py

WHAT THIS FIGURE IS NOT. It is a cost-vs-ACCURACY frontier, not cost-vs-latency. Tier 1's
`per_tier_throughput` is unmeasured and Tier 2's is `not_applicable` (queue latency is not
hardware throughput), so no end-to-end latency or SLA claim can be made from it, and the
escalation rate at which Tier 1 would become a queueing bottleneck is unknown. That
narrowing is printed on the figure itself, not relegated to a footnote, because a
cost-accuracy curve is easy to read as a deployment recommendation.

WHY A SCRIPT AND NOT A PAGE. Tier 0's numbers move when E1b lands. This regenerates from
whatever is on disk; a published snapshot would go stale silently.

HARD RULE 1 IS STRUCTURAL HERE. Thresholds are calibrated on `dev_2000` INT8 logits and
then APPLIED to `test_3000`. The test split is never used to choose a threshold — the
calibration and application sets are loaded separately and the dev loader refuses
anything that is not a threshold split.

PRECISION. Tier 0 is INT8 measured on arm64 — the deployed precision and the deployed
ISA (§3ah). FP32 is not plotted: E3's falsification clause prohibits reporting the FP32
figure as the system's accuracy.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.eval.breakeven import breakeven, device_cost_usd, load_hardware  # noqa: E402
from src.eval.metrics import score  # noqa: E402
from src.eval.pareto import cascade_points, pareto_frontier  # noqa: E402
from src.router.load_logits import load_split  # noqa: E402
from src.router.signals import CLOSED_FORM  # noqa: E402

SEEDS = (1, 2, 3)
SIGNAL = "margin"          # E5's registered clause: spread < 0.05 => use margin (§3ai)
TARGETS = np.linspace(0.0, 1.0, 51)


def _row_index(key: str) -> int:
    """stage1_results.json keys rows by BARE row index; the batch client's custom_ids use
    `tag_manifest-index`. Accept both, and refuse anything else rather than guessing —
    a mis-parsed index silently joins the wrong clause to the wrong prediction."""
    from src.router.join import CUSTOM_ID
    m = CUSTOM_ID.match(key)
    if m:
        return int(m.group("index"))
    if key.isdigit():
        return int(key)
    raise ValueError(f"cannot parse a row index from key {key!r}")


def load_api_leg(path: Path, model: str,
                 labels_by_index: dict[int, str]) -> dict[int, str | None]:
    """Per-row correctness for one Claude leg, keyed by dataset row index.

    USES THE PROJECT'S CANONICAL PARSER (`src.data.schema.parse_response`), not a local
    `json.loads`. A naive parser here scored Haiku at 12% accuracy with 2,523/3,000
    "unparseable", because Haiku wraps its JSON in ```json fences — a well-formed answer
    in a code fence. That is §3e instance 2's exact shape (ad-hoc JSON extraction from
    model output) and it would have put a fabricated Tier 2 leg on the frontier.

    Non-strict parsing is correct here: recovering the answer from a fence is reading the
    output format, not repairing a wrong answer. LABEL matching stays EXACT per §3ag —
    an unrecognised label counts as wrong, because format_failure_rate is a measured
    quantity of this project and folding would delete the finding.
    """
    from src.data.schema import ParseFailure, parse_response

    d = json.loads(path.read_text())
    rows = d["models"][model]["rows"]
    out: dict[int, str | None] = {}
    n_fail = 0
    for k, body in rows.items():
        idx = _row_index(k)
        try:
            pred = parse_response(body.get("text", "")).label
        except ParseFailure:
            pred = None
            n_fail += 1
        out[idx] = pred
    print(f"  {model}: {n_fail}/{len(rows)} format failures "
          f"({n_fail / len(rows):.3%}), counted as WRONG")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-model", default="claude-sonnet-5")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-json", type=Path, default=Path("results/e6_frontier.json"))
    ap.add_argument("--out-fig", type=Path, default=Path("results/e6_frontier.png"))
    ap.add_argument("--volume", type=float, default=1_000_000.0,
                    help="clauses over which capital is amortised, for the greenfield case")
    a = ap.parse_args()

    from src.data.loading import label_names, load_ledgar
    ds = load_ledgar()
    names = label_names(ds)

    # ---- Tier 2 cost per 1k, from the ACTUAL usage recorded for the run ---------------
    stage1 = json.loads((a.results_dir / "stage1_results.json").read_text())
    leg = stage1["models"][a.api_model]
    api_usd_per_1k = 1000.0 * float(leg["actual_usd"]) / int(leg["n"])

    # ---- Tier 0 cost per 1k, both amortisation cases ---------------------------------
    hw = load_hardware()
    bk = breakeven(api_usd_per_1k, hw=hw, allow_assumed_tariff=True)
    tier0_sunk = bk.energy_usd_per_1k                       # capital already spent
    tier0_greenfield = bk.local_usd_per_1k(a.volume)        # capital attributable

    # ---- per-seed cascade sweeps -----------------------------------------------------
    per_seed = {}
    for s in SEEDS:
        dev = load_split(a.results_dir / f"dev_logits_int8_local_ce_seed{s}.npz",
                         "dev_2000", for_calibration=True)
        tst_npz = np.load(a.results_dir / f"test_logits_int8_local_ce_seed{s}.npz")
        t_logits = tst_npz["test_3000_logits"]
        t_labels = tst_npz["test_3000_labels"]
        t_index = tst_npz["test_3000_indices"]

        labels_by_index = {int(i): names[int(l)] for i, l in zip(t_index, t_labels)}
        api_pred = load_api_leg(a.results_dir / "stage1_results.json", a.api_model,
                                labels_by_index)
        missing = [int(i) for i in t_index if int(i) not in api_pred]
        if missing:
            raise SystemExit(
                f"{len(missing)} test rows have no {a.api_model} result (first "
                f"{missing[:5]}). Refusing a partial join: dropping rows changes the "
                f"denominator of every metric on this curve.")

        # CALIBRATE ON DEV, APPLY TO TEST. The thresholds come from dev's signal
        # distribution; only their APPLICATION touches test.
        dev_score = CLOSED_FORM[SIGNAL](dev.logits)
        thresholds = np.quantile(dev_score, TARGETS)
        test_score = CLOSED_FORM[SIGNAL](t_logits)
        gold = np.array([names[int(l)] for l in t_labels], dtype=object)
        t0_pred = np.array([names[int(j)] for j in t_logits.argmax(-1)], dtype=object)
        api_p = np.array([api_pred[int(i)] for i in t_index], dtype=object)
        t0_correct = gold == t0_pred
        api_correct = gold == api_p

        # MACRO-F1 IS THE PRIMARY METRIC (CLAUDE.md), so the cascade's combined
        # PREDICTIONS are scored, not just its hit rate. An escalation policy can raise
        # accuracy while losing tail classes, and accuracy alone would hide that — which
        # is the whole reason this project leads with macro-F1.
        def macro_for(esc_mask, gold=gold, t0_pred=t0_pred, api_p=api_p):
            combined = np.where(esc_mask, api_p, t0_pred)
            return score(list(gold), [None if c is None else str(c) for c in combined],
                         labels=sorted(set(gold))).macro_f1

        for case, t0_cost in (("sunk", tier0_sunk), ("greenfield", tier0_greenfield)):
            pts = cascade_points(score=test_score, tier0_correct=t0_correct,
                                 api_correct=api_correct, tier0_usd_per_1k=t0_cost,
                                 api_usd_per_1k=api_usd_per_1k, thresholds=thresholds,
                                 macro_f1_fn=macro_for)
            per_seed.setdefault(case, {})[s] = pts

    # ---- aggregate over seeds at each threshold index --------------------------------
    agg = {}
    for case, byseed in per_seed.items():
        n = len(byseed[SEEDS[0]])
        rows = []
        for i in range(n):
            esc = [byseed[s][i].escalation_rate for s in SEEDS]
            acc = [byseed[s][i].accuracy for s in SEEDS]
            mf1 = [byseed[s][i].macro_f1 for s in SEEDS]
            usd = [byseed[s][i].usd_per_1k for s in SEEDS]
            rows.append({"escalation_mean": st.mean(esc), "escalation_sd": st.stdev(esc),
                         "macro_f1_mean": st.mean(mf1), "macro_f1_sd": st.stdev(mf1),
                         "accuracy_mean": st.mean(acc), "accuracy_sd": st.stdev(acc),
                         "usd_per_1k_mean": st.mean(usd), "usd_per_1k_sd": st.stdev(usd),
                         "threshold": byseed[SEEDS[0]][i].threshold})
        agg[case] = rows

    from src.eval.pareto import CascadePoint
    fronts = {}
    for case, rows in agg.items():
        # Pareto on MACRO-F1, the primary metric — CascadePoint.accuracy carries it here
        # so pareto_frontier's (cost down, metric up) dominance is computed on macro-F1.
        pts = [CascadePoint(escalation_rate=r["escalation_mean"], threshold=r["threshold"],
                            accuracy=r["macro_f1_mean"], macro_f1=r["macro_f1_mean"],
                            usd_per_1k=r["usd_per_1k_mean"], n_escalated=0) for r in rows]
        fronts[case] = [{"escalation_rate": p.escalation_rate, "macro_f1": p.macro_f1,
                         "usd_per_1k": p.usd_per_1k, "threshold": p.threshold}
                        for p in pareto_frontier(pts)]

    payload = {
        "experiment": "E6 (cost axis only)",
        "axis_limitation": ("cost-vs-ACCURACY only. Tier 1 throughput unmeasured, Tier 2 "
                            "not_applicable; no latency or SLA claim is supported."),
        "signal": SIGNAL, "signal_basis": "E5 registered clause (§3ai): spread < 0.05",
        "precision": "int8", "isa": "arm64_local",
        "calibrated_on": "dev_2000", "applied_to": "test_3000",
        "api_model": a.api_model, "api_usd_per_1k": api_usd_per_1k,
        "tier0_usd_per_1k_sunk": tier0_sunk,
        "tier0_usd_per_1k_greenfield": tier0_greenfield,
        "greenfield_volume": a.volume,
        "device_cost_usd": device_cost_usd(hw),
        "tariff_is_assumed": bk.tariff_is_assumed, "tariff_basis": bk.tariff_basis,
        "energy_is_soc_only": bk.energy_is_soc_only,
        "seeds": list(SEEDS), "curves": agg, "pareto": fronts,
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {a.out_json}")

    _figure(agg, fronts, payload, a.out_fig)
    return 0


def _figure(agg, fronts, meta, out_fig: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, case, title in ((axes[0], "sunk", "Sunk capital (the Mac Mini exists)"),
                            (axes[1], "greenfield",
                             f"Greenfield (capital over {meta['greenfield_volume']:,.0f})")):
        rows = agg[case]
        x = [r["usd_per_1k_mean"] for r in rows]
        y = [r["macro_f1_mean"] for r in rows]
        yerr = [r["macro_f1_sd"] for r in rows]
        ax.errorbar(x, y, yerr=yerr, fmt="o", ms=3, lw=0.8, alpha=0.45,
                    label="operating points (mean ± sd, 3 seeds)")
        f = fronts[case]
        ax.plot([p["usd_per_1k"] for p in f], [p["macro_f1"] for p in f],
                "-", lw=2.0, label="Pareto frontier")
        ax.axhline(rows[0]["macro_f1_mean"], ls=":", lw=1.0, color="grey")
        ax.annotate("Tier 0 alone", (x[0], y[0]), textcoords="offset points",
                    xytext=(8, -12), fontsize=8)
        ax.annotate(f"{meta['api_model']} alone", (x[-1], y[-1]),
                    textcoords="offset points", xytext=(-95, 6), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("USD per 1,000 clauses (log)")
        ax.set_ylabel("macro-F1 on test_3000 (primary metric)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    cap = (
        "E6 — cost vs accuracy, 3-tier cascade. Tier 0 = DeBERTa-v3-base ONNX-INT8 on "
        "arm64 (deployed precision/ISA); escalation by margin (E5 §3ai). Thresholds "
        "calibrated on dev_2000, applied to test_3000. Mean ± sd over 3 seeds.\n"
        "THIS IS A COST AXIS ONLY. Tier 1 throughput is unmeasured and Tier 2's is "
        "not-applicable (queue latency is not hardware throughput), so NO latency, "
        "SLA or queueing claim is supported by this figure.\n"
        f"Tariff {'ASSUMED' if meta['tariff_is_assumed'] else 'measured'}; energy is "
        f"SoC-package only, excluding RAM/SSD/PSU. Tier 2 cost from measured usage."
    )
    fig.text(0.012, -0.015, cap, fontsize=7.4, va="top", ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170, bbox_inches="tight")
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    raise SystemExit(main())
