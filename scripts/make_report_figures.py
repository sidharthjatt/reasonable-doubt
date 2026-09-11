#!/usr/bin/env python3
"""Generate REPORT.md's figures. Every chart reads a results file; none takes a literal.

    python scripts/make_report_figures.py

Costs $0.00. Writes docs/figures/*.png at 150 dpi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R, FIG = Path("results"), Path("docs/figures")
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, LINE = "#1a1a1a", "#6b6b6b", "#d8d3ca"
LOCAL, API_A, API_B, BAD = "#3d6b4f", "#8c6d3f", "#b4642c", "#a33b26"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10.5,
    "axes.edgecolor": LINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "medium",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def bare(ax, keep=("left", "bottom")):
    for k, sp in ax.spines.items():
        sp.set_visible(k in keep)
    ax.tick_params(length=3, width=0.8)


def load(name):
    return json.loads((R / name).read_text())


# ---------------------------------------------------------------- 1. cost per 1k
def fig_cost():
    d = load("cost_per_1k.json")
    items = [("Tier 0 local\nFP32 encoder", d["local_tier0_fp32"]["usd_per_1k"], LOCAL),
             ("Sonnet 5\nbatch + caching", d["api"]["claude-sonnet-5"]["usd_per_1k"], API_B),
             ("Haiku 4.5\nbatch, cannot cache",
              d["api"]["claude-haiku-4-5-20251001"]["usd_per_1k"], API_A)]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    y = range(len(items))
    ax.barh(list(y), [v for _, v, _ in items], color=[c for *_, c in items],
            height=.52, zorder=3)
    for i, (_, v, _) in enumerate(items):
        ax.text(v * 1.35, i, f"${v:,.5f}".rstrip("0") if v < 0.01 else f"${v:.4f}",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), [n for n, *_ in items], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 4)
    ax.set_xlabel("USD per 1,000 clauses  (log scale)")
    ax.invert_yaxis()
    bare(ax)
    ratio = d["api"]["claude-sonnet-5"]["usd_per_1k"] / d["local_tier0_fp32"]["usd_per_1k"]
    ax.set_title(f"Sonnet 5 costs {ratio:,.0f}× the local encoder per clause",
                 loc="left", pad=10)
    fig.savefig(FIG / "cost_per_1k.png")
    plt.close(fig)
    print("docs/figures/cost_per_1k.png  <- results/cost_per_1k.json")


# ------------------------------------------------------- 2. INT8 vs FP32 across ISAs
def fig_isa():
    d = load("isa_matrix.json")
    mf = d["test_3000_macro_f1"]["rows"]
    can = d["canary_accuracy_200_train_rows"]["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.3),
                                 gridspec_kw={"width_ratios": [1, 1.12]})

    x = range(len(mf))
    w = .34
    a1.bar([i - w / 2 for i in x], [r["fp32"] for r in mf], w, color=LOCAL,
           label="FP32", zorder=3)
    a1.bar([i + w / 2 for i in x], [r["int8"] for r in mf], w, color=BAD,
           label="INT8", zorder=3)
    for i, r in enumerate(mf):
        a1.text(i - w / 2, r["fp32"] + .02, f"{r['fp32']:.3f}", ha="center", fontsize=8)
        lab = f"{r['int8']:.6f}" if r["int8"] < .01 else f"{r['int8']:.3f}"
        a1.text(i + w / 2, r["int8"] + .02, lab, ha="center", fontsize=8,
                color=BAD, fontweight="bold" if r["int8"] < .01 else "normal")
    a1.set_xticks(list(x), ["macOS arm64\n(VNNI n/a)",
                            "Linux x86 Xeon\nno AVX-512 VNNI"], fontsize=8)
    a1.set_ylim(0, 1.0)
    a1.set_ylabel("macro-F1, test_3000")
    a1.set_title("Same weights, same rows", loc="left", pad=8)
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    bare(a1)
    a1.annotate("chance", xy=(1.17, 0.02), xytext=(1.17, 0.30), fontsize=8, color=BAD,
                ha="center", arrowprops=dict(arrowstyle="->", color=BAD, lw=.9))

    c = [r for r in can if r["int8"] is not None]
    x2 = range(len(c))
    a2.bar([i - w / 2 for i in x2], [r["fp32"] for r in c], w, color=LOCAL, zorder=3)
    a2.bar([i + w / 2 for i in x2], [r["int8"] for r in c], w, color=BAD, zorder=3)
    for i, r in enumerate(c):
        a2.text(i - w / 2, r["fp32"] + .02, f"{r['fp32']:.4f}", ha="center", fontsize=8)
        a2.text(i + w / 2, r["int8"] + .02, f"{r['int8']:.4f}", ha="center", fontsize=8,
                color=BAD)
    a2.set_xticks(list(x2), ["macOS arm64", "Linux aarch64\n(same CPU, container)"],
                  fontsize=8)
    a2.set_ylim(0, 1.0)
    a2.set_ylabel("accuracy, 200 canary rows")
    a2.set_title("Different metric — the container refusal", loc="left", pad=8)
    bare(a2)
    fig.savefig(FIG / "int8_vs_fp32_by_isa.png")
    plt.close(fig)
    print("docs/figures/int8_vs_fp32_by_isa.png  <- results/isa_matrix.json")


# ------------------------------------------------------------ 3. accuracy vs cost
def fig_acc_cost():
    cost = load("cost_per_1k.json")
    api = load("stage1_scored_test3000.json")["models"]
    t0 = load("fp32_local_onnx_ce10ep_seed1.json")["metrics"]
    pts = [("Tier 0 encoder (FP32, served)", cost["local_tier0_fp32"]["usd_per_1k"],
            t0["macro_f1"], LOCAL),
           ("Sonnet 5 zero-shot", cost["api"]["claude-sonnet-5"]["usd_per_1k"],
            api["claude-sonnet-5"]["macro_f1"], API_B),
           ("Haiku 4.5 zero-shot", cost["api"]["claude-haiku-4-5-20251001"]["usd_per_1k"],
            api["claude-haiku-4-5-20251001"]["macro_f1"], API_A)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for name, c, f, col in pts:
        ax.scatter([c], [f], s=95, color=col, zorder=3, edgecolor="white", linewidth=1.2)
        ax.annotate(f"{name}\n{f:.4f}  ·  ${c:,.5f}/1k".replace("$0.00088", "$0.00088"),
                    (c, f), textcoords="offset points",
                    xytext=(12, -4 if "Tier 0" in name else 8), fontsize=8.3, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 8)
    ax.set_ylim(.5, .9)
    ax.set_xlabel("USD per 1,000 clauses  (log scale)")
    ax.set_ylabel("macro-F1, test_3000")
    ax.set_title("Cheaper and better, on this task", loc="left", pad=10)
    bare(ax)
    fig.savefig(FIG / "accuracy_vs_cost.png")
    plt.close(fig)
    print("docs/figures/accuracy_vs_cost.png  <- results/cost_per_1k.json, "
          "results/stage1_scored_test3000.json, results/fp32_local_onnx_ce10ep_seed1.json")


# --------------------------------------------------------- 4. escalation effect
def fig_escalation():
    d = load("served_config_escalation.json")
    delta, bs = d["delta"], d["bootstrap"]
    lo, hi = bs["lo"], bs["hi"]
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.plot([lo, hi], [0, 0], color=LOCAL, lw=3, solid_capstyle="round", zorder=2)
    ax.scatter([delta], [0], s=85, color=LOCAL, zorder=3, edgecolor="white", linewidth=1.2)
    ax.text(delta, .32, f"+{delta:.6f}", ha="center", fontsize=9, color=INK)
    ax.text(lo, -.42, f"{lo:+.4f}", ha="center", fontsize=8, color=MUTED)
    ax.text(hi, -.42, f"{hi:+.4f}", ha="center", fontsize=8, color=MUTED)
    ax.text(0, .62, "no effect", ha="center", fontsize=8, color=MUTED)
    ax.set_ylim(-.9, .9)
    ax.set_yticks([])
    ax.set_xlabel("cascade minus Tier 0 alone, macro-F1  (95% CI, paired bootstrap, "
                  f"{bs['n_resamples']:,} resamples)")
    ax.set_title(f"The interval covers zero  (p = {bs['p_sign']:.3f}, n = 1 seed, "
                 "in-sample operating point)", loc="left", pad=10)
    bare(ax, keep=("bottom",))
    fig.savefig(FIG / "escalation_effect.png")
    plt.close(fig)
    print("docs/figures/escalation_effect.png  <- results/served_config_escalation.json")


# ------------------------------------------------- 5. what happens on escalated rows
def fig_escalated_rows():
    d = load("served_config_escalation.json")
    m = d["mcnemar_escalated_rows"]
    t0_all = load("fp32_local_onnx_ce10ep_seed1.json")["metrics"]["accuracy"]
    n = d["n_escalated"]
    bars = [(f"Tier 0\nall {d['n_rows']:,} rows", t0_all, "#b8c4bc"),
            (f"Tier 0\nthe {n} flagged", m["tier0_accuracy"], LOCAL),
            (f"Sonnet 5\nthe same {n}", m["api_accuracy"], API_B)]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = range(len(bars))
    ax.bar(list(x), [v for _, v, _ in bars], .5, color=[c for *_, c in bars], zorder=3)
    for i, (_, v, _) in enumerate(bars):
        ax.text(i, v + .018, f"{v:.4f}", ha="center", fontsize=9.5)
    ax.set_xticks(list(x), [n for n, *_ in bars], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    bare(ax)
    ax.plot([1, 1, 2, 2], [.62, .66, .66, .62], color=MUTED, lw=.9, ls=(0, (3, 3)))
    ax.text(1.5, .69, f"McNemar {m['api_right_tier0_wrong']}/{m['tier0_right_api_wrong']},"
                      f" net {m['net']:+d}, p = {m['p_exact_two_sided']:.3f}",
            ha="center", fontsize=8.3, color=MUTED)
    ax.set_title("The router finds the hard rows. The frontier model does not rescue them.",
                 loc="left", pad=10)
    fig.savefig(FIG / "escalated_rows.png")
    plt.close(fig)
    print("docs/figures/escalated_rows.png  <- results/served_config_escalation.json, "
          "results/fp32_local_onnx_ce10ep_seed1.json")


if __name__ == "__main__":
    fig_cost()
    fig_isa()
    fig_acc_cost()
    fig_escalation()
    fig_escalated_rows()
