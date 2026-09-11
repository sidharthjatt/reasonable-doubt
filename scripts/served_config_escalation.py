#!/usr/bin/env python3
"""Does escalation help at the SERVED operating point? Descriptive; NOT an E6 rebuild.

    python scripts/served_config_escalation.py

Everything is read off disk and costs **$0.00**. No API call is made: the Tier 2 arm is
the Sonnet-5 zero-shot predictions already in `results/stage1_results.json`.

WHAT THIS IS. The deployed service (§3bk) serves E1b seed 1 as ONNX-FP32 on arm64 and
escalates when `margin < threshold`, with the threshold read from the SERVED file rather
than recomputed here. This applies exactly that rule to `test_3000` and asks whether the
cascade's answers beat Tier 0's own.

WHAT THIS IS NOT.
  * NOT an E6 rebuild. E6, H1 and E8 remain on E1 and no frontier, break-even or cost
    figure is touched. No USD number is computed at all.
  * NOT a held-out estimate. §3bc records that the 4.056% RATE was selected on test, so
    every test_3000 number at this operating point is IN-SAMPLE for the rate that chose
    it. The threshold VALUE is dev-calibrated (hard rule 1 intact); the RATE is not.
  * NOT a 3-seed result. n=1 — the served artefact. Hard rule 2 forbids reporting this as
    a model result, and it is reported as a property of one deployed configuration.
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
from src.router.signals import CLOSED_FORM  # noqa: E402

BOOTSTRAP_SEED = 20260911
MODEL = "claude-sonnet-5"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar on the discordant pairs."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logits", type=Path,
                    default=Path("results/test_logits_fp32_local_ce10ep_seed1.npz"))
    ap.add_argument("--threshold-file", type=Path,
                    default=Path("configs/router_threshold_fp32.json"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("results/served_config_escalation.json"))
    ap.add_argument("--n-resamples", type=int, default=10_000)
    a = ap.parse_args()

    # ---- the SERVED threshold, read not recomputed -------------------------------
    thr_d = json.loads(a.threshold_file.read_text())
    thr = float(thr_d["threshold"])
    if thr_d["calibrated_on"] != "dev_2000":
        raise SystemExit(f"threshold says calibrated_on={thr_d['calibrated_on']!r}; "
                         f"hard rule 1 permits dev only")
    print(f"  served threshold : {thr:.6f}  ({thr_d['precision']}, "
          f"{thr_d['percentile']:.4f}%, artefact {Path(thr_d['artefact']).name})")

    # ---- Tier 0 logits, with the precision/ISA guard -----------------------------
    npz = np.load(a.logits, allow_pickle=True)
    prov = json.loads(str(npz["provenance"]))
    if prov["precision"] != thr_d["precision"] or prov["isa"] != thr_d["isa"]:
        raise SystemExit(
            f"REFUSING: logits are {prov['precision']}/{prov['isa']} but the threshold "
            f"is {thr_d['precision']}/{thr_d['isa']}. Margin distributions are not "
            f"interchangeable across precision.")
    if Path(prov["artefact"]).name != Path(thr_d["artefact"]).name:
        raise SystemExit(
            f"REFUSING: logits come from {Path(prov['artefact']).name} but the threshold "
            f"was calibrated on {Path(thr_d['artefact']).name}.")
    logits = npz["test_3000_logits"]
    gold_i = npz["test_3000_labels"].astype(int)
    index = npz["test_3000_indices"]
    print(f"  tier 0 logits    : {Path(prov['artefact']).name} "
          f"[{prov['precision']}/{prov['isa']}], {len(gold_i)} rows")

    names: list[str] = json.loads(Path("configs/labels.json").read_text())["labels"]
    name_to_id = {n: i for i, n in enumerate(names)}

    # ---- Tier 2 arm: predictions already on disk ---------------------------------
    from src.data.schema import ParseFailure, parse_response

    rows = json.loads(Path("results/stage1_results.json").read_text())
    rows = rows["models"][MODEL]["rows"]
    api_pred: dict[int, str | None] = {}
    for k, body in rows.items():
        try:
            api_pred[int(k)] = parse_response(body.get("text", "")).label
        except ParseFailure:
            api_pred[int(k)] = None
    missing = [int(i) for i in index if int(i) not in api_pred]
    if missing:
        raise SystemExit(f"{len(missing)} test rows have no {MODEL} result "
                         f"(first {missing[:5]}). Refusing a partial join.")

    # ---- apply the SERVED rule: escalate when margin < threshold ------------------
    sig = CLOSED_FORM["margin"](logits)
    escalate = sig < thr                      # strict <, matching serve.yaml's rule
    n_esc = int(escalate.sum())

    t0 = logits.argmax(-1).astype(int)
    api = np.array([name_to_id.get(api_pred[int(i)] or "", -1) for i in index])
    casc = np.where(escalate, api, t0)

    n_lab = len(names)
    gold_names = [names[int(g)] for g in gold_i]
    ref = {}
    for tag, arm in (("tier0", t0), ("cascade", casc)):
        r = score(gold_names, [names[int(j)] if j >= 0 else None for j in arm],
                  labels=names)
        fast = fast_macro_f1(gold_i, arm, n_lab)
        if abs(fast - r.macro_f1) > 1e-12:
            raise SystemExit(f"fast scorer disagrees on {tag}")
        ref[tag] = r.macro_f1
    delta = ref["cascade"] - ref["tier0"]

    bs = paired_bootstrap(
        len(gold_i),
        lambda i: fast_macro_f1(gold_i[i], t0[i], n_lab),
        lambda i: fast_macro_f1(gold_i[i], casc[i], n_lab),
        n_resamples=a.n_resamples, seed=BOOTSTRAP_SEED)

    # ---- McNemar on the escalated rows only --------------------------------------
    e = escalate
    t0_ok = (t0[e] == gold_i[e])
    api_ok = (api[e] == gold_i[e])
    b = int((api_ok & ~t0_ok).sum())      # API right, Tier 0 wrong
    c = int((~api_ok & t0_ok).sum())      # Tier 0 right, API wrong
    p_mc = mcnemar_exact(b, c)

    print(f"\n  escalated        : {n_esc}/{len(gold_i)} = {n_esc/len(gold_i):.4f}")
    print(f"  Tier 0 alone     : macro-F1 {ref['tier0']:.6f}")
    print(f"  cascade          : macro-F1 {ref['cascade']:.6f}")
    print(f"  delta            : {delta:+.6f}")
    print("\n  PAIRED BOOTSTRAP — cascade minus Tier 0 alone, same rows, n=1 seed")
    print(bs.render("cascade - Tier 0"))
    print(f"\n  McNEMAR on the {n_esc} ESCALATED rows (accuracy, not macro-F1)")
    print(f"    API right / Tier 0 wrong : {b}")
    print(f"    Tier 0 right / API wrong : {c}")
    print(f"    net                      : {b - c:+d}   exact two-sided p = {p_mc:.4f}")
    print(f"    Tier 0 accuracy on them  : {t0_ok.mean():.4f}")
    print(f"    API    accuracy on them  : {api_ok.mean():.4f}")

    helps = bs.excludes_zero and delta > 0
    verdict = ("the cascade beats Tier 0 alone at the served operating point"
               if helps else
               "the cascade does NOT beat Tier 0 alone at the served operating point: "
               "the paired interval includes 0")
    print(f"\n  VERDICT: {verdict}")
    print("  IN-SAMPLE (§3bc): the 4.056% RATE was selected on test. n=1 seed, "
          "descriptive only.")

    a.out_json.write_text(json.dumps({
        "what": "served-config escalation check (§3bk artefact); DESCRIPTIVE, not E6",
        "cost_usd": 0.0,
        "is_e6_rebuild": False,
        "in_sample": True,
        "in_sample_reason": "§3bc: the 4.056% escalation RATE was selected on test_3000; "
                            "the threshold VALUE is dev_2000-calibrated (hard rule 1).",
        "seeds": 1,
        "artefact": prov["artefact"], "precision": prov["precision"], "isa": prov["isa"],
        "threshold": thr, "threshold_file": str(a.threshold_file),
        "escalation_rule": "margin < threshold",
        "n_rows": int(len(gold_i)), "n_escalated": n_esc,
        "escalation_rate": n_esc / len(gold_i),
        "macro_f1": ref, "delta": delta,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap": bs.__dict__,
        "mcnemar_escalated_rows": {
            "api_right_tier0_wrong": b, "tier0_right_api_wrong": c, "net": b - c,
            "p_exact_two_sided": p_mc,
            "tier0_accuracy": float(t0_ok.mean()), "api_accuracy": float(api_ok.mean()),
            "metric": "accuracy on escalated rows, NOT macro-F1"},
        "verdict": verdict,
    }, indent=2) + "\n")
    print(f"\n  wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
