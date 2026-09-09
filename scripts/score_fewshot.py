#!/usr/bin/env python3
"""E8 — few-shot vs zero-shot Sonnet-5, PAIRED on the same 1,000 rows.

Registered in PREREGISTRATION.md ("E8") BEFORE these numbers were opened. The band is
mechanical and is printed back with the result so the disposition is not chosen after
seeing it.

Costs $0.00: every response is already on disk. No API call, no ledger append.

Three things this script refuses to do
--------------------------------------
1. Compare 1,000 few-shot rows against 3,000 zero-shot rows. The few-shot ids are
   asserted to be a strict subset of both stage1's rows and test_3000's manifest, and
   zero-shot is RESTRICTED to that subset. An unpaired 1000-vs-3000 delta would mix a
   prompt effect with a different row draw.
2. Score the two arms over different label sets. Macro-F1 is defined relative to the set
   averaged over, and a 1,000-row subset does not cover 100 classes; both arms use the
   classes present in the gold of THESE rows, and `classes_averaged` is reported.
3. Trust its own fast scorer. `fast_macro_f1` is asserted equal to the project's
   `src.eval.metrics.score` on the real arrays before a single resample is drawn.
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

SEEDS = (1, 2, 3)
BOOTSTRAP_SEED = 20260909
N_RESAMPLES = 10_000
MODEL = "claude-sonnet-5"


def load_leg(path: Path, model: str) -> dict[int, str | None]:
    """Per-row prediction keyed by dataset row index, via the CANONICAL parser.

    `src.data.schema.parse_response`, not a local `json.loads` — §3e instance 2, and the
    defect that scored Haiku at 12% on the E6 frontier. Label matching stays EXACT per
    §3ag: a label outside the 100-class space is None and counts WRONG, because
    format_failure_rate is a measured quantity of this project.
    """
    from src.data.schema import ParseFailure, parse_response

    rows = json.loads(path.read_text())["models"][model]["rows"]
    out: dict[int, str | None] = {}
    n_fail = 0
    for key, body in rows.items():
        if not key.isdigit():
            raise ValueError(f"cannot parse a row index from key {key!r}")
        try:
            out[int(key)] = parse_response(body.get("text", "")).label
        except ParseFailure:
            out[int(key)] = None
            n_fail += 1
    print(f"  {path.name:<24} {model}: {len(rows)} rows, "
          f"{n_fail} format failures ({n_fail / len(rows):.3%}), counted WRONG")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-json", type=Path, default=Path("results/e8_fewshot.json"))
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    a = ap.parse_args()

    names: list[str] = json.loads(Path("configs/labels.json").read_text())["labels"]
    if len(names) != 100:
        raise SystemExit(f"expected 100 labels, got {len(names)}")
    name_to_id = {n: i for i, n in enumerate(names)}

    # ---- the paired row set -----------------------------------------------------------
    fs_pred = load_leg(a.results_dir / "fewshot_results.json", MODEL)
    zs_pred = load_leg(a.results_dir / "stage1_results.json", MODEL)

    manifest = json.loads(Path("configs/manifests/test_3000.json").read_text())
    test_idx = set(manifest["indices"])
    ids = sorted(fs_pred)
    if not set(ids) <= set(zs_pred):
        missing = sorted(set(ids) - set(zs_pred))
        raise SystemExit(f"{len(missing)} few-shot rows absent from stage1 "
                         f"(first {missing[:5]}). Refusing an unpaired comparison.")
    if not set(ids) <= test_idx:
        raise SystemExit("few-shot ids are not a subset of the test_3000 manifest; the "
                         "gold labels would not be the ones these rows were drawn with.")
    print(f"\n  paired row set: {len(ids)} rows, verified subset of stage1 (3,000) "
          f"and of test_3000's manifest")

    # ---- gold, taken from the Tier 0 artefacts (same source the frontier used) ---------
    per_seed_logits: dict[int, np.ndarray] = {}
    gold_ids: np.ndarray | None = None
    for s in SEEDS:
        npz = np.load(a.results_dir / f"test_logits_int8_local_ce_seed{s}.npz")
        row_of = {int(i): k for k, i in enumerate(npz["test_3000_indices"])}
        take = np.array([row_of[i] for i in ids])
        g = npz["test_3000_labels"][take].astype(int)
        if gold_ids is None:
            gold_ids = g
        elif not np.array_equal(gold_ids, g):
            raise SystemExit(f"seed {s} disagrees with seed {SEEDS[0]} on the GOLD "
                             "labels of the same rows; the artefacts are inconsistent.")
        per_seed_logits[s] = npz["test_3000_logits"][take]
    assert gold_ids is not None

    present = sorted(set(int(g) for g in gold_ids))
    label_set = [names[i] for i in present]
    compact = {lab: k for k, lab in enumerate(present)}   # dense ids for the fast scorer
    gold_c = np.array([compact[int(g)] for g in gold_ids])
    n_lab = len(present)
    print(f"  classes present in these {len(ids)} rows: {n_lab} of 100 "
          f"— BOTH arms are averaged over exactly these")

    def to_compact(preds: list[str | None]) -> np.ndarray:
        """Label strings -> dense ids. Unmatched, and any class absent from THIS row
        set's label space, map to -1: never correct, never inflating the average."""
        return np.array([compact.get(name_to_id.get(p, -1), -1) if p is not None else -1
                         for p in preds])

    gold_names = [names[int(g)] for g in gold_ids]
    fs_names = [fs_pred[i] for i in ids]
    zs_names = [zs_pred[i] for i in ids]
    fs_c, zs_c = to_compact(fs_names), to_compact(zs_names)

    # ---- reference scores, through the project's scorer --------------------------------
    rep = {}
    for tag, preds in (("fewshot", fs_names), ("zeroshot", zs_names)):
        r = score(gold_names, preds, labels=label_set)
        rep[tag] = r.as_dict()
        print(f"\n  Sonnet-5 {tag} (n={len(ids)}):\n{r.render()}")

    # ---- Tier 0 on the SAME rows, per seed ---------------------------------------------
    t0_c, t0_f1 = {}, {}
    for s in SEEDS:
        p = per_seed_logits[s].argmax(-1).astype(int)
        t0_c[s] = np.array([compact.get(int(j), -1) for j in p])
        r = score(gold_names, [names[int(j)] for j in p], labels=label_set)
        t0_f1[s] = r.macro_f1
        rep[f"tier0_seed{s}"] = r.as_dict()
    t0_mean = float(np.mean(list(t0_f1.values())))
    t0_sd = float(np.std(list(t0_f1.values()), ddof=1))
    print(f"\n  Tier 0 INT8 on the same {len(ids)} rows: macro-F1 "
          f"{t0_mean:.4f} +/- {t0_sd:.4f} (3 seeds) "
          f"[{', '.join(f'{t0_f1[s]:.4f}' for s in SEEDS)}]")

    # ---- the fast scorer must agree with the slow one before any resample --------------
    for tag, c in (("fewshot", fs_c), ("zeroshot", zs_c),
                   *[(f"tier0_seed{s}", t0_c[s]) for s in SEEDS]):
        fast = fast_macro_f1(gold_c, c, n_lab)
        slow = rep[tag]["macro_f1"]
        if abs(fast - slow) > 1e-12:
            raise SystemExit(f"fast_macro_f1 disagrees with src.eval.metrics.score on "
                             f"{tag}: {fast!r} vs {slow!r}. Refusing to bootstrap a "
                             "statistic that is not the reported one.")
    print(f"\n  fast scorer == src.eval.metrics.score on all "
          f"{2 + len(SEEDS)} arms (exact)")

    # ---- E8's registered quantities -----------------------------------------------------
    delta_fs = rep["fewshot"]["macro_f1"] - rep["zeroshot"]["macro_f1"]
    gap = t0_mean - rep["zeroshot"]["macro_f1"]
    fraction = delta_fs / gap if gap != 0 else float("nan")

    bs = paired_bootstrap(
        len(ids),
        lambda i: fast_macro_f1(gold_c[i], zs_c[i], n_lab),
        lambda i: fast_macro_f1(gold_c[i], fs_c[i], n_lab),
        n_resamples=a.n_resamples, seed=BOOTSTRAP_SEED)
    bs_t0 = paired_bootstrap(
        len(ids),
        lambda i: fast_macro_f1(gold_c[i], zs_c[i], n_lab),
        lambda i: float(np.mean([fast_macro_f1(gold_c[i], t0_c[s][i], n_lab)
                                 for s in SEEDS])),
        n_resamples=a.n_resamples, seed=BOOTSTRAP_SEED)

    print("\n  PAIRED BOOTSTRAP (same rows, resampled together)")
    print(bs.render("few-shot - zero-shot"))
    print(bs_t0.render("Tier 0 - zero-shot"))

    # ---- the registered band, applied mechanically ---------------------------------------
    if not bs.excludes_zero:
        disposition = ("NO MEASURABLE EFFECT — CI includes 0. Treated as <= 0.20 per the "
                       "registered band; the CI is the result, not the point estimate.")
    elif fraction >= 0.50:
        disposition = "PREMISE LIVE — restatement of H1/E4/E4b/E6 is DEFERRED."
    elif fraction <= 0.20:
        disposition = "CAPABILITY, NOT PROMPTING — restatement proceeds."
    else:
        disposition = "INDETERMINATE — neither disposition is licensed."

    print(f"\n  gap to close (Tier 0 - zero-shot) : {gap:+.4f}")
    print(f"  delta_fs                          : {delta_fs:+.4f}")
    print(f"  fraction_closed                   : {fraction:+.4f}")
    print(f"\n  E8 DISPOSITION: {disposition}")
    print("\n  BOUND, registered in advance: exemplars_8 covers 6 distinct classes of "
          "100.\n  E8 falsifies 'prompting closes the gap' ONLY for the 8-exemplar "
          "prompt actually run.\n  A class-covering prompt is untested and unbudgeted.")

    out = {
        "experiment": "E8",
        "cost_usd": 0.0,
        "paired": True,
        "n_rows": len(ids),
        "classes_averaged": n_lab,
        "exemplar_classes_covered": json.loads(
            Path("configs/manifests/exemplars_8.json").read_text())["classes_represented"],
        "model": MODEL,
        "reports": rep,
        "tier0_macro_f1_mean": t0_mean,
        "tier0_macro_f1_sd": t0_sd,
        "tier0_macro_f1_per_seed": t0_f1,
        "delta_fs": delta_fs,
        "gap": gap,
        "fraction_closed": fraction,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_fewshot_vs_zeroshot": bs.__dict__,
        "bootstrap_tier0_vs_zeroshot": bs_t0.__dict__,
        "disposition": disposition,
        "bound": ("exemplars_8 covers 6 of 100 classes; E8 falsifies 'prompting closes "
                  "the gap' only for the 8-exemplar prompt actually run."),
    }
    a.out_json.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
