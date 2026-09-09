"""Aggregate Tier 0 throughput replicates, and split within- from between-artefact variance.

    python scripts/bench_aggregate.py --tier tier0_encoder_onnx_int8

E6 needs ONE throughput figure. Choosing an estimator from three single measurements was
the wrong tool -- at n=1 per artefact, "this graph is slower" and "this run was slower"
are not separable, and only one of them is possible (the graphs are architecturally
identical). Replication separates them; this script does the separation.

THE BETWEEN-ARTEFACT TERM IS A TEST, NOT A SUMMARY. §3aa's weight-independence claim --
identical ops on identical shapes, so throughput does not depend on the weights -- was
validated against the untrained probe. With replicates it becomes checkable directly:
if weights do not matter, between-artefact variance should be indistinguishable from
within-artefact variance, and the F-ratio near 1. A large between-artefact term would
contradict a claim E6's provenance now rests on, so it is computed by the tooling and
reported every time rather than left to whoever reads the files.

NO RUN IS EXCLUDED FOR BEING AN OUTLIER. The rule is registered in PREREGISTRATION §3aj
before these runs: the headline is the mean over ALL runs. Exclusion requires an
identifiable, stated cause, applies symmetrically to fast and slow runs, and is recorded
with its reason -- `--exclude` demands `--exclude-reason` for exactly that purpose.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

FIELDS = ("throughput_rps", "p50_latency_ms", "p95_latency_ms",
          "marginal_soc_watts", "joules_per_request")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="tier0_encoder_onnx_int8")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--expect-artefacts", type=int, default=3)
    ap.add_argument("--expect-runs", type=int, default=3)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="filenames to exclude; requires --exclude-reason")
    ap.add_argument("--exclude-reason", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.exclude and not a.exclude_reason:
        raise SystemExit(
            "--exclude requires --exclude-reason. An unexplained exclusion is "
            "outlier-hunting; §3aj permits exclusion only for an identifiable, stated "
            "cause, applied symmetrically to fast and slow runs.")
    out = a.out or a.results_dir / f"bench_{a.tier}_aggregate.json"

    files = sorted(p for p in a.results_dir.glob(f"bench_{a.tier}_*_run*.json")
                   if p.name not in set(a.exclude))
    if not files:
        raise SystemExit(f"no bench_{a.tier}_*_run*.json in {a.results_dir}")

    by_artefact: dict[str, list[dict]] = {}
    for p in files:
        d = json.loads(p.read_text())
        if not d.get("complete_for_e6", True):
            raise SystemExit(
                f"{p.name} is marked complete_for_e6=false (--skip-power was used). "
                f"It cannot contribute to an E6 figure; re-run it with power sampling.")
        by_artefact.setdefault(d["artefact"], []).append(d)

    n_art, runs = len(by_artefact), {k: len(v) for k, v in by_artefact.items()}
    print(f"{len(files)} runs over {n_art} artefacts: "
          + ", ".join(f"{Path(k).name}x{v}" for k, v in runs.items()))
    if n_art != a.expect_artefacts or set(runs.values()) != {a.expect_runs}:
        print(f"  WARNING: expected {a.expect_artefacts} artefacts x {a.expect_runs} runs")

    summary = {}
    for f in FIELDS:
        allv = [d[f] for v in by_artefact.values() for d in v if d.get(f) is not None]
        if not allv:
            summary[f] = {"mean": None, "sd": None, "note": "not measured in any run"}
            continue
        means = [st.mean([d[f] for d in v]) for v in by_artefact.values()]
        within = [st.pstdev([d[f] for d in v]) for v in by_artefact.values() if len(v) > 1]
        s = {"mean": st.mean(allv), "sd": st.stdev(allv) if len(allv) > 1 else None,
             "n_runs": len(allv),
             "between_artefact_sd": st.stdev(means) if len(means) > 1 else None,
             "within_artefact_sd_pooled": st.mean(within) if within else None,
             "per_artefact_mean": {Path(k).name: st.mean([d[f] for d in v])
                                   for k, v in by_artefact.items()}}
        # `is not None`, NOT truthiness: a between-artefact sd of exactly 0.0 is the
        # BEST possible outcome for the weight-independence check, and truthiness would
        # silently drop the ratio in precisely that case. Zero within-artefact sd is
        # guarded separately because it makes the ratio undefined rather than zero.
        if (s["between_artefact_sd"] is not None
                and s["within_artefact_sd_pooled"] is not None):
            w = s["within_artefact_sd_pooled"]
            s["between_over_within"] = (s["between_artefact_sd"] / w) if w > 0 else None
        summary[f] = s

    print(f"\n{'field':22s} {'mean':>10s} {'sd(all)':>9s} {'between':>9s} {'within':>9s} {'b/w':>6s}")
    for f, s in summary.items():
        if s.get("mean") is None:
            print(f"{f:22s} {'not measured':>10s}"); continue
        bw = s.get("between_over_within")
        print(f"{f:22s} {s['mean']:10.4f} "
              f"{(s['sd'] if s['sd'] is not None else float('nan')):9.4f} "
              f"{(s['between_artefact_sd'] or float('nan')):9.4f} "
              f"{(s['within_artefact_sd_pooled'] or float('nan')):9.4f} "
              f"{(bw if bw is not None else float('nan')):6.2f}")

    bw = summary.get("throughput_rps", {}).get("between_over_within")
    if bw is not None:
        print(f"\nWEIGHT-INDEPENDENCE CHECK (§3aa): between/within throughput sd = {bw:.2f}")
        print("  ~1 or below => throughput does not depend on the weights, as §3aa argues"
              if bw <= 2.0 else
              "  >2 => between-artefact spread EXCEEDS run-to-run noise. §3aa's "
              "weight-independence claim is contradicted; do not update costs.yaml "
              "until this is explained.")

    payload = {"tier": a.tier, "n_runs": len(files), "n_artefacts": n_art,
               "runs_per_artefact": {Path(k).name: v for k, v in runs.items()},
               "excluded": a.exclude, "exclude_reason": a.exclude_reason,
               "rule": "PREREGISTRATION 3aj: mean over ALL runs; no outlier exclusion",
               "fields": summary}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
