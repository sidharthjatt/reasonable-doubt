"""Submit a Claude batch. Spends money; gated by --confirm and the hard stop.

    python scripts/submit_batch.py --run rung2                # 10-row smoke test
    python scripts/submit_batch.py --run stage1  --confirm    # 3000 x 2 models
    python scripts/submit_batch.py --run fewshot --confirm    # C2/C4 probe, Sonnet

Submits and exits. Batches are bounded at 24h; poll separately with
`scripts/poll_batch.py`. The batch id is persisted the instant it is known, so a
crash cannot cause a resubmit (which is how a batch run double-spends).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.batch import CACHE_TTL, BatchClient, build_batch_requests  # noqa: E402
from src.api.cost import load_rate_card  # noqa: E402
from src.api.env import load_env  # noqa: E402
from src.api.ledger import (  # noqa: E402
    SpendLedger, estimate_run_cost_bracket, require_confirmation,
)
from src.data.loading import get_split, label_names, load_ledgar  # noqa: E402
from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest, verify_manifest  # noqa: E402
from src.data.prompts import render_fewshot, render_zeroshot  # noqa: E402

H, S = "claude-haiku-4-5-20251001", "claude-sonnet-5"
# Measured, 600-row paced count_tokens sample (see PREREGISTRATION 3i).
MEAN_TOK = {(H, "zs"): 938.6, (S, "zs"): 1303.2, (S, "fs"): 2925.2}
SD_TOK = {(H, "zs"): 137.0, (S, "zs"): 204.2, (S, "fs"): 204.2}
SAMPLE_N = 600

# `tag` disambiguates custom_ids across legs. Stage 1 runs two models over the SAME
# 3000 rows, so a row-index-only id collides; the tag makes each leg's ids distinct
# while leaving the row index recoverable by parse_custom_id.
RUNS = {
    "rung2":   [{"model": H, "prompt": "zs", "n": 10, "tag": "r2h"},
                {"model": S, "prompt": "zs", "n": 10, "tag": "r2s"}],
    "stage1":  [{"model": H, "prompt": "zs", "n": 3000, "tag": "s1h"},
                {"model": S, "prompt": "zs", "n": 3000, "tag": "s1s"}],
    "fewshot": [{"model": S, "prompt": "fs", "n": 1000, "tag": "fss"}],
}


class SDKBatch:
    def __init__(self, client): self.c = client
    def create_batch(self, requests): return self.c.messages.batches.create(requests=requests)
    def retrieve_batch(self, bid): return self.c.messages.batches.retrieve(bid)
    def batch_results(self, bid): return self.c.messages.batches.results(bid)


def upper_bound(model, prompt, n, pop=3000):
    """One-sided mean+2SE bound with finite-population correction (rule 8 gating)."""
    import math
    sd, mean = SD_TOK[(model, prompt)], MEAN_TOK[(model, prompt)]
    se = sd / math.sqrt(SAMPLE_N) * math.sqrt((pop - SAMPLE_N) / (pop - 1))
    return mean + 2 * se


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", choices=sorted(RUNS), required=True)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--manifest", default="test_3000")
    args = ap.parse_args()

    load_env()
    import anthropic

    card = load_rate_card()
    OUT = card["budget"]["stages"]["stage_1"]["max_output_tokens"]
    ds = load_ledgar()
    man = load_manifest(args.manifest, DEFAULT_MANIFEST_DIR)
    verify_manifest(man, ds, DEFAULT_MANIFEST_DIR)   # every run verifies first
    names = label_names(ds)
    split = get_split(ds, man.split)
    prompts = {"zs": render_zeroshot(names),
               "fs": render_fewshot(names, ds["train"], n_exemplars=8)}

    ledger = SpendLedger()
    legs, all_reqs, est_total_gate, est_total_opt = [], [], 0.0, 0.0

    print("=" * 74)
    print(f"BATCH SUBMISSION — {args.run}")
    print(f"manifest {man.name} sha={man.text_sha256[:16]}… VERIFIED")
    print("=" * 74)

    for leg in RUNS[args.run]:
        model, ptag, n = leg["model"], leg["prompt"], leg["n"]
        rows = man.indices[:n]
        texts = list(split.select(rows)["text"])
        p = prompts[ptag]
        out_budget = OUT[model]
        reqs = build_batch_requests(
            man.name, rows, texts, model=model, system_prompt=p.text,
            # temperature is OMITTED for BOTH models. Sonnet 5 rejects it
            # ("`temperature` is deprecated for this model"); Haiku 4.5 accepts it.
            # Setting it on one leg and not the other would make the two legs differ
            # by an uncontrolled parameter — and comparing them is the point.
            max_tokens=out_budget, temperature=None, cache_ttl=CACHE_TTL,
            prefix=f"{leg['tag']}_",
            extra_params={"thinking": {"type": "disabled"}},
        )
        ub = upper_bound(model, ptag, n)
        est = estimate_run_cost_bracket(
            f"{args.run}_{model}", provider="anthropic", model=model, n_requests=n,
            system_tokens=0, per_request_input_tokens=[ub] * n,
            expected_output_tokens=out_budget, batch=True, cache_ttl=CACHE_TTL,
            token_source=f"count_tokens, {SAMPLE_N}-row sample, mean+2SE bound",
        )
        est_total_gate += est.gating_usd
        est_total_opt += est.optimistic_usd
        legs.append((model, ptag, n, out_budget, est, p, leg["tag"]))
        all_reqs.extend(reqs)
        print(f"  {model:<28} {ptag}  n={n:<5} out={out_budget:<4} "
              f"prompt={p.sha256[:12]}…  gating ${est.gating_usd:.4f}")

    print(f"\n  requests in batch : {len(all_reqs):,} (API max 10,000)")
    print(f"  cache_control     : ttl={CACHE_TTL} on the system block")
    print(f"  COMBINED GATING   : ${est_total_gate:.4f}   (optimistic ${est_total_opt:.4f})")

    combined = legs[0][4]
    combined.pessimistic.estimated_usd = est_total_gate
    combined.optimistic.estimated_usd = est_total_opt
    combined.pessimistic.run_id = combined.optimistic.run_id = args.run
    combined.pessimistic.n_requests = combined.optimistic.n_requests = len(all_reqs)

    print()
    require_confirmation(combined, ledger, confirm=args.confirm)

    client = BatchClient(SDKBatch(anthropic.Anthropic()), state_dir=Path("results/batches"))
    state = client.submit(args.run, all_reqs, model=",".join(l[0] for l in legs),
                          manifest_name=man.name, cache_ttl=CACHE_TTL)
    print(f"  SUBMITTED. batch_id={state.batch_id}")
    print(f"  state persisted -> {state.path(Path('results/batches'))}")

    ledger.record_estimate(combined.pessimistic)
    meta = Path("results/batches") / f"{args.run}_meta.json"
    meta.write_text(json.dumps({
        "run": args.run, "batch_id": state.batch_id, "manifest": man.name,
        "manifest_sha256": man.text_sha256, "n_requests": len(all_reqs),
        "cache_ttl": CACHE_TTL,
        "legs": [{"model": m, "prompt": t, "n": n, "max_output_tokens": o,
                  "custom_id_tag": tg,
                  "prompt_sha256": p.sha256, "gating_usd": e.gating_usd}
                 for m, t, n, o, e, p, tg in legs],
        "gating_usd_total": est_total_gate, "optimistic_usd_total": est_total_opt,
    }, indent=2))
    print(f"  metadata -> {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
