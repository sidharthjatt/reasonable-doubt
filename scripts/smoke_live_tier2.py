"""LIVE smoke test for Tier 2 — the FIRST REAL CLAUDE CALL this service has ever made.

    python scripts/smoke_live_tier2.py              # estimate only, sends nothing
    python scripts/smoke_live_tier2.py --confirm    # actually sends

Everything before this has run against a stubbed client. This sends 3 real escalations
through the deployed `Tier2Claude`, built from `configs/serve.yaml`, so the cap, the
response cache, the ledger write and the usage parsing are all exercised as they will be
in production rather than as tests configure them.

WHICH ROWS, AND WHY IT MATTERS. The clauses come from `configs/canary_200.json` — TRAIN
rows that carry NO evaluation role (they exclude `train_holdout_3000` and `exemplars_8`,
and train is neither the reporting split nor the threshold split). A smoke test must not
spend a reporting row's first-ever API call on a throwaway run, and hard rule 1 keeps
`dev_2000` off limits entirely. The split is asserted, not assumed.

By default the rows are the ones Tier 0 is LEAST confident about, scored through the real
encoder and ranked by the deployed margin signal, so these are clauses the router would
genuinely escalate. That makes it a smoke test of the deployed path rather than of an
arbitrary request. `--no-tier0` skips the encoder load and takes the first N rows instead.

HARD RULE 8. This script prints an itemised estimate, gates on the PESSIMISTIC bound, and
refuses to run if recorded spend + this run's estimate would breach
`budget.hard_stop_usd`. That check runs BEFORE the `--confirm` check, so a run that would
breach the budget is refused even with `--confirm`.

HARD RULE 9. Token counts come from Sonnet's own `count_tokens` endpoint (free), never a
local tokenizer, and are cached keyed by (model, prompt_hash).

⚠ THE ESTIMATE-ONLY RUN IS NOT INERT. Without `--confirm` no message is sent and nothing
is billed, but the pre-flight still calls `count_tokens` for real, which needs the key and
reaches Anthropic. `load_env()` also reads the project `.env`, so a shell with no
ANTHROPIC_API_KEY exported is NOT a dry-run guarantee. Those calls are free, write no
ledger entry, and are cached to `cache/token_counts/`, so a later run reuses them — but if
you need a genuinely offline preview, run with the key removed from `.env` as well.

A SECOND RUN OF THIS SCRIPT NORMALLY COSTS $0. The on-disk response cache is keyed by
(provider, model, prompt, params), so identical clauses are served from disk and write no
ledger entry. The per-response `cache_hit` column says which happened, and the ledger
delta at the end is the money actually spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.env import load_env  # noqa: E402
from src.api.ledger import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    SpendLedger,
    estimate_run_cost_bracket,
    require_confirmation,
)
from src.api.tokens import TokenCounter  # noqa: E402
from src.data.labels import load_labels  # noqa: E402
from src.data.loading import get_split, label_names, load_ledgar  # noqa: E402
from src.data.prompts import render_zeroshot  # noqa: E402
from src.serve.canary import load_canary, verify_canary  # noqa: E402
from src.serve.config import ServiceConfig  # noqa: E402
from src.serve.pricing import api_usd_for_result  # noqa: E402
from src.serve.tier2 import SpendCapReached, Tier2Claude, Tier2Unavailable  # noqa: E402


class AnthropicCounterAdapter:
    """Adapts the Anthropic SDK to the TokenCounter protocol."""

    def __init__(self, client):
        self._client = client

    def count_tokens(self, *, model, system, messages) -> int:
        return self._client.messages.count_tokens(
            model=model, system=system, messages=messages
        ).input_tokens


def pick_clauses(config: ServiceConfig, n: int, *, use_tier0: bool):
    """Return (texts, note). Prefers rows the router would actually escalate."""
    ds = load_ledgar()
    canary = load_canary(config.canary_row_set)
    verify_canary(canary, ds)
    if canary.split != "train":
        raise SystemExit(
            f"canary rows are on split {canary.split!r}, not train. Refusing to send "
            f"evaluation rows to the API for a smoke test.")

    rows = get_split(ds, canary.split).select(canary.indices)
    texts = list(rows["text"])
    if not use_tier0:
        return texts[:n], "first N canary rows (--no-tier0; NOT margin-selected)"

    from src.serve.tier0 import Tier0Encoder

    print(f"  scoring {len(texts)} canary rows through Tier 0 to find real escalations…")
    t0 = Tier0Encoder(model_dir=config.tier0_model_dir, labels=load_labels(),
                      max_length=config.tier0_max_length)
    scored = sorted(((t0.classify(t).confidence, t) for t in texts), key=lambda x: x[0])
    chosen = [t for _, t in scored[:n]]
    below = sum(1 for m, _ in scored[:n] if m < config.threshold.threshold)
    return chosen, (f"{n} LOWEST-margin canary rows (margins "
                    f"{', '.join(f'{m:.4f}' for m, _ in scored[:n])}; "
                    f"{below}/{n} below the deployed threshold "
                    f"{config.threshold.threshold:.6f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true",
                    help="Actually send. Without this, nothing is sent.")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--no-tier0", action="store_true",
                    help="skip the encoder load; use the first N canary rows instead")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    ap.add_argument("--out", type=Path, default=Path("results/smoke_live_tier2.json"))
    args = ap.parse_args()

    load_env()
    config = ServiceConfig.load()
    ledger = SpendLedger(args.ledger)

    print("=" * 74)
    print("TIER 2 LIVE SMOKE TEST — real Claude calls")
    print("=" * 74)
    print(f"model            : {config.tier2_model}")
    print(f"max_output_tokens: {config.tier2_max_output_tokens}")
    print(f"temperature      : {config.tier2_temperature!r}  (None = key OMITTED; "
          f"Sonnet 5 rejects it)")
    print(f"batch            : {config.tier2_batch}  (interactive; no 50% discount)")
    print(f"ledger           : {args.ledger}")
    print(f"serve spend cap  : ${config.tier2_spend_cap_usd:.2f}  "
          f"(tier2.spend_cap_usd, configs/serve.yaml)")

    if not Tier2Claude.api_key_present():
        raise SystemExit(
            "\nANTHROPIC_API_KEY is not set. Nothing to smoke test — the service would "
            "return escalation_skipped=true. Export the key and re-run.")
    print("ANTHROPIC_API_KEY: present (value never read, logged or persisted)")

    tier2 = Tier2Claude(
        model=config.tier2_model, labels=load_labels(),
        max_output_tokens=config.tier2_max_output_tokens,
        temperature=config.tier2_temperature, batch=config.tier2_batch,
        spend_cap_usd=config.tier2_spend_cap_usd, run_id=config.tier2_run_id,
        ledger=ledger)

    # ------------------------------------------------------------------- selection
    print("\n" + "-" * 74)
    print("CLAUSE SELECTION — TRAIN rows only, no evaluation role touched")
    print("-" * 74)
    clauses, note = pick_clauses(config, args.n, use_tier0=not args.no_tier0)
    print(f"  {note}")
    for i, c in enumerate(clauses, 1):
        print(f"    [{i}] {c[:90]!r}{'…' if len(c) > 90 else ''}")

    # ------------------------------------------------------------- token counts
    print("\n" + "-" * 74)
    print("PRE-FLIGHT TOKEN COUNTS — Sonnet's own count_tokens endpoint (free)")
    print("-" * 74)
    import anthropic

    client = anthropic.Anthropic()
    counter = TokenCounter(AnthropicCounterAdapter(client), provider="anthropic")
    system_text = render_zeroshot(load_labels()).text
    # The FULL request (system prompt + clause), which is what is billed on an uncached
    # call. The prefix is not counted separately: count_tokens rejects an empty user
    # message, and subtracting one measurement from another would be arithmetic dressed
    # as a measurement (the same choice scripts/rung1_smoke.py makes).
    per_request = [counter.count_tokens(
        config.tier2_model, [{"role": "user", "content": c}], system=system_text)
        for c in clauses]
    print(f"  per-request input tokens : {per_request}")
    print(f"  total                    : {sum(per_request):,}")
    print(f"  counter cache            : {counter.hits} hit(s), {counter.misses} miss(es)")

    # ----------------------------------------------------------------- estimate
    print("\n" + "-" * 74)
    print("COST ESTIMATE — every rate from configs/costs.yaml (hard rule 5)")
    print("-" * 74)
    estimate = estimate_run_cost_bracket(
        config.tier2_run_id, provider="anthropic", model=config.tier2_model,
        n_requests=len(clauses), system_tokens=0,
        per_request_input_tokens=per_request,
        expected_output_tokens=config.tier2_max_output_tokens,
        batch=config.tier2_batch, cache_ttl="1h")

    spent_before = ledger.cumulative_usd()
    print(f"\nLEDGER BEFORE : ${spent_before:.6f} cumulative "
          f"({len(ledger.entries())} entries)")
    print(f"  remaining to hard stop : ${ledger.remaining_usd():.4f} "
          f"of ${ledger.hard_stop_usd:.2f}")
    print(f"  serve cap headroom     : "
          f"${config.tier2_spend_cap_usd - spent_before:.6f}")

    # Gates on the PESSIMISTIC bound and refuses over the hard stop BEFORE it checks
    # --confirm, so a budget breach is refused even when confirmed (hard rule 8).
    require_confirmation(estimate, ledger, confirm=args.confirm)

    if tier2.cap_reached():
        raise SystemExit(
            f"\nREFUSING: the serve spend cap ${config.tier2_spend_cap_usd:.2f} is "
            f"already reached (${spent_before:.6f} recorded). Raise "
            f"tier2.spend_cap_usd deliberately, or let the service answer from Tier 0.")

    # ------------------------------------------------------------------- send
    print("\n" + "-" * 74)
    print(f"SENDING {len(clauses)} ESCALATION(S)")
    print("-" * 74)
    records = []
    for i, clause in enumerate(clauses, 1):
        try:
            r = tier2.classify(clause)
        except SpendCapReached as exc:
            print(f"\n  [{i}] STOPPED — {exc}")
            break
        except Tier2Unavailable as exc:
            print(f"\n  [{i}] UNAVAILABLE — {exc}")
            break

        cost = api_usd_for_result(config.tier2_model, r, batch=config.tier2_batch)
        u = r.usage
        print(f"\n  [{i}] label      : {r.label!r}")
        print(f"      confidence : {r.confidence}   (verbalized; expected miscalibrated)")
        print(f"      cache_hit  : {r.api_cache_hit}"
              + ("   <- served from disk, $0 spent, no ledger entry"
                 if r.api_cache_hit else "   <- real API call"))
        # Hard rule 10: the four fields reported SEPARATELY, never summed into one.
        print(f"      usage      : input_tokens                = {u.input_tokens}")
        print(f"                   cache_creation_input_tokens = "
              f"{u.cache_creation_input_tokens}")
        print(f"                   cache_read_input_tokens     = "
              f"{u.cache_read_input_tokens}")
        print(f"                   output_tokens               = {u.output_tokens}")
        print(f"      cost       : ${cost.usd:.6f}   ({cost.basis})")
        records.append({"clause_prefix": clause[:120], "label": r.label,
                        "confidence": r.confidence, "api_cache_hit": r.api_cache_hit,
                        "usage": u.as_dict(), "cost_usd": cost.usd})

    # ------------------------------------------------------------------ after
    spent_after = ledger.cumulative_usd()
    print("\n" + "=" * 74)
    print(f"LEDGER AFTER  : ${spent_after:.6f} cumulative "
          f"({len(ledger.entries())} entries)")
    print(f"DELTA         : ${spent_after - spent_before:.6f}  "
          f"<- real money spent by this run")
    print(f"  estimate was : ${estimate.gating_usd:.6f} pessimistic / "
          f"${estimate.optimistic_usd:.6f} optimistic")
    print(f"  serve cap    : ${config.tier2_spend_cap_usd:.2f}, "
          f"headroom now ${config.tier2_spend_cap_usd - spent_after:.6f}, "
          f"reached={tier2.cap_reached()}")
    print(f"  hard stop    : ${ledger.hard_stop_usd:.2f}, "
          f"remaining ${ledger.remaining_usd():.4f}")
    print("=" * 74)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "run_id": config.tier2_run_id,
        "model": config.tier2_model,
        "n_sent": len(records),
        "clause_source": note,
        "per_request_input_tokens_count_tokens": per_request,
        "estimate_pessimistic_usd": estimate.gating_usd,
        "estimate_optimistic_usd": estimate.optimistic_usd,
        "ledger_before_usd": spent_before,
        "ledger_after_usd": spent_after,
        "ledger_delta_usd": spent_after - spent_before,
        "spend_cap_usd": config.tier2_spend_cap_usd,
        "responses": records,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
