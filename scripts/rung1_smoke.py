"""RUNG 1 — paid smoke test. ~$0.01. Synchronous, no batch, no caching.

The smallest possible real transaction. The purpose is NOT accuracy — it is to verify
the whole money path end to end:

  * the four usage fields parse (hard rule 10)
  * count_tokens vs the actual usage.input_tokens — how close is our pre-flight estimate?
  * compute_cost against the returned usage reconciles with the Console
  * the ledger records estimate AND actual
  * the cache stores and re-serves without a second call

Run on Haiku 4.5 first, then the SAME 20 rows on Sonnet 5, so the two models' token
counts for identical text can be compared directly. That comparison is the measured
justification for hard rule 9 and belongs in the final report.

    python scripts/rung1_smoke.py                 # dry run: prints everything, sends nothing
    python scripts/rung1_smoke.py --confirm       # actually spends

Deliberately NOT batch and NOT cached: this rung isolates the billing path. Caching and
batching are Rung 2's job, and mixing them here would make an unexpected cost
ambiguous between three causes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.cache import ResponseCache  # noqa: E402
from src.api.cost import compute_cost, load_rate_card  # noqa: E402
from src.api.env import load_env  # noqa: E402
from src.api.ledger import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    SpendLedger,
    estimate_run_cost_bracket,
    require_confirmation,
)
from src.api.providers import get_provider  # noqa: E402
from src.api.tokens import TokenCounter  # noqa: E402
from src.data.labels import LabelNormalizer, load_labels  # noqa: E402
from src.data.loading import get_split, label_names, load_ledgar  # noqa: E402
from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest, verify_manifest  # noqa: E402
from src.data.prompts import render_zeroshot  # noqa: E402
from src.data.schema import ParseFailure, parse_response  # noqa: E402

# Haiku FIRST: it is the cheaper model, so if the money path is broken we find out on
# the cheaper half. Same 20 rows for both, so token counts are directly comparable.
MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5"]
MAX_TOKENS = 64          # the JSON answer is ~25 tokens; no reasoning block on Claude
TEMPERATURE = 0.0


class AnthropicCounterAdapter:
    """Adapts the Anthropic SDK to the TokenCounter protocol."""

    def __init__(self, client):
        self._client = client

    def count_tokens(self, *, model, system, messages) -> int:
        return self._client.messages.count_tokens(
            model=model, system=system, messages=messages
        ).input_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true",
                    help="Actually send. Without this, nothing is sent.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--manifest", default="test_3000")
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    ap.add_argument("--out", type=Path, default=Path("results/rung1_smoke.json"))
    args = ap.parse_args()

    load_env()
    import anthropic

    client = anthropic.Anthropic()
    counter = TokenCounter(AnthropicCounterAdapter(client), provider="anthropic")
    ledger = SpendLedger(args.ledger)
    cache = ResponseCache()
    provider = get_provider("anthropic")
    rate_card = load_rate_card()

    # ---------------------------------------------------------------- selection
    ds = load_ledgar()
    manifest = load_manifest(args.manifest, DEFAULT_MANIFEST_DIR)
    verify_manifest(manifest, ds, DEFAULT_MANIFEST_DIR)
    names = label_names(ds)
    split = get_split(ds, manifest.split)
    rows = manifest.indices[: args.n]
    data = split.select(rows)
    clauses = list(data["text"])
    gold = [names[int(i)] for i in data["label"]]
    prompt = render_zeroshot(names)

    print("=" * 72)
    print("RUNG 1 — paid smoke test")
    print("=" * 72)
    print(f"manifest      : {manifest.name} (sha256 {manifest.text_sha256[:16]}…) VERIFIED")
    print(f"split         : {manifest.split}")
    print(f"rows          : first {len(rows)} of the manifest = dataset indices")
    print(f"                {rows}")
    print(f"prompt        : {prompt.template_name} sha256={prompt.sha256[:16]}…")
    print(f"models        : {' then '.join(MODELS)} (same rows, same prompt)")
    print(f"batch         : NO   caching: NO   temperature: {TEMPERATURE}   "
          f"max_tokens: {MAX_TOKENS}")
    print(f"ledger        : {ledger.path}")
    print(f"recorded spend: ${ledger.cumulative_usd():,.4f}   "
          f"hard stop ${ledger.hard_stop_usd:,.2f}")

    # ------------------------------------------------------- exact request payload
    example = {
        "model": MODELS[0],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": prompt.text,
        "messages": [{"role": "user", "content": clauses[0]}],
    }
    print("\n" + "-" * 72)
    print("EXACT REQUEST PAYLOAD (row 1 of 20, system prompt elided for length)")
    print("-" * 72)
    shown = dict(example, system=f"<{len(prompt.text)} chars — zeroshot "
                                 f"sha256 {prompt.sha256[:16]}…>")
    print(json.dumps(shown, indent=2)[:2000])
    print("\nsystem prompt, first 300 chars:")
    print(prompt.text[:300] + "…")
    print("\nsystem prompt, last 200 chars:")
    print("…" + prompt.text[-200:])

    # ------------------------------------------------------------- token counting
    # Hard rule 9: per-model counts, from each model's own endpoint. Never shared.
    print("\n" + "-" * 72)
    print("PRE-FLIGHT TOKEN COUNTS (each model's own count_tokens endpoint — free)")
    print("-" * 72)
    # Counts are for the FULL request (system prompt + clause), which is exactly what
    # is billed on this uncached rung. The system prompt is not counted separately:
    # count_tokens rejects an empty user message, and subtracting one measurement from
    # another to isolate the prefix would be arithmetic dressed as a measurement. The
    # cacheable-prefix size is measured directly at Rung 2, where caching makes it
    # meaningful.
    per_model_tokens: dict[str, dict] = {}
    for model in MODELS:
        per_request = [
            counter.count_tokens(model, [{"role": "user", "content": c}], system=prompt.text)
            for c in clauses
        ]
        per_model_tokens[model] = {
            "per_request_input_tokens": per_request,
            "total_input_tokens": sum(per_request),
            "min_request_tokens": min(per_request),
            "max_request_tokens": max(per_request),
        }
        print(f"  {model}")
        print(f"    total input tokens  : {sum(per_request):,}  "
              f"(mean {sum(per_request) / len(per_request):,.0f}/row, "
              f"min {min(per_request):,}, max {max(per_request):,})")

    a, b = MODELS
    ta, tb = (per_model_tokens[m]["total_input_tokens"] for m in (a, b))
    print(f"\n  TOKENIZER DELTA on identical text: {b} counts "
          f"{(tb / ta - 1):+.1%} vs {a}  ({tb:,} vs {ta:,})")
    print("  This is hard rule 9 measured rather than assumed.")

    # ------------------------------------------------------------------ estimate
    print("\n" + "-" * 72)
    print("COST ESTIMATE — all rates from configs/costs.yaml")
    print("-" * 72)
    estimates = {}
    total_gating = 0.0
    for model in MODELS:
        t = per_model_tokens[model]
        est = estimate_run_cost_bracket(
            f"rung1_{model}",
            provider="anthropic",
            model=model,
            n_requests=len(rows),
            system_tokens=0,  # no caching on this rung; prefix billed per request
            per_request_input_tokens=t["per_request_input_tokens"],
            expected_output_tokens=MAX_TOKENS,
            batch=False,
            cache_ttl="5m",
            token_source="anthropic count_tokens endpoint",
        )
        estimates[model] = est
        total_gating += est.gating_usd
        rates = rate_card["providers"]["anthropic"]["models"][model]
        print(f"\n  {model}  (${rates['input']}/${rates['output']} per MTok)")
        print(f"    input tokens  : {t['total_input_tokens']:,}")
        print(f"    output budget : {MAX_TOKENS * len(rows):,} "
              f"({MAX_TOKENS} x {len(rows)}, worst case)")
        print(f"    cost          : ${est.gating_usd:,.6f}")

    print(f"\n  COMBINED WORST-CASE COST : ${total_gating:,.6f}")
    print("  (worst case because every request is assumed to emit the full "
          f"{MAX_TOKENS}-token budget; real answers are ~25 tokens)")

    # ------------------------------------------------------------ gate and send
    combined = estimates[MODELS[0]]
    combined.pessimistic.estimated_usd = total_gating
    combined.pessimistic.run_id = "rung1_combined"
    combined.optimistic.estimated_usd = sum(e.optimistic_usd for e in estimates.values())

    print("\n" + "=" * 72)
    projected = require_confirmation(combined, ledger, confirm=args.confirm)
    print(f"Proceeding. Projected cumulative after this rung: ${projected:,.4f}")

    # ... send path deliberately below the gate; unreachable without --confirm.
    raise SystemExit(
        "\nSEND PATH NOT YET IMPLEMENTED — this script currently stops at the gate by "
        "design. Rung 1 execution is added once the payload above has been reviewed."
    )


if __name__ == "__main__":
    raise SystemExit(main())
