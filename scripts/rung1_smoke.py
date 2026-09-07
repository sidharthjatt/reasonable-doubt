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
MAX_TOKENS = 64          # the JSON answer is ~25 tokens

# anthropic SDK 1.4.0 dropped `temperature` from the Messages.create SIGNATURE, so
# passing it as a kwarg raises TypeError client-side. The API still accepts and
# VALIDATES it (range 0..1, type-checked; unknown fields are rejected outright), so it
# is pinnable via extra_body. Verified 2026-09-07 by raw HTTP. Note that temperature 0
# is not a determinism guarantee — seed variance is still measured (hard rule 2).
TEMPERATURE = 0.0

# Extended thinking is EXPLICITLY DISABLED rather than left to a server-side default.
# Rung 0 showed twice what happens when a reasoning budget eats the output budget:
# gpt-oss returned empty content at 128 tokens, Qwen still truncated at 512. On a paid
# model a truncated response still bills for the output tokens it burned. The SDK omits
# this key by default, but the SDK also exposes an "adaptive" thinking mode, so we state
# our intent in the payload instead of relying on what the default happens to be.
THINKING = {"type": "disabled"}


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
    print(f"batch         : NO   caching: NO   max_tokens: {MAX_TOKENS}")
    print(f"temperature   : {TEMPERATURE} — pinned via extra_body (SDK 1.4.0 dropped the "
          "kwarg; the API still validates it)")
    print(f"thinking      : {THINKING['type'].upper()} — stated explicitly in the payload, "
          "not inherited from a default")
    print(f"ledger        : {ledger.path}")
    print(f"recorded spend: ${ledger.cumulative_usd():,.4f}   "
          f"hard stop ${ledger.hard_stop_usd:,.2f}")

    # ------------------------------------------------------- exact request payload
    def build_params(model: str, clause: str) -> dict:
        return {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "thinking": THINKING,
            "system": prompt.text,
            "messages": [{"role": "user", "content": clause}],
            # SDK 1.4.0 has no `temperature` kwarg; the API does. See TEMPERATURE above.
            "extra_body": {"temperature": TEMPERATURE},
        }

    example = build_params(MODELS[0], clauses[0])
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
    print(f"Proceeding. Projected cumulative after this rung: ${projected:,.4f}\n")

    # ---------------------------------------------------------------- send
    normalizer = LabelNormalizer(load_labels())
    results: dict[str, dict] = {}

    for model in MODELS:
        print("=" * 72)
        print(f"SENDING — {model}")
        print("=" * 72)
        usage_total = None
        rows_out = []

        for i, (row, clause) in enumerate(zip(rows, clauses)):
            params = build_params(model, clause)
            key = provider.cache_key(
                model, system=prompt.text,
                messages=params["messages"],
                params={"max_tokens": params["max_tokens"],
                        "thinking": params["thinking"],
                        "temperature": TEMPERATURE},
            )
            entry = cache.get_or_none(key)
            if entry is not None:
                body, usage_block = entry.response, entry.usage
            else:
                message = client.messages.create(**params)
                body = message.model_dump()
                usage_block = body["usage"]
                # Cache BEFORE use (hard rule 3): a crash after the call must never
                # cause the same prompt to be paid for twice.
                cache.put(key, body, usage=usage_block, request_params=params)

            # Thinking must be off. Verify against the RESPONSE, not our intent.
            blocks = [b.get("type") for b in body.get("content", [])]
            if any(b in ("thinking", "redacted_thinking") for b in blocks):
                raise RuntimeError(
                    f"{model} row {row}: response contains a thinking block "
                    f"{blocks!r} despite thinking={THINKING}. Stop and investigate — "
                    "a reasoning budget will truncate answers and bill for it."
                )
            if body.get("stop_reason") == "max_tokens":
                print(f"  WARNING row {row}: stop_reason=max_tokens — answer truncated "
                      f"at the {MAX_TOKENS}-token budget, and billed for it.")

            u = provider.parse_usage(usage_block, model=model)
            usage_total = u if usage_total is None else usage_total + u

            text = "".join(b.get("text", "") for b in body.get("content", [])
                           if b.get("type") == "text")
            try:
                parsed = parse_response(text)
                label = normalizer.normalize(parsed.label)
                conf = parsed.confidence
            except ParseFailure:
                label, conf = None, None

            rows_out.append({
                "row": row, "gold": gold[i], "predicted": label,
                "confidence": conf, "raw": text,
                "stop_reason": body.get("stop_reason"),
                "usage": u.as_dict(),
                "predicted_input_tokens": per_model_tokens[model]["per_request_input_tokens"][i],
            })
            print(f"  [{i + 1:2d}/{len(rows)}] row {row:<4} gold={gold[i]:<24} "
                  f"pred={str(label):<24} conf={conf}")

        est = estimates[model]
        entry = ledger.record_actual(
            f"rung1_{model}", usage_total, provider="anthropic", model=model,
            batch=False, cache_ttl="5m", estimated_usd=est.gating_usd,
            n_requests=len(rows), notes="Rung 1 smoke test; no batch, no caching",
        )
        results[model] = {
            "usage": usage_total.as_dict(),
            "estimated_usd": est.gating_usd,
            "actual_usd": entry.cost_usd,
            "predicted_input_tokens": per_model_tokens[model]["total_input_tokens"],
            "rows": rows_out,
        }
        print(f"\n  estimated ${est.gating_usd:.6f}  ->  ACTUAL ${entry.cost_usd:.6f}"
              f"  ({entry.cost_usd / est.gating_usd - 1:+.1%})")
        print(f"  cumulative spend: ${ledger.cumulative_usd():.6f}\n")

    # ---------------------------------------------------- reconciliation report
    print("=" * 72)
    print("RECONCILIATION")
    print("=" * 72)
    print(f"{'model':<28} {'pred in':>9} {'actual in':>10} {'delta':>8} "
          f"{'est $':>9} {'actual $':>9}")
    for model in MODELS:
        r = results[model]
        pred, act = r["predicted_input_tokens"], r["usage"]["input_tokens"]
        print(f"{model:<28} {pred:>9,} {act:>10,} {act / pred - 1:>+7.1%} "
              f"{r['estimated_usd']:>9.6f} {r['actual_usd']:>9.6f}")

    ha, so = (results[m]["usage"]["input_tokens"] for m in MODELS)
    print(f"\nTOKENIZER DELTA, actual billed input on identical text: "
          f"{so / ha - 1:+.1%} ({so:,} vs {ha:,})")
    print(f"\nTOTAL SPENT THIS RUNG: ${sum(r['actual_usd'] for r in results.values()):.6f}")
    print(f"CUMULATIVE           : ${ledger.cumulative_usd():.6f} of "
          f"${ledger.hard_stop_usd:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "rung": 1, "manifest": manifest.name,
        "manifest_sha256": manifest.text_sha256,
        "prompt_sha256": prompt.sha256, "rows": rows,
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        "temperature_note": "pinned via extra_body; SDK 1.4.0 dropped the kwarg",
        "thinking": THINKING,
        "batch": False, "caching": False, "models": results,
        "cache_stats": cache.cache_stats().as_dict(),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
