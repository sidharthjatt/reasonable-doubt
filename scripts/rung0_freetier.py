"""RUNG 0 — free-tier validation. Costs $0.

Runs the zero-shot prompt against a free provider (Gemini or Groq) on the first N rows
of `test_3000`, to prove the prompt produces parseable, in-vocabulary output BEFORE any
paid request is sent.

    python scripts/rung0_freetier.py --provider groq --model <model-id> --n 100

Gate: if JSON parse rate is below the threshold (default 98%), the prompt needs work
and we do NOT proceed to Rung 1. The script exits non-zero in that case.

No Claude spend. Responses are cached to disk by the standard four-part key, so a
re-run costs nothing and re-serves from cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from src.api.cache import ResponseCache  # noqa: E402
from src.api.providers import get_provider  # noqa: E402
from src.data.labels import FormatFailureCounter, LabelNormalizer, load_labels  # noqa: E402
from src.data.loading import get_split, label_names, load_ledgar  # noqa: E402
from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest, verify_manifest  # noqa: E402
from src.data.prompts import render_zeroshot  # noqa: E402
from src.data.schema import ParseFailure, parse_response  # noqa: E402
from src.eval.metrics import score  # noqa: E402

ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}
ENV_KEYS = {
    "groq": ("GROQ_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def api_key(provider: str) -> str:
    for name in ENV_KEYS[provider]:
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit(
        f"No API key for {provider}. Set one of {', '.join(ENV_KEYS[provider])} "
        "in the environment or in .env (which is gitignored)."
    )


def call_groq(client, model, system, clause, params):
    r = client.post(
        ENDPOINTS["groq"],
        headers={"Authorization": f"Bearer {api_key('groq')}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": clause},
            ],
            **params,
        },
    )
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"]["content"], body.get("usage")


def call_gemini(client, model, system, clause, params):
    r = client.post(
        ENDPOINTS["gemini"].format(model=model),
        headers={"x-goog-api-key": api_key("gemini")},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": clause}]}],
            "generationConfig": {
                "temperature": params.get("temperature", 0.0),
                "maxOutputTokens": params.get("max_tokens", 128),
            },
        },
    )
    r.raise_for_status()
    body = r.json()
    text = body["candidates"][0]["content"]["parts"][0]["text"]
    return text, body.get("usageMetadata")


CALLERS = {"groq": call_groq, "gemini": call_gemini}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=sorted(CALLERS), required=True)
    ap.add_argument("--model", required=True, help="provider model id")
    ap.add_argument("--n", type=int, default=100, help="first N rows of test_3000")
    ap.add_argument("--manifest", default="test_3000")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min-parse-rate", type=float, default=0.98)
    ap.add_argument("--out", type=Path, default=Path("results/rung0_freetier.json"))
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    args = ap.parse_args()

    print(f"RUNG 0 — {args.provider}/{args.model}, first {args.n} rows of {args.manifest}")
    print("Free tier. This script cannot spend Claude budget.\n")

    ds = load_ledgar()
    manifest = load_manifest(args.manifest, DEFAULT_MANIFEST_DIR)
    verify_manifest(manifest, ds, DEFAULT_MANIFEST_DIR)  # every run verifies first
    print(f"manifest verified: {manifest.name} sha256={manifest.text_sha256[:16]}…")

    names = label_names(ds)
    normalizer = LabelNormalizer(load_labels())
    split = get_split(ds, manifest.split)
    rows = manifest.indices[: args.n]
    data = split.select(rows)
    clauses = list(data["text"])
    gold = [names[int(i)] for i in data["label"]]

    prompt = render_zeroshot(names)
    print(f"prompt: {prompt.template_name} sha256={prompt.sha256[:16]}…\n")

    provider = get_provider(args.provider)
    cache = ResponseCache()
    params = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    caller = CALLERS[args.provider]
    counter = FormatFailureCounter(max_samples=args.n)

    predictions: list[str | None] = []
    parse_failures: list[dict] = []
    unmatched: list[dict] = []
    confidences: list[float] = []

    with httpx.Client(timeout=90.0) as client:
        for i, (row, clause) in enumerate(zip(rows, clauses)):
            key = provider.cache_key(
                args.model,
                system=prompt.text,
                messages=[{"role": "user", "content": clause}],
                params=params,
            )
            entry = cache.get_or_none(key)
            if entry is not None:
                raw, usage = entry.response, entry.usage
            else:
                raw, usage = caller(client, args.model, prompt.text, clause, params)
                cache.put(key, raw, usage=usage, request_params=params)
                if args.sleep:
                    time.sleep(args.sleep)

            try:
                parsed = parse_response(raw)
            except ParseFailure as exc:
                parse_failures.append({"row": row, "raw": raw, "error": str(exc)})
                predictions.append(counter.record(raw, None))
                continue

            confidences.append(parsed.confidence)
            matched = normalizer.normalize(parsed.label)
            if matched is None:
                unmatched.append({"row": row, "raw_label": parsed.label, "raw": raw})
            predictions.append(counter.record(parsed.label, matched))

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)} done")

    n = len(predictions)
    n_parse_ok = n - len(parse_failures)
    parse_rate = n_parse_ok / n
    report = score(gold, predictions, labels=sorted(set(gold)))

    print("\n" + "=" * 64)
    print(f"JSON parse success rate   : {parse_rate:.2%}  ({n_parse_ok}/{n})")
    print(f"label match rate          : {1 - counter.failure_rate:.2%}")
    print(f"  parse failures          : {len(parse_failures)}")
    print(f"  parsed but out-of-vocab : {len(unmatched)}")
    print("-" * 64)
    print(report.render())
    if confidences:
        conf = sorted(confidences)
        print(
            f"  confidence p50/p90      : {conf[len(conf) // 2]:.2f} / "
            f"{conf[int(0.9 * len(conf))]:.2f}   (zero-shot: unanchored, see C2)"
        )
    print("=" * 64)

    if parse_failures or unmatched:
        print("\nEVERY FAILURE, VERBATIM:\n")
        for f in parse_failures:
            print(f"--- row {f['row']} — PARSE FAILURE ({f['error']})")
            print(f"{f['raw']!r}\n")
        for f in unmatched:
            print(f"--- row {f['row']} — OUT OF VOCABULARY: {f['raw_label']!r}")
            print(f"{f['raw']!r}\n")
    else:
        print("\nNo parse failures and no out-of-vocabulary labels.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "rung": 0,
                "provider": args.provider,
                "model": args.model,
                "cost_usd": 0.0,
                "manifest": manifest.name,
                "manifest_sha256": manifest.text_sha256,
                "prompt_sha256": prompt.sha256,
                "n": n,
                "parse_success_rate": parse_rate,
                "label_match_rate": 1 - counter.failure_rate,
                "metrics": report.as_dict(),
                "parse_failures": parse_failures,
                "out_of_vocabulary": unmatched,
                "cache_stats": cache.cache_stats().as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")

    if parse_rate < args.min_parse_rate:
        print(
            f"\nGATE FAILED: parse rate {parse_rate:.2%} is below "
            f"{args.min_parse_rate:.0%}. The prompt needs work. DO NOT proceed to "
            "Rung 1 (paid).",
            file=sys.stderr,
        )
        return 1
    print(f"\nGATE PASSED: parse rate >= {args.min_parse_rate:.0%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
