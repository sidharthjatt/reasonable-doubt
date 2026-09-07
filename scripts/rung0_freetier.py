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
from src.api.env import load_env  # noqa: E402
from src.api.redaction import redact  # noqa: E402
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


def retry_after_seconds(header: str | None, *, default: float = 2.0) -> float:
    """Parse a Retry-After header. It may be seconds OR an HTTP-date (RFC 9110).

    A bare ``float(header)`` crashes mid-run on the date form. Returns ``default``
    when the header is absent or unparseable — this is a RETRY POLICY, not a
    measurement, so a default here cannot corrupt any reported number.
    """
    if not header:
        return default
    try:
        return max(0.0, float(header))
    except ValueError:
        pass
    from email.utils import parsedate_to_datetime
    from datetime import datetime, timezone

    try:
        when = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return default
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class ShapeError(RuntimeError):
    """The provider response did not have the shape we parse. Never substituted with
    an empty answer — that would be scored as a model failure when it is our bug."""


class RateLimited(Exception):
    """Provider returned 429. Tracked, because it determines whether a 3000-row
    shadow run is feasible on a free tier."""


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
    if r.status_code == 429:
        raise RateLimited(retry_after_seconds(r.headers.get("retry-after")))
    r.raise_for_status()
    body = r.json()
    choice = body["choices"][0]
    message = choice["message"]
    # Hard rule 11: a MISSING content field means the response shape is not what we
    # think it is, and must not be recorded as an empty model answer — that would be
    # counted as a model failure when it is our bug. A present-but-empty content
    # (reasoning consumed the budget) is real data and passes through.
    if "content" not in message:
        raise ShapeError(
            f"groq response message has no 'content' field; keys={sorted(message)}"
        )
    return (
        {
            "text": message["content"] or "",
            "finish_reason": choice.get("finish_reason"),
            # Reasoning models spend the token budget before emitting content; recorded
            # so a truncated answer is never mistaken for a bad prompt.
            "reasoning_tokens": (body.get("usage") or {})
            .get("completion_tokens_details", {})
            .get("reasoning_tokens"),
        },
        body.get("usage"),
    )


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
    if r.status_code == 429:
        raise RateLimited(retry_after_seconds(r.headers.get("retry-after")))
    r.raise_for_status()
    body = r.json()
    candidate = body["candidates"][0]
    if "content" not in candidate:
        # A blocked or filtered candidate has no content. That is not an empty answer.
        raise ShapeError(
            f"gemini candidate has no 'content'; finishReason="
            f"{candidate.get('finishReason')!r}, keys={sorted(candidate)}"
        )
    parts = candidate["content"].get("parts") or []
    if not parts or "text" not in parts[0]:
        raise ShapeError(f"gemini candidate content has no text part: {parts!r:.200}")
    return (
        {
            "text": parts[0]["text"],
            "finish_reason": candidate.get("finishReason"),
            "reasoning_tokens": (body.get("usageMetadata") or {}).get("thoughtsTokenCount"),
        },
        body.get("usageMetadata"),
    )


CALLERS = {"groq": call_groq, "gemini": call_gemini}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=sorted(CALLERS), required=True)
    ap.add_argument("--model", required=True, help="provider model id")
    ap.add_argument("--n", type=int, default=100, help="first N rows of test_3000")
    ap.add_argument("--manifest", default="test_3000")
    # Reasoning models (gpt-oss, Qwen3 thinking) spend tokens before emitting content,
    # so an output budget sized for the JSON alone truncates them to an empty string.
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min-parse-rate", type=float, default=0.98)
    ap.add_argument("--out", type=Path, default=Path("results/rung0_freetier.json"))
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--max-retries", type=int, default=6,
                    help="rate-limit retry attempts before giving up on a row")
    args = ap.parse_args()

    load_env()
    started = time.monotonic()
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

    backoffs: list[dict] = []
    api_calls = [0]
    predictions: list[str | None] = []
    parse_failures: list[dict] = []
    truncated: list[dict] = []
    unmatched: list[dict] = []
    confidences: list[float] = []
    clean_match: list[dict] = []
    normalized_match: list[dict] = []

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
                delay = 2.0
                for attempt in range(args.max_retries):
                    try:
                        raw, usage = caller(client, args.model, prompt.text, clause, params)
                        break
                    except RateLimited as exc:
                        wait = max(delay, float(str(exc)))
                        backoffs.append({"row": row, "attempt": attempt + 1, "waited_s": wait})
                        print(f"  rate limited on row {row}; waiting {wait:.1f}s")
                        time.sleep(wait)
                        delay = min(delay * 2, 60.0)
                else:
                    raise RuntimeError(
                        f"row {row}: still rate limited after {args.max_retries} "
                        f"attempts. {api_calls[0]} rows were fetched and cached this "
                        "session; re-run the same command to resume from the cache."
                    )
                cache.put(key, raw, usage=usage, request_params=params)
                api_calls[0] += 1
                if args.sleep:
                    time.sleep(args.sleep)

            text = raw["text"] if isinstance(raw, dict) else raw
            finish = raw.get("finish_reason") if isinstance(raw, dict) else None
            record = {"row": row, "raw": text, "gold": gold[i], "finish_reason": finish}

            try:
                parsed = parse_response(text)
            except ParseFailure as exc:
                record["error"] = str(exc)
                # Truncation is a third failure mode, distinct from malformed JSON and
                # from an out-of-vocabulary label. Conflating them hides the real fix.
                (truncated if finish == "length" else parse_failures).append(record)
                predictions.append(counter.record(text, None))
                continue

            confidences.append(parsed.confidence)
            matched = normalizer.normalize(parsed.label)
            record["raw_label"] = parsed.label
            if matched is None:
                unmatched.append(record)
            elif matched == parsed.label:
                clean_match.append(record)
            else:
                record["normalized_to"] = matched
                normalized_match.append(record)
            predictions.append(counter.record(parsed.label, matched))

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)} done")

    elapsed = time.monotonic() - started
    n = len(predictions)
    n_parse_ok = n - len(parse_failures) - len(truncated)
    parse_rate = n_parse_ok / n
    report = score(gold, predictions, labels=sorted(set(gold)))

    print("\n" + "=" * 64)
    print(f"JSON parse success rate   : {parse_rate:.2%}  ({n_parse_ok}/{n})")
    print(f"label match rate          : {1 - counter.failure_rate:.2%}")
    print(f"  malformed JSON          : {len(parse_failures)}")
    print(f"  truncated (finish=length): {len(truncated)}")
    print(f"  parsed but out-of-vocab : {len(unmatched)}")
    print(f"  matched verbatim        : {len(clean_match)}")
    print(f"  matched via normalizer  : {len(normalized_match)}")
    print("-" * 64)
    print(report.render())
    if confidences:
        conf = sorted(confidences)
        print(
            f"  confidence p50/p90      : {conf[len(conf) // 2]:.2f} / "
            f"{conf[int(0.9 * len(conf))]:.2f}   (zero-shot: unanchored, see C2)"
        )
    print("-" * 64)
    print(f"wall clock                : {elapsed:.1f}s  ({elapsed / max(1, n):.2f}s/row)")
    print(f"live API calls            : {api_calls[0]}  (rest served from cache)")
    print(f"rate-limit backoffs       : {len(backoffs)}"
          + (f"  total wait {sum(b['waited_s'] for b in backoffs):.1f}s" if backoffs else ""))
    if api_calls[0]:
        rate = api_calls[0] / elapsed
        print(f"throughput                : {rate:.2f} req/s "
              f"-> 3000 rows ≈ {3000 / rate / 60:.1f} min")
    print("=" * 64)

    failures = parse_failures + truncated + unmatched
    if failures:
        print("\nEVERY FAILURE, VERBATIM (raw output beside expected label):\n")
        for f in sorted(failures, key=lambda x: x["row"]):
            kind = (
                "OUT OF VOCABULARY" if "raw_label" in f
                else "TRUNCATED" if f.get("finish_reason") == "length"
                else "MALFORMED JSON"
            )
            print(f"--- row {f['row']} — {kind}")
            print(f"    expected label : {f['gold']!r}")
            print(f"    finish_reason  : {f.get('finish_reason')!r}")
            if "raw_label" in f:
                print(f"    returned label : {f['raw_label']!r}")
            print(f"    raw output     : {f['raw']!r}\n")
    else:
        print("\nNo failures of any kind.")

    if normalized_match:
        print("OUTPUTS THAT REQUIRED NORMALIZATION:\n")
        for f in normalized_match:
            print(f"  row {f['row']}: {f['raw_label']!r} -> {f['normalized_to']!r}")
        print()

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
                "malformed_json": parse_failures,
                "truncated": truncated,
                "out_of_vocabulary": unmatched,
                "matched_verbatim": len(clean_match),
                "matched_via_normalizer": normalized_match,
                "cache_stats": cache.cache_stats().as_dict(),
                "wall_clock_s": elapsed,
                "live_api_calls": api_calls[0],
                "rate_limit_backoffs": backoffs,
                "confidences": confidences,
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
