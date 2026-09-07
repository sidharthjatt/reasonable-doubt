"""Score whatever is already in the response cache. Makes NO API calls.

    python scripts/score_cached.py --provider groq --model openai/gpt-oss-120b \
        --max-tokens 512 --manifest test_3000

Exists so a partial run can be evaluated before deciding whether to spend more
requests finishing it. Rows absent from the cache are reported as such and excluded —
they are NOT scored as wrong, because an unsent request is not a model failure.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.cache import ResponseCache, content_sha256  # noqa: E402
from src.api.providers import get_provider  # noqa: E402
from src.data.labels import LabelNormalizer, load_labels  # noqa: E402
from src.data.loading import get_split, label_names, load_ledgar  # noqa: E402
from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest, verify_manifest  # noqa: E402
from src.data.prompts import render_zeroshot  # noqa: E402
from src.data.schema import ParseFailure, parse_response  # noqa: E402
from src.eval.metrics import score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--manifest", default="test_3000")
    ap.add_argument("--limit", type=int, default=None, help="first N manifest rows")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = load_ledgar()
    manifest = load_manifest(args.manifest, DEFAULT_MANIFEST_DIR)
    verify_manifest(manifest, ds, DEFAULT_MANIFEST_DIR)
    names = label_names(ds)
    split = get_split(ds, manifest.split)
    prompt = render_zeroshot(names)
    normalizer = LabelNormalizer(load_labels())
    provider = get_provider(args.provider)
    cache = ResponseCache()

    rows = manifest.indices[: args.limit] if args.limit else manifest.indices
    params = {"temperature": args.temperature, "max_tokens": args.max_tokens}

    gold, pred, missing, truncated, oov, malformed = [], [], [], [], [], []
    kept_rows = []
    for i in rows:
        row = split[i]
        key = provider.cache_key(args.model, system=prompt.text,
                                 messages=[{"role": "user", "content": row["text"]}],
                                 params=params)
        entry = cache.get_or_none(key)
        if entry is None:
            missing.append(i)
            continue
        raw = entry.response
        text = raw["text"] if isinstance(raw, dict) else raw
        finish = raw.get("finish_reason") if isinstance(raw, dict) else None
        g = names[int(row["label"])]
        try:
            parsed = parse_response(text)
        except ParseFailure:
            (truncated if finish == "length" else malformed).append(i)
            gold.append(g); pred.append(None); kept_rows.append(i)
            continue
        m = normalizer.normalize(parsed.label)
        if m is None:
            oov.append(i)
        gold.append(g); pred.append(m); kept_rows.append(i)

    if not gold:
        print("nothing cached for that (model, params) combination", file=sys.stderr)
        return 1

    n = len(gold)
    report = score(gold, pred, labels=sorted(set(gold)))
    trunc = set(truncated)
    kept = [(g, p) for r, g, p in zip(kept_rows, gold, pred) if r not in trunc]
    report_excl = None
    if trunc and kept:
        gk, pk = zip(*kept)
        report_excl = score(list(gk), list(pk), labels=sorted(set(gk)))

    print("=" * 68)
    print(f"CACHED-ONLY SCORING — {args.provider}/{args.model} @ max_tokens={args.max_tokens}")
    print(f"manifest {manifest.name} sha={manifest.text_sha256[:12]}…")
    print("=" * 68)
    print(f"  rows requested   : {len(rows)}")
    print(f"  rows cached      : {n}   ({n / len(rows):.1%})")
    print(f"  rows NOT cached  : {len(missing)}   (excluded, NOT scored wrong — an "
          "unsent request is not a model failure)")
    print(f"  malformed JSON   : {len(malformed)}")
    print(f"  truncated        : {len(truncated)}   ({len(truncated) / n:.2%})")
    print(f"  out-of-vocabulary: {len(oov)}")
    print(f"  class coverage   : {report.classes_in_gold}/100 classes present in gold")
    print("-" * 68)
    print("AS RUN — truncations scored WRONG:")
    print(report.render())
    if report_excl is not None:
        print("-" * 68)
        print(f"TRUNCATED ROWS EXCLUDED — n={report_excl.n}:")
        print(report_excl.render())
        print(f"\n  macro-F1 attributable to the output budget: "
              f"{report_excl.macro_f1 - report.macro_f1:+.4f}")
    print("=" * 68)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "provider": args.provider, "model": args.model,
            "max_tokens": args.max_tokens, "manifest": manifest.name,
            "manifest_sha256": manifest.text_sha256,
            "rows_requested": len(rows), "rows_cached": n,
            "rows_missing": len(missing),
            "truncation_count": len(truncated), "truncation_rate": len(truncated) / n,
            "malformed_json": len(malformed), "out_of_vocabulary": len(oov),
            "metrics": report.as_dict(),
            "metrics_excluding_truncated": report_excl.as_dict() if report_excl else None,
            "scored_rows": kept_rows,
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
