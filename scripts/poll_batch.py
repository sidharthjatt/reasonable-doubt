"""Poll a submitted batch and, when it ends, join results and record actual cost.

    python scripts/poll_batch.py --run stage1 [--once]

Safe to re-run: it reads persisted state and never resubmits. `--once` checks status
and exits, for use from a scheduler rather than a long-lived process.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.batch import BatchClient, BatchState, join_on_custom_id, parse_custom_id
from src.api.cache import ResponseCache
from src.api.env import load_env
from src.api.ledger import SpendLedger
from src.api.providers import get_provider

STATE_DIR = Path("results/batches")


class SDKBatch:
    def __init__(self, c): self.c = c
    def create_batch(self, r): raise RuntimeError("poll must never create a batch")
    def retrieve_batch(self, b): return self.c.messages.batches.retrieve(b)
    def batch_results(self, b): return [json.loads(x.model_dump_json())
                                        for x in self.c.messages.batches.results(b)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    load_env()
    import anthropic

    state = BatchState.load(args.run, STATE_DIR)
    if state is None:
        print(f"no persisted state for run {args.run!r}", file=sys.stderr); return 1
    client = BatchClient(SDKBatch(anthropic.Anthropic()), state_dir=STATE_DIR)

    if args.once:
        b = client.client.retrieve_batch(state.batch_id)
        st = b.processing_status if hasattr(b, "processing_status") else b["processing_status"]
        counts = getattr(b, "request_counts", None)
        print(f"{args.run}: {st}" + (f"  {counts}" if counts else ""))
        if st != "ended":
            return 2
    else:
        client.poll(state)

    joined = join_on_custom_id(state.custom_ids, client.client.batch_results(state.batch_id))
    print(f"joined: {joined.n_succeeded} succeeded, {joined.n_failed} failed")

    meta = json.loads((STATE_DIR / f"{args.run}_meta.json").read_text())
    provider = get_provider("anthropic"); cache = ResponseCache(); ledger = SpendLedger()
    by_tag: dict[str, list] = {}
    for cid, body in joined.succeeded.items():
        by_tag.setdefault(cid.split("_", 1)[0], []).append((cid, body))

    out = {}
    for leg in meta["legs"]:
        tag, model = leg["custom_id_tag"], leg["model"]
        items = by_tag.get(tag, [])
        if not items:
            print(f"  {model}: no results for tag {tag!r}"); continue
        total = None
        rows = {}
        for cid, body in items:
            u = provider.parse_usage(body["usage"], model=model)
            total = u if total is None else total + u
            text = "".join(b.get("text", "") for b in body.get("content", [])
                           if b.get("type") == "text")
            rows[parse_custom_id(cid)[1]] = {"text": text,
                                             "stop_reason": body.get("stop_reason"),
                                             "usage": u.as_dict()}
        entry = ledger.record_actual(
            f"{args.run}_{model}", total, provider="anthropic", model=model,
            batch_id=state.batch_id, batch=True, cache_ttl=state.cache_ttl,
            estimated_usd=leg["gating_usd"], n_requests=len(items),
            notes=f"{args.run} batch; tag {tag}")
        cr = total.cache_read_input_tokens; cw = total.cache_creation_input_tokens
        billed = total.input_tokens + (cr or 0) + (cw or 0)
        out[model] = {"n": len(items), "usage": total.as_dict(),
                      "estimated_usd": leg["gating_usd"], "actual_usd": entry.cost_usd,
                      "cache_hit_rate": (cr / billed) if billed else 0.0, "rows": rows}
        print(f"  {model}: est ${leg['gating_usd']:.4f} -> ACTUAL ${entry.cost_usd:.4f} "
              f"({entry.cost_usd/leg['gating_usd']-1:+.1%})")
        print(f"    input {total.input_tokens:,} | cache_write {cw:,} | "
              f"cache_read {cr:,} | output {total.output_tokens:,}")
        print(f"    CACHE HIT RATE (read / all input): {out[model]['cache_hit_rate']:.1%}")

    res = Path(f"results/{args.run}_results.json")
    res.write_text(json.dumps({"run": args.run, "batch_id": state.batch_id,
                               "failed": {k: str(v) for k, v in joined.failed.items()},
                               "models": out}, indent=2))
    print(f"\nwrote {res}   cumulative spend ${ledger.cumulative_usd():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
