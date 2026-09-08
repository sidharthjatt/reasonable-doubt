"""Exercise Tier 1's EVAL path end to end, before any Tier 1 seed has ever finished.

The eval path has never run: it sits after 13.4h of training, so its first execution
would be 13.4h into a run. This drives it with an UNTRAINED LoRA adapter on the real
base model, so every step executes and the output file's shape can be checked.

Accuracy will be garbage. That is expected and is not what is being tested.

DEVIATIONS FROM THE NOTEBOOK, stated because they are the limits of this check:
  * CPU, so no 4-bit: bitsandbytes is CUDA-only. The LOAD path differs; the EVAL path
    (chat template -> generation -> left-padded slice -> normalize -> score -> write)
    is identical and is what is under test.
  * a few rows, not 3,000.
  * the OOM-halving branch is CUDA-only (`torch.cuda.OutOfMemoryError`) and cannot fire
    here — noted, not exercised.

    python scripts/tier1_eval_smoke.py [--rows 8] [--bs 4] [--model ...]
"""

from __future__ import annotations

import argparse, json, re, sys, time, unicodedata
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NB = (ROOT / "notebooks" / "kaggle_tier1.py").read_text()


def lift(start: str, end: str) -> str:
    """Take a block VERBATIM from the notebook, so this harness cannot drift from it."""
    a = NB.index(start)
    b = NB.index(end, a)
    return NB[a:b]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", type=Path, default=Path("/tmp/tier1_smoke"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    problems: list[str] = []

    print("[1] dataset + label space")
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    NAMES = list(ds["train"].features["label"].names)

    print("[2] normalizer, lifted verbatim from the notebook")
    ns = {"re": re, "unicodedata": unicodedata, "NAMES": NAMES}
    exec(lift("EDGE = ", "# ============================ RESTORE"), ns)
    normalize, LOOKUP = ns["normalize"], ns["LOOKUP"]
    # LEDGAR label names are PLURAL ("Governing Laws", "Notices"). That matters below.
    assert normalize("Governing Laws") == "Governing Laws"
    assert normalize("  governing laws  ") == "Governing Laws"
    assert normalize("Governing Laws.") == "Governing Laws"
    assert normalize("not a label") is None
    print(f"    LOOKUP {len(LOOKUP)} labels; exact/case/edge/miss all behave")

    # The normaliser is EXACT MATCH by design ("no fuzzy rescue"). LEDGAR class names
    # are plural and the natural model output is the singular; every miss is BOTH a
    # format failure and a wrong answer, so measure the exposure.
    plural = [n for n in NAMES if n.endswith("s")]
    lost = [n for n in plural if normalize(n[:-1]) is None]
    print(f"    NOTE: {len(plural)}/{len(NAMES)} labels end in 's'; for {len(lost)} of",
          f"them the singular form does NOT normalise (e.g. Governing Law ->",
          f"{normalize('Governing Law')!r}). Exact match is deliberate, but the",
          f"fine-tune must reproduce the plural exactly or the row scores wrong.")

    print("[3] base model + UNTRAINED LoRA adapter (CPU, no 4-bit)")
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    print(f"    loaded in {time.time()-t0:.0f}s; "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")

    print("[4] prompt builder, lifted verbatim")
    ns2 = {"tok": tok, "NAMES": NAMES}
    exec(lift("LABEL_BLOCK = ", "train_txt = Dataset.from_dict"), ns2)
    to_text, SYS = ns2["to_text"], ns2["SYS"]
    sample = to_text("This Agreement shall be governed by the laws of New York.")
    assert sample.endswith(("assistant\n", "assistant\n\n")) or "assistant" in sample[-40:]
    print(f"    prompt ends with the generation prompt; {len(tok(sample)['input_ids'])} tokens")

    print("[5] generation with LEFT padding")
    MAX_LEN, MAX_NEW_TOKENS = 2560, 16
    model.config.use_cache = True
    if hasattr(model, "generation_config"): model.generation_config.use_cache = True
    model.eval()
    tok.padding_side = "left"
    assert tok.padding_side == "left"

    EVAL_IDX = list(range(a.rows))
    preds, raws, trunc = [], [], 0

    def generate_chunk(texts):
        enc = tok([to_text(t) for t in texts], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        res = []
        for j in range(len(texts)):
            gen = out[j][enc["input_ids"].shape[1]:]
            res.append((gen, tok.decode(gen, skip_special_tokens=True).strip()))
        return res

    s0 = 0
    while s0 < len(EVAL_IDX):
        texts = [ds["test"][i]["text"] for i in EVAL_IDX[s0:s0 + a.bs]]
        got = generate_chunk(texts)
        for gen, txt in got:
            if len(gen) >= MAX_NEW_TOKENS and tok.eos_token_id not in gen.tolist():
                trunc += 1
            raws.append(txt); preds.append(normalize(txt))
        s0 += len(texts)
        print(f"    {s0}/{len(EVAL_IDX)} in {time.time()-t0:.0f}s", flush=True)

    print(f"    raw samples: {raws[:3]}")
    print(f"    normalized : {preds[:3]}")
    print(f"    truncated  : {trunc}/{len(EVAL_IDX)}  "
          f"format failures: {sum(x is None for x in preds)}/{len(EVAL_IDX)}")

    print("[6] scoring, exactly as the notebook does it")
    gold = [NAMES[int(ds["test"][i]["label"])] for i in EVAL_IDX]
    SENT = "\x00UNMATCHED"
    p = [x if x is not None else SENT for x in preds]
    nb_macro = f1_score(gold, p, labels=sorted(set(gold)), average="macro", zero_division=0)
    print(f"    notebook macro_f1 = {nb_macro:.4f} (garbage by design)")

    print("[7] the same numbers through src.eval.metrics.score")
    from src.eval.metrics import score as project_score
    try:
        rep = project_score(gold, preds)
        print(f"    project macro_f1 = {rep.macro_f1:.4f} over "
              f"{rep.classes_averaged} classes")
    except Exception as e:
        problems.append(f"src.eval.metrics.score raised where the notebook did not: {e!r}")
        print(f"    RAISED: {e!r}")

    print("[8] results JSON write + shape")
    res = {"model": a.model, "seed": 0, "n": len(EVAL_IDX),
           "eval_manifest": "test_3000", "eval_manifest_sha256": "smoke",
           "max_seq_length": MAX_LEN, "max_new_tokens": MAX_NEW_TOKENS,
           "padding_side_train": "right", "padding_side_eval": "left",
           "macro_f1": nb_macro,
           "accuracy": float(np.mean([g == q for g, q in zip(gold, p)])),
           "format_failure_rate": float(np.mean([x is None for x in preds])),
           "generation_truncated": trunc, "final_eval_batch_size": a.bs}
    (a.out / "tier1_preds_seed0.json").write_text(json.dumps(
        {"row_indices": EVAL_IDX, "predictions": preds, "raw": raws}))
    (a.out / "tier1_seed0.json").write_text(json.dumps(res, indent=2))
    back = json.loads((a.out / "tier1_seed0.json").read_text())
    expected = {"model", "seed", "n", "eval_manifest", "eval_manifest_sha256",
                "max_seq_length", "max_new_tokens", "padding_side_train",
                "padding_side_eval", "macro_f1", "accuracy", "format_failure_rate",
                "generation_truncated", "final_eval_batch_size"}
    missing = expected - set(back)
    if missing:
        problems.append(f"results JSON missing keys: {sorted(missing)}")
    if not isinstance(back["macro_f1"], float):
        problems.append(f"macro_f1 serialised as {type(back['macro_f1']).__name__}, "
                        f"not float — np.float64 is not JSON-serialisable in general")
    print(f"    wrote {a.out}/tier1_seed0.json with {len(back)} keys")

    print("\n" + "=" * 70)
    if problems:
        print(f"EVAL PATH: {len(problems)} PROBLEM(S)")
        for x in problems: print(f"  - {x}")
    else:
        print("EVAL PATH: every step executed; output shape is correct")
    print(f"({time.time()-t0:.0f}s total)")
    print("=" * 70)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
