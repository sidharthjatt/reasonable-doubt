# =============================================================================
# TIER 1 — Qwen2.5-1.5B-Instruct + QLoRA on LEDGAR.  PASTE INTO ONE KAGGLE CELL.
# Settings: Accelerator = GPU T4 x2 (or P100).  Internet = ON.
#
# RESUMES. Re-run this exact cell after a session dies: completed seeds are skipped,
# a seed whose adapter exists skips straight to eval, and an interrupted training run
# restarts from its last checkpoint.
#
# Registered as E4. If E4's rule (macro-F1 >= E1 + 0.04) is not met, E4b (two-tier
# Tier 0 -> Claude) is the reported architecture — registered in advance.
#
# GENERATIVE, not a classification head: Tier 1 is served via mlx_lm.server, so it must
# emit a label STRING as it will at serving time (same reason E1 attaches to INT8).
# =============================================================================
!pip -q install "transformers>=4.44" "datasets>=2.19" peft trl accelerate \
    bitsandbytes sentencepiece scikit-learn 2>&1 | tail -2

import os, json, gc, hashlib, random, re, shutil, unicodedata
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
# max_seq_length lives on SFTConfig in current TRL, NOT TrainingArguments — passing it
# to TrainingArguments raises TypeError. Import defensively.
try:
    from trl import SFTConfig
    HAS_SFTCONFIG = True
except ImportError:
    from transformers import TrainingArguments as SFTConfig
    HAS_SFTCONFIG = False

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # record the actual model in PREREGISTRATION (E4)
SEEDS = [1, 2, 3]
# MAX_LEN bounded over the FULL 57,000-row fit set, not a sample. A 300-row head
# sample said max 1154; the true max is 2378 — LEDGAR is chronologically ordered so a
# head sample is not representative. Counts over the full set:
#   >1024: 754 rows (1.32%)   >1536: 17   >2048: 2   >2560: 0
# The LABEL sits at the end of every training example, so any truncation removes the
# answer. 2560 is the smallest bound with zero truncation; dynamic padding means the
# cost is paid only on the rare long batches (mean length is 607).
MAX_LEN, EPOCHS, LR, BS, GA = 2560, 1, 2e-4, 4, 4
EVAL_BS = 8            # halved automatically on OOM; 2560-token prompts are large
MAX_NEW_TOKENS = 16                    # longest label = 5 tokens + eos; 16 is 2.7x margin
EVAL_N = 3000
HOLDOUT_SHA12, TEST3000_SHA12 = "97eebc04d0c7", "e719c1109069"
print(f"trl SFTConfig available: {HAS_SFTCONFIG}")

# ---- split guard (duplicated so the cell is standalone) ----------------------
REPORTING = {"test_3000", "test_stratified_764", "test"}
THRESHOLD = {"dev_2000", "validation", "dev"}
def assert_selection_split(n):
    if n in REPORTING: raise RuntimeError(f"REFUSING: {n!r} is a REPORTING split")
    if n in THRESHOLD: raise RuntimeError(f"REFUSING: {n!r} is THRESHOLD-only")
    return n
assert_selection_split("train_holdout_3000")

def hash_texts(ts):
    h = hashlib.sha256()
    for t in ts:
        b = t.encode(); h.update(f"{len(b)}\n".encode()); h.update(b)
    return h.hexdigest()

def stratified(labels, n, seed):
    by = {}
    for i, l in enumerate(labels): by.setdefault(int(l), []).append(i)
    q = {c: len(r) * n / len(labels) for c, r in by.items()}
    a = {c: int(v) for c, v in q.items()}
    for c in sorted(q, key=lambda c: (-(q[c] - a[c]), c))[: n - sum(a.values())]: a[c] += 1
    rng = np.random.default_rng(seed); out = []
    for c in sorted(by):
        if a[c]: out += [by[c][i] for i in rng.choice(len(by[c]), a[c], replace=False)]
    return sorted(out)

ds = load_dataset("coastalcph/lex_glue", "ledgar")
assert len(ds["train"]) == 60000 and len(ds["test"]) == 10000, "split sizes drifted"
NAMES = list(ds["train"].features["label"].names); assert len(NAMES) == 100

hold = stratified(list(ds["train"]["label"]), 3000, 20260907)
assert hash_texts(list(ds["train"].select(hold)["text"])).startswith(HOLDOUT_SHA12)
print("train_holdout_3000 verified")
fit_idx = [i for i in range(60000) if i not in set(hold)]

# EVAL ROWS: test_3000 is a proportional stratified sample spanning the WHOLE 10k test
# split (indices 1..9992), not the first 3000 rows. Using range(3000) would cover only
# 881 of the 3000 manifest rows — 70.6% missed — and the join to Claude's Stage 1
# results would silently match 881 rows instead of 3000.
EVAL_IDX = stratified(list(ds["test"]["label"]), EVAL_N, 20260907)
_sha = hash_texts(list(ds["test"].select(EVAL_IDX)["text"]))
assert _sha.startswith(TEST3000_SHA12), f"test_3000 sha {_sha[:12]} != {TEST3000_SHA12}"
print(f"test_3000 verified {_sha[:16]}… ({len(EVAL_IDX)} rows, "
      f"range {min(EVAL_IDX)}..{max(EVAL_IDX)})")

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token
# PADDING SIDE DIFFERS BY PHASE — set at each point of use, never once globally.
#   TRAINING  -> "right". With left padding the first real token is predicted from a
#                masked pad position, so its loss term is garbage. Measured on
#                Qwen2.5-0.5B: batched loss 7.548 under left padding vs 6.206 unpadded,
#                a drift of 1.34 on the padded example. Silent.
#   GENERATION -> "left". With right padding the model conditions on trailing pads and
#                every sequence but the longest in a batch emits fluent nonsense —
#                reproduced: 3 of 4 prompts wrong.
tok.padding_side = "right"

LABEL_BLOCK = "\n".join(f"- {n}" for n in NAMES)
SYS = ("You are a legal contract analyst. Classify the contract provision into exactly "
       f"one of these 100 clause types:\n\n{LABEL_BLOCK}\n\n"
       "Reply with the label only, copied character-for-character.")

def to_text(text, label=None):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s + (label + tok.eos_token if label is not None else "")

train_txt = Dataset.from_dict({"text": [
    to_text(ds["train"][i]["text"], NAMES[int(ds["train"][i]["label"])]) for i in fit_idx]})
# Verify over EVERY fit row, not a sample, and FAIL if any label would be cut off.
print(f"fit rows: {len(train_txt):,} | checking all of them against MAX_LEN={MAX_LEN}…")
_lens = []
for _s in range(0, len(train_txt), 2000):
    _lens += [len(x) for x in
              tok(train_txt["text"][_s:_s+2000], add_special_tokens=False)["input_ids"]]
_lens = np.array(_lens)
print(f"  train token length: mean {_lens.mean():.0f} p99 {np.percentile(_lens,99):.0f} "
      f"MAX {_lens.max()}  |  over MAX_LEN: {(_lens > MAX_LEN).sum()}")
assert (_lens > MAX_LEN).sum() == 0, (
    f"{(_lens > MAX_LEN).sum()} training examples exceed MAX_LEN={MAX_LEN}. The LABEL is "
    f"at the end of every example, so these would train on a prompt with no answer. "
    f"Raise MAX_LEN to at least {int(_lens.max())}.")

# Exact-match normalisation, mirroring src/data/labels.py — no fuzzy rescue.
EDGE = " \t\r\n\"'`“”‘’*_.,;:!?()[]{}<>"
def _key(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).strip(EDGE)).casefold()
LOOKUP = {_key(n): n for n in NAMES}
def normalize(s): return LOOKUP.get(_key(s))

for seed in SEEDS:
    done = WORK / f"tier1_seed{seed}.json"
    if done.exists():
        print(f"seed {seed}: complete, skipping"); continue
    ck = WORK / f"t1_ck_{seed}"; adapter = WORK / f"t1_adapter_{seed}"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    if adapter.exists():
        # RESUME GAP FIX: eval can take >1h. If the session died during eval, the
        # adapter already exists and retraining the whole seed would waste hours.
        print(f"\n=== seed {seed}: adapter exists, SKIPPING TRAINING, going to eval ===")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto",
                                                    quantization_config=quant)
        model = PeftModel.from_pretrained(base, str(adapter))
    else:
        resume = ck.exists() and any(ck.glob("checkpoint-*"))
        print(f"\n=== seed {seed} ({'RESUMING' if resume else 'fresh'}) ===")
        base = prepare_model_for_kbit_training(
            AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto",
                                                 quantization_config=quant))
        assert tok.padding_side == "right", (
            "training requires RIGHT padding; left padding corrupts the loss on the "
            "first real token of every padded sequence")
        cfg_kw = dict(output_dir=str(ck), seed=seed, num_train_epochs=EPOCHS,
            learning_rate=LR, per_device_train_batch_size=BS,
            gradient_accumulation_steps=GA, fp16=True, logging_steps=100,
            save_strategy="steps", save_steps=500, save_total_limit=2,
            report_to=[], gradient_checkpointing=True)
        if HAS_SFTCONFIG: cfg_kw["max_seq_length"] = MAX_LEN
        trainer = SFTTrainer(model=base, train_dataset=train_txt,
            peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj","k_proj","v_proj","o_proj",
                                "gate_proj","up_proj","down_proj"]),
            args=SFTConfig(**cfg_kw))
        trainer.train(resume_from_checkpoint=resume)
        trainer.model.save_pretrained(str(adapter)); tok.save_pretrained(str(adapter))
        shutil.rmtree(ck, ignore_errors=True)   # adapter saved — never retrain this seed
        model = trainer.model
        del trainer

    # gradient_checkpointing forces use_cache=False; generation needs it back on or it
    # is very slow. Restore explicitly rather than relying on the default.
    model.config.use_cache = True
    if hasattr(model, "generation_config"): model.generation_config.use_cache = True
    model.eval()

    # switch to LEFT for generation, and assert it here rather than trusting the top
    tok.padding_side = "left"
    assert tok.padding_side == "left", "batched generation requires left padding"
    preds, raws, trunc = [], [], 0

    def generate_chunk(texts):
        """Generate for one chunk. Returns list of decoded strings."""
        enc = tok([to_text(t) for t in texts], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        res = []
        for j in range(len(texts)):
            gen = out[j][enc["input_ids"].shape[1]:]      # left padding => safe slice
            res.append((gen, tok.decode(gen, skip_special_tokens=True).strip()))
        return res

    # OOM here would land AFTER 3-4 hours of training. Halve the batch and retry
    # rather than losing the seed; the adapter is already on disk either way.
    eval_bs = EVAL_BS
    s0 = 0
    while s0 < len(EVAL_IDX):
        texts = [ds["test"][i]["text"] for i in EVAL_IDX[s0:s0 + eval_bs]]
        try:
            got = generate_chunk(texts)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            if eval_bs == 1:
                raise RuntimeError("OOM at eval batch size 1 — cannot proceed")
            eval_bs = max(1, eval_bs // 2)
            print(f"  OOM -> halving eval batch to {eval_bs} and retrying", flush=True)
            continue
        for gen, txt in got:
            if len(gen) >= MAX_NEW_TOKENS and tok.eos_token_id not in gen.tolist():
                trunc += 1
            raws.append(txt); preds.append(normalize(txt))
        s0 += len(texts)
        if s0 % 480 < eval_bs: print(f"  eval {s0}/{len(EVAL_IDX)} (bs={eval_bs})", flush=True)

    gold = [NAMES[int(ds["test"][i]["label"])] for i in EVAL_IDX]
    SENT = "\x00UNMATCHED"
    p = [x if x is not None else SENT for x in preds]
    res = {"model": MODEL, "seed": seed, "n": len(EVAL_IDX),
           "eval_manifest": "test_3000", "eval_manifest_sha256": _sha,
           "max_seq_length": MAX_LEN, "max_new_tokens": MAX_NEW_TOKENS,
           "padding_side_train": "right", "padding_side_eval": "left",
           "macro_f1": f1_score(gold, p, labels=sorted(set(gold)), average="macro",
                                zero_division=0),
           "accuracy": float(np.mean([g == q for g, q in zip(gold, p)])),
           "format_failure_rate": float(np.mean([x is None for x in preds])),
           "generation_truncated": trunc, "final_eval_batch_size": eval_bs}
    json.dump({"row_indices": EVAL_IDX, "predictions": preds, "raw": raws},
              open(WORK / f"tier1_preds_seed{seed}.json", "w"))
    done.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
    del model, base; gc.collect(); torch.cuda.empty_cache()

print("\nDONE — download /kaggle/working/tier1_*.json and t1_adapter_* dirs")
