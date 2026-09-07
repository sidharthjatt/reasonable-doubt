# ============================================================================
# TIER 1 — Qwen2.5-1.5B-Instruct + QLoRA on LEDGAR.  PASTE INTO ONE KAGGLE CELL.
# Settings: Accelerator = GPU T4 x2 (or P100).  Internet = ON.
#
# RESUMES, and resumption matters more here: 3 seeds will NOT fit one session.
# Re-run this exact cell; completed seeds are skipped and an interrupted seed
# restarts from its last checkpoint. Run seeds across as many sessions as needed.
#
# Registered as E4. If E4's accept rule (macro-F1 >= E1 + 0.04) is not met, E4b
# (two-tier Tier 0 -> Claude) is the reported architecture — registered in advance.
#
# GENERATIVE, not a classification head: Tier 1 is served via mlx_lm.server in the
# cascade, so it must emit a label STRING exactly as it will at serving time. A
# sequence-classification head would train faster but would measure a different
# system from the one deployed (the same reason E1 attaches to INT8, not FP32).
# ============================================================================
!pip -q install "transformers>=4.44" "datasets>=2.19" peft trl accelerate \
    bitsandbytes sentencepiece scikit-learn 2>&1 | tail -2

import os, json, gc, hashlib, random, shutil
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # primary. Llama-3.2-3B if access lands;
                                       # else Qwen2.5-3B-Instruct. Record which in
                                       # PREREGISTRATION before the run (E4).
SEEDS = [1, 2, 3]
MAX_LEN, EPOCHS, LR, BS, GA = 1024, 1, 2e-4, 4, 4
HOLDOUT_SHA12 = "97eebc04d0c7"
EVAL_N = 3000                          # test_3000 rows, scored offline by row index

SELECTION_SPLITS = {"train_holdout_3000", "train"}
REPORTING_SPLITS = {"test_3000", "test_stratified_764", "test"}
THRESHOLD_SPLITS = {"dev_2000", "validation", "dev"}
def assert_selection_split(n):
    if n in REPORTING_SPLITS: raise RuntimeError(f"REFUSING: {n!r} is a REPORTING split")
    if n in THRESHOLD_SPLITS: raise RuntimeError(f"REFUSING: {n!r} is THRESHOLD-only")
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
    q = {c: len(r)*n/len(labels) for c, r in by.items()}
    a = {c: int(v) for c, v in q.items()}
    for c in sorted(q, key=lambda c: (-(q[c]-a[c]), c))[: n-sum(a.values())]: a[c] += 1
    rng = np.random.default_rng(seed); out = []
    for c in sorted(by):
        if a[c]: out += [by[c][i] for i in rng.choice(len(by[c]), a[c], replace=False)]
    return sorted(out)

ds = load_dataset("coastalcph/lex_glue", "ledgar")
assert len(ds["train"]) == 60000 and len(ds["test"]) == 10000
NAMES = list(ds["train"].features["label"].names); assert len(NAMES) == 100
hold = stratified(list(ds["train"]["label"]), 3000, 20260907)
assert hash_texts(list(ds["train"].select(hold)["text"])).startswith(HOLDOUT_SHA12)
print("train_holdout_3000 verified")
holdset = set(hold); fit_idx = [i for i in range(60000) if i not in holdset]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token
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
print(f"fit rows: {len(train_txt):,}")

# Exact-match normalisation, mirroring src/data/labels.py — no fuzzy rescue.
import re, unicodedata
EDGE = " \t\r\n\"'`“”‘’*_.,;:!?()[]{}<>"
LOOKUP = {re.sub(r"\s+", " ", unicodedata.normalize("NFKC", n).strip(EDGE)).casefold(): n
          for n in NAMES}
def normalize(s): 
    return LOOKUP.get(re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).strip(EDGE)).casefold())

for seed in SEEDS:
    done = WORK / f"tier1_seed{seed}.json"
    if done.exists(): print(f"seed {seed}: complete, skipping"); continue
    ck = WORK / f"t1_ck_{seed}"
    resume = ck.exists() and any(ck.glob("checkpoint-*"))
    print(f"\n=== seed {seed} ({'RESUMING' if resume else 'fresh'}) ===")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # QLoRA: 4-bit base + LoRA adapters. bitsandbytes is CUDA-only — Kaggle only,
    # never on the Mac (see CLAUDE.md > Hardware).
    base = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True))
    base = prepare_model_for_kbit_training(base)
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

    trainer = SFTTrainer(model=base, train_dataset=train_txt, peft_config=peft_cfg,
        args=TrainingArguments(output_dir=str(ck), seed=seed, num_train_epochs=EPOCHS,
            learning_rate=LR, per_device_train_batch_size=BS,
            gradient_accumulation_steps=GA, fp16=True, logging_steps=100,
            save_strategy="steps", save_steps=500, save_total_limit=2,
            report_to=[], max_seq_length=MAX_LEN, gradient_checkpointing=True))
    trainer.train(resume_from_checkpoint=resume)
    adapter = WORK / f"t1_adapter_{seed}"; trainer.model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))

    # ---- evaluate generatively, the way it will be served ----
    model = trainer.model.eval()
    preds, raws = [], []
    rows = list(range(EVAL_N))
    for s in range(0, len(rows), 16):
        chunk = [ds["test"][i]["text"] for i in rows[s:s+16]]
        enc = tok([to_text(t) for t in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=12, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            raws.append(txt); preds.append(normalize(txt))
        if s % 480 == 0: print(f"  eval {s}/{len(rows)}", flush=True)

    gold = [NAMES[int(ds["test"][i]["label"])] for i in rows]
    SENT = "\x00UNMATCHED"
    p = [x if x is not None else SENT for x in preds]
    res = {"model": MODEL, "seed": seed, "n": len(rows),
           "macro_f1": f1_score(gold, p, labels=sorted(set(gold)), average="macro", zero_division=0),
           "accuracy": float(np.mean([g == q for g, q in zip(gold, p)])),
           "format_failure_rate": float(np.mean([x is None for x in preds])),
           "note": "scored on the FULL 10k test split by row index; join to test_3000 offline"}
    json.dump({"predictions": preds, "raw": raws}, open(WORK/f"tier1_preds_seed{seed}.json","w"))
    done.write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
    shutil.rmtree(ck, ignore_errors=True)
    del trainer, base, model; gc.collect(); torch.cuda.empty_cache()

print("\nDONE — download /kaggle/working/tier1_*.json and t1_adapter_* dirs")
