# ============================================================================
# TIER 0 — DeBERTa-v3-base on LEDGAR.  PASTE INTO ONE KAGGLE CELL.
# Settings: Accelerator = GPU T4 x2 (or P100).  Internet = ON.
#
# RESUMES. Kaggle sessions are time-capped; if one dies, re-run this exact cell
# and it skips seeds whose result JSON already exists under /kaggle/working and
# restarts an interrupted seed from its last epoch checkpoint. Nothing is lost.
#
# Registered as E1 (CE baseline, 3 seeds) and E2 (loss arms). Selection happens
# on train_holdout_3000 ONLY — the guard below refuses dev and test.
# ============================================================================
!pip -q install "transformers>=4.44" "datasets>=2.19" sentencepiece protobuf \
    "optimum[onnxruntime]" onnx onnxruntime evaluate scikit-learn 2>&1 | tail -2

import os, json, gc, hashlib, random, shutil
from pathlib import Path
from collections import Counter
import numpy as np, torch
from torch import nn
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
MODEL = "microsoft/deberta-v3-base"
SEEDS = [1, 2, 3]
LOSS_ARM = "ce"            # E1 baseline. E2 arms: sqrt_inv_freq | effective_number | inv_freq
MAX_LENGTH = 512
EPOCHS, LR, BS = 3, 2e-5, 16
HOLDOUT_SHA12 = "97eebc04d0c7"   # committed train_holdout_3000

# ---------------- split guard (duplicated so the cell is standalone) ---------
SELECTION_SPLITS = {"train_holdout_3000", "train"}
THRESHOLD_SPLITS = {"dev_2000", "validation", "dev"}
REPORTING_SPLITS = {"test_3000", "test_stratified_764", "test"}
def assert_selection_split(name):
    if name in REPORTING_SPLITS: raise RuntimeError(f"REFUSING: {name!r} is a REPORTING split")
    if name in THRESHOLD_SPLITS: raise RuntimeError(f"REFUSING: {name!r} is THRESHOLD-only (hard rule 1)")
    if name not in SELECTION_SPLITS: raise RuntimeError(f"unknown selection split {name!r}")
    return name
assert_selection_split("train_holdout_3000")

# ---------------- data + frozen manifest verification ------------------------
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
assert len(ds["train"]) == 60000 and len(ds["validation"]) == 10000 and len(ds["test"]) == 10000, \
    "split sizes drifted — the chronological 60k/10k/10k split is load-bearing"
NAMES = list(ds["train"].features["label"].names); assert len(NAMES) == 100
hold = stratified(list(ds["train"]["label"]), 3000, 20260907)
sha = hash_texts(list(ds["train"].select(hold)["text"]))
assert sha.startswith(HOLDOUT_SHA12), f"holdout sha {sha[:12]} != {HOLDOUT_SHA12}"
print(f"train_holdout_3000 verified {sha[:16]}…")

holdset = set(hold)
fit_idx = [i for i in range(len(ds["train"])) if i not in holdset]
tok = AutoTokenizer.from_pretrained(MODEL)

def prep(split, idx):
    sub = ds[split].select(idx)
    enc = tok(list(sub["text"]), truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = [int(x) for x in sub["label"]]
    return [dict(zip(enc, v)) for v in zip(*enc.values())]

train_ds, sel_ds = prep("train", fit_idx), prep("train", hold)
test_ds = prep("test", list(range(len(ds["test"]))))   # scored offline against the manifest
print(f"fit {len(train_ds):,} | select {len(sel_ds):,} | test {len(test_ds):,}")

def weights(arm):
    if arm == "ce": return None
    cnt = Counter(int(ds["train"][i]["label"]) for i in fit_idx)
    f = np.array([max(1, cnt.get(c, 0)) for c in range(100)], float)
    w = {"inv_freq": 1/f, "sqrt_inv_freq": 1/np.sqrt(f),
         "effective_number": (1-0.999)/(1-np.power(0.999, f))}[arm]
    return w / w.mean()

def metrics(p):
    y, pr = p.label_ids, p.predictions.argmax(-1)
    return {"macro_f1": f1_score(y, pr, average="macro", zero_division=0),
            "accuracy": float((pr == y).mean())}

for seed in SEEDS:
    done = WORK / f"tier0_{LOSS_ARM}_seed{seed}.json"
    if done.exists():
        print(f"seed {seed}: already complete, skipping"); continue
    ckpt_dir = WORK / f"ck_{LOSS_ARM}_{seed}"
    resume = ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    print(f"\n=== seed {seed} ({'RESUMING' if resume else 'fresh'}) ===")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    w = weights(LOSS_ARM)
    wt = None if w is None else torch.tensor(w, dtype=torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=100, id2label=dict(enumerate(NAMES)),
        label2id={n: i for i, n in enumerate(NAMES)})

    class WT(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            lb = inputs.pop("labels"); out = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=None if wt is None else wt.to(out.logits.device))(out.logits, lb)
            return (loss, out) if return_outputs else loss

    tr = WT(model=model,
        args=TrainingArguments(output_dir=str(ckpt_dir), seed=seed,
            num_train_epochs=EPOCHS, learning_rate=LR,
            per_device_train_batch_size=BS, per_device_eval_batch_size=BS*2,
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="macro_f1",
            greater_is_better=True, save_total_limit=2, fp16=True,
            report_to=[], logging_steps=200, dataloader_num_workers=2),
        train_dataset=train_ds, eval_dataset=sel_ds,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    tr.train(resume_from_checkpoint=resume)

    sel = tr.predict(sel_ds); tst = tr.predict(test_ds)
    np.savez_compressed(WORK / f"logits_{LOSS_ARM}_seed{seed}.npz",
                        sel_logits=sel.predictions, sel_labels=sel.label_ids,
                        test_logits=tst.predictions, test_labels=tst.label_ids)
    out = {"model": MODEL, "loss_arm": LOSS_ARM, "seed": seed, "max_length": MAX_LENGTH,
           "epochs": EPOCHS, "lr": LR, "selection_split": "train_holdout_3000",
           "selection": metrics(sel), "test_fp32": metrics(tst)}
    fp32 = WORK / f"fp32_{LOSS_ARM}_{seed}"; tr.save_model(str(fp32)); tok.save_pretrained(str(fp32))

    # INT8 is the DEPLOYED precision — E1's accept rule attaches to it (see E3).
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    ort = ORTModelForSequenceClassification.from_pretrained(str(fp32), export=True)
    ort.save_pretrained(str(WORK / f"onnx_{LOSS_ARM}_{seed}"))
    ORTQuantizer.from_pretrained(str(WORK / f"onnx_{LOSS_ARM}_{seed}")).quantize(
        save_dir=str(WORK / f"int8_{LOSS_ARM}_{seed}"),
        quantization_config=AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True))
    out["int8_dir"] = f"int8_{LOSS_ARM}_{seed}"
    done.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["selection"], indent=2))
    shutil.rmtree(ckpt_dir, ignore_errors=True)   # free disk; result JSON marks completion
    del tr, model; gc.collect(); torch.cuda.empty_cache()

print("\nALL SEEDS DONE — download /kaggle/working/*.json, *.npz and int8_* dirs")
