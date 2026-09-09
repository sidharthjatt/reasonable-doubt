# ============================================================================
# TIER 0 — DEV INFERENCE + E3 DISCRIMINATOR.  NO TRAINING.
#
# The first Tier 0 run predates commit 7bc7143 and wrote NO dev arrays:
#   sel_logits (3000,100) sel_labels test_logits (10000,100) test_labels
#   test_3000_indices
# E5 calibrates on dev_2000 (hard rule 1). sel_ is the SELECTION split and test_ is
# REPORTING, so neither may substitute — src/router/calibrate.py raises on both. E5
# therefore cannot run on anything currently in hand, and this script exists to produce
# the missing arrays WITHOUT retraining.
#
# It also folds in the E3 discriminator (PREREGISTRATION 3ah), because it re-exports the
# same checkpoints anyway and a second GPU session for it would be waste.
#
# INPUT: the previous run's /kaggle/working attached, containing fp32_ce_{1,2,3}/.
# OUTPUT: dev_logits_ce_seed{n}.npz  (E5's key convention)
#         tier0_dev_ce_seed{n}.json  (discriminator arms + execution environment)
#
# ============================ CELL 1 of 2 =====================================
# Same install protocol and pins as kaggle_tier0.py. RESTART THE KERNEL after this cell.
# !pip install "transformers==4.57.6" "optimum[onnxruntime]" onnx onnxruntime \
#     datasets scikit-learn sentencepiece --quiet
# !pip check || echo "NOTE: pip check reported conflicts above — read them before continuing"
#
# ============================ CELL 2 of 2 =====================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # same reason as kaggle_tier0.py 3r

import json, hashlib, shutil, time
from pathlib import Path
import numpy as np, torch
import transformers as _tf
assert _tf.__version__ == "4.57.6", f"transformers {_tf.__version__} != pinned 4.57.6"
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from datasets import load_dataset
from sklearn.metrics import f1_score

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
IN = Path("/kaggle/input")
MODEL, MAX_LENGTH, SEEDS, LOSS_ARM = "microsoft/deberta-v3-base", 512, [1, 2, 3], "ce"
DEV2000_SHA12 = "77d3341979f2"

# ---- split guard: identical to kaggle_tier0.py, applied to the NAME before any data --
REPORTING_SPLITS = {"test_3000", "test"}
SELECTION_SPLITS = {"train_holdout_3000", "train"}
THRESHOLD_SPLITS = {"dev_2000", "validation", "dev"}
def assert_threshold_split(name):
    if name in REPORTING_SPLITS: raise RuntimeError(f"REFUSING: {name!r} is a REPORTING split")
    if name in SELECTION_SPLITS: raise RuntimeError(f"REFUSING: {name!r} is a SELECTION split")
    if name not in THRESHOLD_SPLITS: raise RuntimeError(f"unknown threshold split {name!r}")
    return name

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

# ---- LOCATE THE CHECKPOINTS. Loudly, before anything expensive. -----------------------
# A missing checkpoint here must stop the run, not fall through to a fresh
# `from_pretrained(MODEL)` — that would silently produce dev logits from an UNTRAINED
# encoder, which is 3e's class exactly: correctly-named arrays, plausible file sizes,
# and a router calibrated on noise.
def find_fp32(seed):
    name = f"fp32_{LOSS_ARM}_{seed}"
    cands = [WORK / name] + sorted(IN.glob(f"*/{name}")) + sorted(IN.glob(f"*/*/{name}"))
    for c in cands:
        if (c / "config.json").exists() and (any(c.glob("*.safetensors")) or any(c.glob("*.bin"))):
            return c
    raise FileNotFoundError(
        f"{name}/ not found. Looked in: {[str(x) for x in cands]}. Attach the previous "
        f"Tier 0 run's /kaggle/working as a notebook input. REFUSING to fall back to the "
        f"pretrained encoder: that would write dev logits from an untrained model under "
        f"the same filenames.")

CKPT = {s: find_fp32(s) for s in SEEDS}
for s, p in CKPT.items(): print(f"  seed {s}: {p}")

ds = load_dataset("coastalcph/lex_glue", "ledgar")
assert len(ds["validation"]) == 10000, "validation split drifted"
NAMES = list(ds["train"].features["label"].names); assert len(NAMES) == 100

assert_threshold_split("dev_2000")
DEV_IDX = stratified(list(ds["validation"]["label"]), 2000, 20260907)
_dsha = hash_texts(list(ds["validation"].select(DEV_IDX)["text"]))
assert _dsha.startswith(DEV2000_SHA12), (
    f"dev_2000 sha {_dsha[:12]} != {DEV2000_SHA12}. STOP — a threshold calibrated on a "
    f"different calibration set is uncomparable to everything else in the report.")
print(f"dev_2000 verified {_dsha[:16]}… ({len(DEV_IDX)} rows)")

_dev_labels = np.array([int(ds["validation"][i]["label"]) for i in DEV_IDX])
DEV_ABSENT = sorted(set(range(100)) - set(_dev_labels.tolist()))
print(f"  dev_2000 covers {100-len(DEV_ABSENT)}/100 classes; absent: "
      f"{[NAMES[c] for c in DEV_ABSENT]} — macro-F1 on dev averages "
      f"{100-len(DEV_ABSENT)} classes, NOT 100.")

TEST3000_SHA12 = "e719c1109069"
TEST_IDX = stratified(list(ds["test"]["label"]), 3000, 20260907)
_tsha = hash_texts(list(ds["test"].select(TEST_IDX)["text"]))
assert _tsha.startswith(TEST3000_SHA12), f"test_3000 sha {_tsha[:12]} != {TEST3000_SHA12}"
print(f"test_3000 verified {_tsha[:16]}… ({len(TEST_IDX)} rows)")

tok = AutoTokenizer.from_pretrained(MODEL)
def prep(split, idx):
    sub = ds[split].select(idx)
    enc = tok(list(sub["text"]), truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = [int(x) for x in sub["label"]]
    return [dict(zip(enc, v)) for v in zip(*enc.values())]
dev_ds = prep("validation", DEV_IDX)

def macro(y, pred): return f1_score(y, pred, average="macro", zero_division=0)

def onnx_predict(model_dir, texts, max_length, batch=1):
    """Run an ONNX graph. Both discriminator arms use THIS function, so a disagreement
    between them cannot be an artefact of two different harnesses."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(next(Path(model_dir).glob("*.onnx"))),
                                providers=["CPUExecutionProvider"])
    names_in = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], truncation=True, max_length=max_length,
                  padding=True, return_tensors="np")
        out.append(sess.run(None, {k: v.astype(np.int64)
                                   for k, v in enc.items() if k in names_in})[0])
        if i % 500 == 0: print(f"    onnx {i}/{len(texts)}", flush=True)
    return np.concatenate(out, 0)

def onnx_env():
    """The INT8 kernel's EXECUTION environment — recorded nowhere in the first run,
    which is why its collapse was un-diagnosable from its own output (3ah)."""
    import onnxruntime as ort
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError as e:
        flags = f"__unreadable__ {e}"
    if flags.startswith("__unreadable__"):
        fset, model = None, None                 # None = not measured. NEVER False.
    else:
        fline = next((l for l in flags.splitlines() if l.startswith("flags")), "")
        fset = set(fline.split(":", 1)[-1].split())
        model = next((l.split(":", 1)[1].strip() for l in flags.splitlines()
                      if l.startswith("model name")), None)
    def has(f): return None if fset is None else (f in fset)
    env = {"onnxruntime_version": ort.__version__,
           "available_providers": ort.get_available_providers(),
           "avx512_vnni": has("avx512_vnni"), "avx512f": has("avx512f"),
           "avx2": has("avx2"), "cpuinfo_readable": fset is not None, "model_name": model}
    print(f"  ONNX ENV: ort {env['onnxruntime_version']} providers={env['available_providers']}")
    print(f"  ONNX ENV: cpu={env['model_name']!r} avx512_vnni={env['avx512_vnni']} "
          f"avx512f={env['avx512f']} avx2={env['avx2']}")
    return env

y3 = np.array([int(ds["test"][i]["label"]) for i in TEST_IDX])
_T0 = time.time()
for seed in SEEDS:
    t_seed = time.time()
    print(f"\n=== seed {seed} — dev inference + E3 discriminator (NO TRAINING) ===", flush=True)
    fp32 = CKPT[seed]
    model = AutoModelForSequenceClassification.from_pretrained(str(fp32), num_labels=100)
    tr = Trainer(model=model,
        args=TrainingArguments(output_dir=str(WORK / f"tmp_dev_{seed}"),
            per_device_eval_batch_size=64, fp16=True, report_to=[],
            dataloader_num_workers=2),
        data_collator=DataCollatorWithPadding(tok))

    dev = tr.predict(dev_ds)
    # E5's key convention (src/router/load_logits.py _PREFIX["dev"] == "dev"): the arrays
    # are dev_logits / dev_labels, NOT dev_2000_logits. The first run's INT8 npz used the
    # dev_2000_ prefix for the same quantity and the loader cannot read it — one name per
    # quantity, chosen here to match the consumer.
    np.savez_compressed(WORK / f"dev_logits_{LOSS_ARM}_seed{seed}.npz",
                        dev_logits=dev.predictions, dev_labels=dev.label_ids,
                        dev_2000_indices=np.array(DEV_IDX),
                        dev_absent_classes=np.array(DEV_ABSENT))
    dev_macro = macro(dev.label_ids, dev.predictions.argmax(-1))
    print(f"  dev_2000 FP32 macro-F1 {dev_macro:.4f} over {100-len(DEV_ABSENT)} classes "
          f"(convenience number; src/eval/score_tier0.py is authoritative — 3af)")
    del tr, model; torch.cuda.empty_cache()

    # ---- E3 discriminator, three arms, identical rows, one code path (3ah) -----------
    onnx_dir, int8_dir = WORK / f"onnx_{LOSS_ARM}_{seed}", WORK / f"int8_{LOSS_ARM}_{seed}"
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    if not onnx_dir.exists():
        print("  exporting ONNX FP32 (discriminator arm 2 — kept until MEASURED)…", flush=True)
        ORTModelForSequenceClassification.from_pretrained(str(fp32), export=True
            ).save_pretrained(str(onnx_dir))
        tok.save_pretrained(str(onnx_dir))
    if not int8_dir.exists():
        # `arm64` is NOT an ISA-targeting choice: .arm64/.avx512/.avx512_vnni are equal on
        # all 15 dataclass fields (3e instance 6). Kept for byte-identity with the run
        # being diagnosed, NOT because it targets the Mac Mini.
        print("  quantising INT8…", flush=True)
        ORTQuantizer.from_pretrained(str(onnx_dir)).quantize(save_dir=str(int8_dir),
            quantization_config=AutoQuantizationConfig.arm64(is_static=False, per_channel=True))
        tok.save_pretrained(str(int8_dir))

    env = onnx_env()
    texts3000 = [ds["test"][i]["text"] for i in TEST_IDX]
    print("  arm 2/3: ONNX-FP32 on test_3000…", flush=True)
    onnx_fp32_logits = onnx_predict(onnx_dir, texts3000, MAX_LENGTH)
    print("  arm 3/3: ONNX-INT8 on test_3000…", flush=True)
    int8_logits = onnx_predict(int8_dir, texts3000, MAX_LENGTH)
    a2, a3 = onnx_fp32_logits.argmax(-1), int8_logits.argmax(-1)

    # INT8 dev logits too — deployed precision. Written to a SEPARATE file so a noise-
    # valued array can never be mistaken for the FP32 one E5 calibrates on.
    print("  INT8 on dev_2000…", flush=True)
    dev_int8 = onnx_predict(int8_dir, [ds["validation"][i]["text"] for i in DEV_IDX], MAX_LENGTH)
    np.savez_compressed(WORK / f"dev_logits_int8_{LOSS_ARM}_seed{seed}.npz",
                        dev_logits=dev_int8, dev_labels=_dev_labels,
                        dev_2000_indices=np.array(DEV_IDX),
                        dev_absent_classes=np.array(DEV_ABSENT))
    shutil.rmtree(onnx_dir, ignore_errors=True)   # ~740MB, and now MEASURED

    out = {"seed": seed, "loss_arm": LOSS_ARM, "checkpoint": str(fp32),
           "retrained": False, "max_length": MAX_LENGTH,
           "dev_2000_fp32_macro_f1_convenience": float(dev_macro),
           "dev_2000_absent_classes": DEV_ABSENT,
           "test_3000_onnx_fp32": {"macro_f1": macro(y3, a2),
                                   "accuracy": float((a2 == y3).mean())},
           "test_3000_int8": {"macro_f1": macro(y3, a3),
                              "accuracy": float((a3 == y3).mean())},
           "e3_discriminator": {"onnx_fp32_vs_int8_argmax_agree": float((a2 == a3).mean()),
                                "onnx_env": env},
           "seed_wall_clock_s": round(time.time() - t_seed, 1)}
    (WORK / f"tier0_dev_{LOSS_ARM}_seed{seed}.json").write_text(json.dumps(out, indent=2))
    print(f"  seed {seed} done in {out['seed_wall_clock_s']}s  "
          f"ONNX-FP32 {out['test_3000_onnx_fp32']['macro_f1']:.4f} | "
          f"INT8 {out['test_3000_int8']['macro_f1']:.4f} | "
          f"agree {out['e3_discriminator']['onnx_fp32_vs_int8_argmax_agree']:.4f}", flush=True)

print(f"\nALL SEEDS DONE in {(time.time()-_T0)/60:.1f} min — download "
      f"dev_logits_*.npz, dev_logits_int8_*.npz, tier0_dev_*.json and int8_ce_*/")
