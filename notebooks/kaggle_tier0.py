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
# ============================ CELL 1 of 2 =====================================
# RUN THIS CELL, THEN **RESTART THE KERNEL**, THEN RUN CELL 2.
# pip cannot replace a package the kernel has already imported; a half-replaced
# package is what produced the peft 'velora_config' crash in the Tier 1 notebook.
# Output is NOT suppressed — resolver errors must be visible.
#
# transformers is pinned BELOW 4.58 here, unlike Tier 1: `optimum-onnx` declares
# `transformers<4.58.0,>=4.36`, and this notebook needs optimum for the ONNX export.
# Kaggle ships transformers 5.0.0, so this IS a downgrade — check the output.
!pip install "transformers==4.57.6" "optimum[onnxruntime]" onnx onnxruntime \
    "datasets>=2.19" sentencepiece protobuf scikit-learn
!pip check || echo "NOTE: pip check reported conflicts above — read them before continuing"
print("\n" + "=" * 70)
print("NOW RESTART THE KERNEL, THEN RUN CELL 2.")
print("=" * 70)

# ============================ CELL 2 of 2 =====================================

# ONE GPU, PINNED BEFORE TORCH IS IMPORTED. This must be the first executable line:
# torch reads CUDA_VISIBLE_DEVICES when it initialises CUDA, and setting it after
# `import torch` is a no-op that leaves no trace.
#
# Kaggle's "GPU T4 x2" gives two devices, and on two devices this notebook trains at the
# WRONG BATCH SIZE without saying so:
#
#   1. TrainingArguments: train_batch_size = per_device_train_batch_size * max(1, n_gpu).
#      BS is 16, so two devices make the effective batch 32 against the 16 registered in
#      PREREGISTRATION 3t. Nothing in the output reports this.
#   2. transformers Trainer._wrap_model, line ~1665:
#          if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
#              model = nn.DataParallel(model)
#      DeBERTa is neither 4-bit nor 8-bit, so the guard passes and the model is wrapped.
#      Unlike Tier 1's 4-bit case this does not crash — DataParallel on a standard fp16
#      model works — which is what makes it the more dangerous of the two.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---- PREFLIGHT: verify signatures BEFORE the dataset downloads or weights load ----
import inspect

def _params(o):
    t = o.__init__ if inspect.isclass(o) else o
    return set(inspect.signature(t).parameters) | set(getattr(o, "__dataclass_fields__", {}))

def _accepts_var_kwargs(o):
    t = o.__init__ if inspect.isclass(o) else o
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in inspect.signature(t).parameters.values())

def _require(cls, kwargs, label):
    if _accepts_var_kwargs(cls):
        print(f"  PREFLIGHT: {label} takes **kwargs — NOT VERIFIABLE ({len(kwargs)} unchecked)")
        return
    missing = [k for k in kwargs if k not in _params(cls)]
    if missing:
        raise RuntimeError(f"{label} does not accept {missing}. Available: "
                           f"{sorted(n for n in _params(cls) if not n.startswith('_'))}")
    print(f"  PREFLIGHT: {label} accepts all {len(kwargs)} kwargs")

import os, json, gc, hashlib, random, shutil, threading, time
from pathlib import Path
from collections import Counter
import numpy as np, torch
from torch import nn
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, TrainerCallback,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import transformers as _tf
print(f"versions: transformers {_tf.__version__}")
# optimum-onnx requires transformers<4.58; a mismatch here means the pins did not
# take, which almost always means the kernel was not restarted after CELL 1.
_v = tuple(int(x) for x in _tf.__version__.split(".")[:2] if x.isdigit())
if not ((4, 36) <= _v < (4, 58)):
    raise RuntimeError(
        f"transformers {_tf.__version__} is outside optimum-onnx's supported range "
        f"(>=4.36,<4.58). Run CELL 1 and RESTART THE KERNEL. PREFLIGHT previously "
        f"passed while running versions that were never introspected — this check "
        f"exists so that cannot recur.")
print("  PREFLIGHT: transformers version is inside optimum-onnx's supported range")
print("PREFLIGHT — validating signatures before anything expensive")
_require(TrainingArguments, ["output_dir","seed","num_train_epochs","learning_rate",
    "per_device_train_batch_size","per_device_eval_batch_size","eval_strategy",
    "save_strategy","load_best_model_at_end","metric_for_best_model","greater_is_better",
    "save_total_limit","fp16","report_to","logging_steps","dataloader_num_workers"],
    "TrainingArguments")
_require(Trainer, ["model","args","train_dataset","eval_dataset","data_collator",
                   "compute_metrics"], "Trainer")
_require(DataCollatorWithPadding, ["tokenizer"], "DataCollatorWithPadding")
from optimum.onnxruntime.configuration import AutoQuantizationConfig as _AQC
from optimum.onnxruntime import ORTQuantizer as _ORTQ
assert hasattr(_AQC, "arm64"), "AutoQuantizationConfig.arm64 missing in this optimum"
_require(_AQC.arm64, ["is_static","per_channel"], "AutoQuantizationConfig.arm64")
_require(_ORTQ.quantize, ["save_dir","quantization_config"], "ORTQuantizer.quantize")
# Device count is a REGISTERED experimental parameter (PREREGISTRATION 3t), not a
# session setting: a second visible device multiplies the effective batch via
# train_batch_size = per_device_train_batch_size * max(1, n_gpu), and makes Trainer wrap
# the model in DataParallel (its guard tests is_loaded_in_8bit only, and DeBERTa is
# neither 4-bit nor 8-bit). This check deliberately references NO training constant:
# they are defined below, and a PREFLIGHT check that reads a name defined later dies of
# NameError having verified nothing.
import torch as _torch
N_VISIBLE_GPUS = _torch.cuda.device_count()
print(f"  PREFLIGHT: visible GPUs={N_VISIBLE_GPUS} "
      f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
assert _torch.cuda.is_available(), "no GPU — set Accelerator to GPU"
assert N_VISIBLE_GPUS == 1, (
    f"{N_VISIBLE_GPUS} GPUs visible. CUDA_VISIBLE_DEVICES must pin exactly one BEFORE "
    f"torch is imported. E1/E2/E3 are registered at 1 visible GPU (PREREGISTRATION 3t); "
    f"with {N_VISIBLE_GPUS} the effective batch is multiplied by {N_VISIBLE_GPUS} and "
    f"Trainer wraps the model in DataParallel, both silently.")
print("PREFLIGHT PASSED\n")

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
MODEL = "microsoft/deberta-v3-base"
SEEDS = [1, 2, 3]
LOSS_ARM = "ce"            # E1 baseline. E2 arms: sqrt_inv_freq | effective_number | inv_freq
MAX_LENGTH = 512
EPOCHS, LR, BS = 3, 2e-5, 16

# ---- E1b (PREREGISTRATION, experiment E1b): 10 epochs, ONE VARIABLE changed from E1 --
# Set E1B = True to run it. Everything except EPOCHS stays byte-identical to E1 — that is
# the whole design, so that a result is attributable to the epoch count and nothing else.
#
# RUN_TAG, not LOSS_ARM, names the artefacts. Two reasons, both load-bearing:
#   1. COLLISION. E1's outputs are tier0_ce_seed*.json / logits_ce_seed*.npz / fp32_ce_*.
#      Re-running with the same tag would hit E1's completion markers and SKIP every
#      seed, or overwrite E1's artefacts with E1b's. E1's measurements must survive
#      exactly as taken.
#   2. `weights()` dispatches on the LOSS ARM and raises KeyError on anything that is not
#      ce / inv_freq / sqrt_inv_freq / effective_number. Overloading LOSS_ARM with a run
#      label would either crash there or, worse, silently select a different loss.
# E1b's loss is unchanged CE; only the artefact namespace differs.
E1B = False
RUN_TAG = LOSS_ARM
if E1B:
    EPOCHS, RUN_TAG, SEEDS = 10, f"{LOSS_ARM}10ep", [1]   # seed 1 GATES seeds 2-3 (E1b)
    print(f"E1b MODE: EPOCHS={EPOCHS}, RUN_TAG={RUN_TAG!r}, SEEDS={SEEDS} — "
          f"seed 1 is a GATE, not a result (hard rule 2 needs >=3 seeds)")
assert (RUN_TAG == LOSS_ARM) or E1B, "RUN_TAG may only diverge from LOSS_ARM under E1B"
# gradient_accumulation_steps is not set below, so it is the library default of 1 and
# the effective train batch is BS * 1 * n_gpu. PREREGISTRATION 3t registers E1/E2/E3 at
# an effective batch of 16 on ONE visible GPU; both halves are asserted, because either
# one alone permits the other to drift.
REGISTERED_EFFECTIVE_BATCH = 16   # PREREGISTRATION 3t, E1/E2/E3
GA = 1                            # TrainingArguments default; stated so it is visible
assert BS * GA * N_VISIBLE_GPUS == REGISTERED_EFFECTIVE_BATCH, (
    f"effective batch is BS {BS} x GA {GA} x n_gpu {N_VISIBLE_GPUS} = "
    f"{BS * GA * N_VISIBLE_GPUS}, but E1/E2/E3 are registered at "
    f"{REGISTERED_EFFECTIVE_BATCH} in PREREGISTRATION 3t. Changing it is an amendment, "
    f"not a tuning decision.")
print(f"  effective train batch = BS {BS} x GA {GA} x n_gpu {N_VISIBLE_GPUS} = "
      f"{BS * GA * N_VISIBLE_GPUS}, matching the registered {REGISTERED_EFFECTIVE_BATCH}")
HOLDOUT_SHA12 = "97eebc04d0c7"    # committed train_holdout_3000
TEST3000_SHA12 = "e719c1109069"   # committed test_3000
DEV2000_SHA12 = "77d3341979f2"    # committed dev_2000 — THRESHOLD-ONLY (hard rule 1)

# ---------------- split guard (duplicated so the cell is standalone) ---------
SELECTION_SPLITS = {"train_holdout_3000", "train"}
THRESHOLD_SPLITS = {"dev_2000", "validation", "dev"}
REPORTING_SPLITS = {"test_3000", "test_stratified_764", "test"}
def assert_threshold_split(name):
    """Hard rule 1, enforced here as well as in src/router/calibrate.py.

    dev_2000 is calibration-ONLY. It must never be used to select a model (that is what
    train_holdout_3000 is for) and never to report (that is test_3000)."""
    if name in REPORTING_SPLITS: raise RuntimeError(f"REFUSING: {name!r} is a REPORTING split")
    if name in SELECTION_SPLITS: raise RuntimeError(f"REFUSING: {name!r} is a SELECTION split")
    if name not in THRESHOLD_SPLITS: raise RuntimeError(f"unknown threshold split {name!r}")
    return name

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

# test_3000 is a proportional stratified sample spanning the WHOLE 10k test split
# (indices 1..9992), NOT the first 3000 rows. We predict on all 10k so any manifest can
# be scored offline by row index, and additionally report the test_3000 subset here,
# because E1's accept rule is stated on test_3000.
TEST_IDX = stratified(list(ds["test"]["label"]), 3000, 20260907)
_tsha = hash_texts(list(ds["test"].select(TEST_IDX)["text"]))
assert _tsha.startswith(TEST3000_SHA12), f"test_3000 sha {_tsha[:12]} != {TEST3000_SHA12}"
print(f"test_3000 verified {_tsha[:16]}… ({len(TEST_IDX)} rows, "
      f"range {min(TEST_IDX)}..{max(TEST_IDX)})")

# dev_2000 — the ROUTER THRESHOLD CALIBRATION set (hard rule 1). Regenerated here rather
# than read from configs/manifests (the notebook is standalone on Kaggle) and verified
# against the committed sha exactly as train_holdout_3000 and test_3000 are. Verified
# locally that this stratified() call reproduces the manifest's 2,000 indices and its
# text_sha256 byte for byte before the constant was written down.
#
# The split guard is applied to the NAME before the data is touched, so dev cannot be
# mistaken for a selection or reporting split at any point in this cell.
assert_threshold_split("dev_2000")
DEV_IDX = stratified(list(ds["validation"]["label"]), 2000, 20260907)
_dsha = hash_texts(list(ds["validation"].select(DEV_IDX)["text"]))
assert _dsha.startswith(DEV2000_SHA12), (
    f"dev_2000 sha {_dsha[:12]} != {DEV2000_SHA12}. The calibration set does not match "
    f"the committed manifest, so any threshold calibrated on it would be uncomparable "
    f"to everything else in the report. STOP — do not work around this.")
print(f"dev_2000 verified {_dsha[:16]}… ({len(DEV_IDX)} rows, "
      f"range {min(DEV_IDX)}..{max(DEV_IDX)})")
# dev_2000 covers 99 of 100 classes — `Books` (class 14) is too rare to earn a
# proportional seat. This is deliberate (PREREGISTRATION 3a): a >=1-per-class floor would
# distort the class distribution away from realistic traffic and bias the calibrated
# threshold. Recorded in the npz so nothing downstream has to rediscover it, and so E5
# must pass allow_absent_classes=True KNOWINGLY rather than by accident.
_dev_labels = np.array([int(ds["validation"][i]["label"]) for i in DEV_IDX])
DEV_ABSENT = sorted(set(range(100)) - set(_dev_labels.tolist()))
print(f"  dev_2000 covers {100 - len(DEV_ABSENT)}/100 classes; absent: "
      f"{[NAMES[c] for c in DEV_ABSENT]} — any macro-F1 on dev is an average over "
      f"{100 - len(DEV_ABSENT)} classes, NOT 100. State this wherever it appears.")

train_ds, sel_ds = prep("train", fit_idx), prep("train", hold)
dev_ds = prep("validation", DEV_IDX)
test_ds = prep("test", list(range(len(ds["test"]))))   # all 10k; subset offline by index
print(f"fit {len(train_ds):,} | select {len(sel_ds):,} | dev {len(dev_ds):,} | "
      f"test {len(test_ds):,}")

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

def macro(y, pred): return f1_score(y, pred, average="macro", zero_division=0)

def onnx_predict(model_dir, texts, max_length, batch=1):
    """Run an ONNX graph. E1's accept rule attaches to the INT8 one, not to FP32.

    Takes any directory holding one *.onnx, because the E3 discriminator (below) needs
    to run the FP32 ONNX graph through this SAME code path. If FP32-ONNX and INT8 are
    scored by different code, a disagreement between them does not isolate the
    quantised kernel — it could be the harness."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(next(Path(model_dir).glob("*.onnx"))),
                                providers=["CPUExecutionProvider"])
    names_in = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], truncation=True, max_length=max_length,
                  padding=True, return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names_in}
        out.append(sess.run(None, feed)[0])
        if i % 500 == 0: print(f"    onnx {i}/{len(texts)}", flush=True)
    return np.concatenate(out, 0)

def onnx_env():
    """Capture the EXECUTION environment of the INT8 kernel, not just its config.

    E3 measured INT8 at chance (macro-F1 0.0037 vs FP32 0.7636) while the identical
    export, reproduced locally on arm64, agreed with FP32 to r=+0.9982. The surviving
    candidate is that `reduce_range=False` saturates on a NON-VNNI x86 kernel — a
    property of THIS HOST, which nothing in the run currently records. Unrecorded, the
    next run reproduces the failure and is equally unable to explain it."""
    import onnxruntime as ort
    flags = ""
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError as e:                       # not Linux; the field must say so, not lie
        flags = f"__unreadable__ {e}"
    if flags.startswith("__unreadable__"):
        fset, model = None, None            # None = not measured. NEVER False.
    else:
        fline = next((l for l in flags.splitlines() if l.startswith("flags")), "")
        fset = set(fline.split(":", 1)[-1].split())
        model = next((l.split(":", 1)[1].strip() for l in flags.splitlines()
                      if l.startswith("model name")), None)
    def has(f):
        return None if fset is None else (f in fset)
    env = {"onnxruntime_version": ort.__version__,
           "available_providers": ort.get_available_providers(),
           "avx512_vnni": has("avx512_vnni"), "avx512f": has("avx512f"),
           "avx2": has("avx2"), "cpuinfo_readable": fset is not None,
           "model_name": model}
    print(f"  ONNX ENV: ort {env['onnxruntime_version']} providers={env['available_providers']}")
    print(f"  ONNX ENV: cpu={env['model_name']!r} avx512_vnni={env['avx512_vnni']} "
          f"avx512f={env['avx512f']} avx2={env['avx2']}")
    return env

# ======================== LIVENESS INSTRUMENTATION ===========================
# Ported from kaggle_tier1.py after commit 1 there ran 2h22m with no output and a hang
# could not be distinguished from a silent log (PREREGISTRATION 3z).
#
# Tier 0 needs this MORE than Tier 1 did, not less: Tier 1's silence was diagnosed by
# ruling out DataLoader worker deadlock, because Tier 1 never sets
# dataloader_num_workers. TIER 0 SETS IT TO 2, so worker processes actually exist here
# and that candidate is NOT ruled out.
#
# The heartbeat is a DAEMON THREAD, not a Trainer callback, and that is the point: a
# callback only fires when the loop reaches on_step_end, so it is silent both when
# training is hung and when it is merely quiet. A thread keeps printing while the main
# thread is blocked. Silence from it means the process is dead or the log pipe is
# broken; output from it while step does not advance means the loop is stuck.
_HB = {"phase": "cell-2 start", "step": None, "total": None,
       "t0": time.time(), "t_train": None}

def _heartbeat_loop(interval_s=120):
    while True:
        time.sleep(interval_s)
        el = (time.time() - _HB["t0"]) / 60
        st, tot, t_tr = _HB["step"], _HB["total"], _HB["t_train"]
        line = f"[heartbeat {el:7.1f} min] phase={_HB['phase']} step={st}/{tot}"
        if st and t_tr and st > 0:
            rate = st / (time.time() - t_tr)
            line += f" | {rate:.3f} it/s"
            if tot:
                line += f" | {(tot - st) / rate / 60:.0f} min to step {tot}"
        print(line, flush=True)

threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat").start()
print("[heartbeat] thread started; prints every 2 min regardless of Trainer state",
      flush=True)


class FlushingLog(TrainerCallback):
    """Trainer's own log path writes through tqdm to STDERR; this re-emits on stdout
    with flush. Runs alongside ProgressCallback, so nothing is lost if stderr works."""
    def on_train_begin(self, args, state, control, **kw):
        _HB["phase"] = "training"; _HB["total"] = int(state.max_steps)
        _HB["t_train"] = time.time(); _HB["step"] = int(state.global_step)
        print(f"[train] begin at step {state.global_step} of {state.max_steps}", flush=True)
    def on_step_end(self, args, state, control, **kw):
        _HB["step"] = int(state.global_step)
    def on_log(self, args, state, control, logs=None, **kw):
        print(f"[log step {state.global_step}] {logs}", flush=True)
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        print(f"[eval step {state.global_step}] {metrics}", flush=True)
    def on_save(self, args, state, control, **kw):
        print(f"[save] checkpoint at step {state.global_step}", flush=True)


# ============================ RESTORE ========================================
# /kaggle/working does NOT carry over between commits — measured, 3v. Each commit runs in
# a fresh container and the previous run's working directory becomes THAT VERSION'S
# OUTPUT. To carry results forward: File -> Add input -> Your Work -> Notebook, and pick
# this notebook's latest version. It mounts read-only at
#     /kaggle/input/notebooks/<username>/<notebook-slug>/
# The slug is globbed, never hardcoded: a hardcoded path that stopped matching would
# silently start fresh, which here means re-running seeds that are already done.
#
# Tier 0 needs this for two reasons even though all three seeds fit in ONE commit (3y):
#   1. a commit that dies or overruns leaves nothing behind otherwise;
#   2. E2's second arm (LOSS_ARM = "sqrt_inv_freq") is a SEPARATE commit run after C3
#      reports, and the CE results must be carried into it so one final output holds
#      every arm.
# There is deliberately NO step budget here — see 3y. Tier 0 is 3.0-6.6h for all three
# seeds against a 9h design cap, so a budget would be machinery that never fires.
_IN_NB = Path("/kaggle/input/notebooks")
_RESTORE_GLOBS = ["ck_*", "fp32_*", "int8_*", "onnx_*",
                  "tier0_*_seed*.json", "logits_*_seed*.npz", "int8_logits_*_seed*.npz"]
_attached = sorted(d for d in _IN_NB.glob("*/*") if d.is_dir()) if _IN_NB.exists() else []
print("\n=== RESTORE ===")
print(f"attached notebook inputs: {[str(d) for d in _attached] or 'NONE'}")
_sources = [d for d in _attached if any(any(d.glob(g)) for g in _RESTORE_GLOBS)]
if len(_sources) > 1:
    raise RuntimeError(
        f"{len(_sources)} attached notebook inputs contain Tier 0 artefacts: "
        f"{[str(d) for d in _sources]}. Which is current is ambiguous, and guessing "
        f"could resurrect a superseded result. Detach all but the newest.")
if _sources:
    src = _sources[0]; print(f"restoring from {src}")
    _restored = []
    for g in _RESTORE_GLOBS:
        for item in sorted(src.glob(g)):
            dst = WORK / item.name
            if dst.exists():
                print(f"    {item.name}: already present, not overwritten"); continue
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, dst)
            _restored.append(item.name)
    print(f"restored {len(_restored)}: {_restored}")
elif _attached:
    raise RuntimeError(
        f"A notebook input is attached but contains no Tier 0 artefacts "
        f"{_RESTORE_GLOBS}. Refusing to start fresh: that is indistinguishable from a "
        f"working carry-forward until every seed has been recomputed.\n"
        f"Attached: {[str(d) for d in _attached]}\n"
        f"Contents: { {str(d): sorted(x.name for x in d.iterdir())[:20] for d in _attached} }")
else:
    print("no notebook input attached -> first commit for this arm, starting fresh.")
print(f"completion markers present: "
      f"{sorted(f.name for f in WORK.glob('tier0_*_seed*.json')) or 'none'}")
print("=== RESTORE OK ===\n")

_T_START = time.time()
for seed in SEEDS:
    _t_seed = time.time()
    done = WORK / f"tier0_{RUN_TAG}_seed{seed}.json"
    if done.exists():
        # A completion marker written BEFORE the E3 discriminator existed describes a
        # seed that is trained but not diagnosed. Skipping on the marker alone would
        # silently produce a run in which the discriminator never executes for the very
        # seeds that motivated it — 3e's class exactly: normal-looking output, nothing
        # measured. Re-enter instead; fp32 exists, so training is skipped and only the
        # ~16k prediction rows and the export are redone.
        if "e3_discriminator" in json.loads(done.read_text()):
            print(f"seed {seed}: complete WITH discriminator, skipping"); continue
        print(f"seed {seed}: complete but PRE-DISCRIMINATOR — re-entering to score the "
              f"three E3 arms. Training is skipped (fp32 exists); nothing is retrained.")
        done.unlink()
    ckpt_dir = WORK / f"ck_{RUN_TAG}_{seed}"
    fp32 = WORK / f"fp32_{RUN_TAG}_{seed}"
    # RESUME GAP FIX: ONNX export, INT8 quantisation and two evaluations run AFTER
    # training. If the session dies in there, fp32 already exists and retraining the
    # whole seed would waste an hour.
    skip_training = fp32.exists()
    resume = (not skip_training) and ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint-*"))
    print(f"\n=== seed {seed} ("
          f"{'fp32 EXISTS, skipping training' if skip_training else ('RESUMING' if resume else 'fresh')}) ===")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    w = weights(LOSS_ARM)
    wt = None if w is None else torch.tensor(w, dtype=torch.float32)
    if w is not None:
        print(f"  class weights: min {w.min():.3f} max {w.max():.3f} ratio {w.max()/w.min():.1f}x")

    class WT(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            lb = inputs.pop("labels"); out = model(**inputs)
            loss = nn.CrossEntropyLoss(
                weight=None if wt is None else wt.to(out.logits.device))(out.logits, lb)
            return (loss, out) if return_outputs else loss

    src_model = str(fp32) if skip_training else MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        src_model, num_labels=100, id2label=dict(enumerate(NAMES)),
        label2id={n: i for i, n in enumerate(NAMES)})

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
    tr.add_callback(FlushingLog())

    if not skip_training:
        _HB["phase"] = f"seed {seed}: trainer.train() entered"
        tr.train(resume_from_checkpoint=resume)
        tr.save_model(str(fp32)); tok.save_pretrained(str(fp32))
        shutil.rmtree(ckpt_dir, ignore_errors=True)   # fp32 saved — never retrain

    sel = tr.predict(sel_ds); tst = tr.predict(test_ds); dev = tr.predict(dev_ds)
    # dev_ arrays are what E5 calibrates router thresholds on (hard rule 1). They ride in
    # the same npz as sel_ and test_ so one file carries every split E5 needs and cannot
    # be paired with logits from a different seed or arm.
    np.savez_compressed(WORK / f"logits_{RUN_TAG}_seed{seed}.npz",
                        sel_logits=sel.predictions, sel_labels=sel.label_ids,
                        test_logits=tst.predictions, test_labels=tst.label_ids,
                        test_3000_indices=np.array(TEST_IDX),
                        dev_logits=dev.predictions, dev_labels=dev.label_ids,
                        dev_2000_indices=np.array(DEV_IDX),
                        dev_absent_classes=np.array(DEV_ABSENT))
    del tr, model; gc.collect(); torch.cuda.empty_cache()

    # --- ONNX + INT8. INT8 is the DEPLOYED precision; E1 attaches to it, E3 is the delta.
    onnx_dir, int8_dir = WORK / f"onnx_{RUN_TAG}_{seed}", WORK / f"int8_{RUN_TAG}_{seed}"
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    if not onnx_dir.exists():
        # The FP32 ONNX export is REQUIRED, not incidental: it is the middle arm of the
        # E3 discriminator. Previously it was deleted immediately after quantising, which
        # is why the first INT8 failure could not be localised from the saved artefacts —
        # only torch-FP32 and INT8 survived, and those differ in TWO steps (export AND
        # quantisation), so their disagreement isolated neither.
        print("  exporting ONNX FP32…")
        ORTModelForSequenceClassification.from_pretrained(str(fp32), export=True
            ).save_pretrained(str(onnx_dir))
        tok.save_pretrained(str(onnx_dir))
    if not int8_dir.exists():
        # `arm64` is NOT an ISA-targeting choice. AutoQuantizationConfig.arm64,
        # .avx512 and .avx512_vnni are IDENTICAL in every parameter (QInt8 weights,
        # QUInt8 activations, reduce_range=False, per_channel=True) — verified by
        # comparing the dataclasses, see PREREGISTRATION 3e instance 6. The earlier
        # comment here claimed this factory produced an arm-specific graph for the Mac
        # Mini. It does not; the bytes are the same as avx512's.
        print("  quantising INT8…")
        ORTQuantizer.from_pretrained(str(onnx_dir)).quantize(save_dir=str(int8_dir),
            quantization_config=AutoQuantizationConfig.arm64(is_static=False, per_channel=True))
        tok.save_pretrained(str(int8_dir))

    # --- score both precisions on test_3000, which is what E1's rule names ---
    y3 = np.array([int(ds["test"][i]["label"]) for i in TEST_IDX])
    fp32_test3000 = tst.predictions[TEST_IDX].argmax(-1)

    # ---- E3 DISCRIMINATOR (PREREGISTRATION 3ah) -------------------------------------
    # Three arms on IDENTICAL rows, so a disagreement localises to ONE step:
    #   torch-FP32 -> ONNX-FP32   isolates the EXPORT
    #   ONNX-FP32  -> ONNX-INT8   isolates the QUANTISED KERNEL
    # Diagnostic only. It does NOT rescue E3, whose falsification clause already fired;
    # it decides what re-verification means. DEPLOYMENT IS arm64 (Mac Mini), where this
    # exact export was reproduced locally at r=+0.9982 against FP32. So if the locus is
    # a non-VNNI x86 kernel, the broken thing is THIS MEASUREMENT ENVIRONMENT, not the
    # serving path — and E3 must be re-scored on the trained INT8 artefact on arm64
    # (scripts/score_int8_local.py) before any INT8 number is believed either way.
    env = onnx_env()
    print("  evaluating ONNX-FP32 on test_3000 (discriminator arm 2 of 3)…")
    onnx_fp32_logits = onnx_predict(onnx_dir, [ds["test"][i]["text"] for i in TEST_IDX],
                                    MAX_LENGTH)
    onnx_fp32_test3000 = onnx_fp32_logits.argmax(-1)
    print("  evaluating INT8 on test_3000…")
    int8_logits = onnx_predict(int8_dir, [ds["test"][i]["text"] for i in TEST_IDX], MAX_LENGTH)
    # INT8 dev logits too: INT8 is the DEPLOYED precision (E1, E3), so a threshold
    # calibrated on FP32 dev logits would be calibrated for a model that is never served.
    # 2,000 extra rows at batch 1 — seconds.
    dev_int8_logits = onnx_predict(int8_dir,
                                   [ds["validation"][i]["text"] for i in DEV_IDX], MAX_LENGTH)
    int8_test3000 = int8_logits.argmax(-1)
    shutil.rmtree(onnx_dir, ignore_errors=True)   # ~740MB, and now MEASURED, not assumed
    np.savez_compressed(WORK / f"int8_logits_{RUN_TAG}_seed{seed}.npz",
                        test_3000_logits=int8_logits, test_3000_indices=np.array(TEST_IDX),
                        onnx_fp32_test_3000_logits=onnx_fp32_logits,
                        dev_2000_logits=dev_int8_logits,
                        dev_2000_indices=np.array(DEV_IDX),
                        dev_absent_classes=np.array(DEV_ABSENT))

    out = {"model": MODEL, "loss_arm": LOSS_ARM, "run_tag": RUN_TAG, "experiment": "E1b" if E1B else "E1",
           "seed": seed, "max_length": MAX_LENGTH,
           "epochs": EPOCHS, "lr": LR, "selection_split": "train_holdout_3000",
           "test_manifest": "test_3000", "test_manifest_sha256": _tsha,
           "selection": metrics(sel),
           "test_3000_fp32": {"macro_f1": macro(y3, fp32_test3000),
                              "accuracy": float((fp32_test3000 == y3).mean())},
           "test_3000_int8": {"macro_f1": macro(y3, int8_test3000),
                              "accuracy": float((int8_test3000 == y3).mean())},
           "test_3000_onnx_fp32": {"macro_f1": macro(y3, onnx_fp32_test3000),
                                   "accuracy": float((onnx_fp32_test3000 == y3).mean())},
           "e3_discriminator": {
               "torch_fp32_vs_onnx_fp32_argmax_agree":
                   float((fp32_test3000 == onnx_fp32_test3000).mean()),
               "onnx_fp32_vs_int8_argmax_agree":
                   float((onnx_fp32_test3000 == int8_test3000).mean()),
               "onnx_env": env},
           "test_full10k_fp32": metrics(tst),
           "int8_dir": int8_dir.name}
    out["e3_int8_minus_fp32_macro_f1"] = (out["test_3000_int8"]["macro_f1"]
                                          - out["test_3000_fp32"]["macro_f1"])
    done.write_text(json.dumps(out, indent=2))
    # Per-seed wall clock. Tier 0 has NO step budget (3y): all three seeds are estimated
    # at 3.0-6.6h against a 9h cap, so a budget would be machinery that never fires. This
    # number is the check on that estimate — if a seed exceeds ~2.5h the pessimistic end
    # is real, and the remaining seeds should be split into their own commits by editing
    # SEEDS, before a commit hits the cap and loses its whole output (3v).
    _seed_h = (time.time() - _t_seed) / 3600
    print(f"  seed {seed} wall clock: {_seed_h:.2f}h  (estimate 1.0-2.2h; if >2.5h, run "
          f"the remaining seeds as separate commits)")
    print(json.dumps({k: out[k] for k in
          ("selection", "test_3000_fp32", "test_3000_int8", "e3_int8_minus_fp32_macro_f1")},
          indent=2))
    gc.collect(); torch.cuda.empty_cache()

print("\nALL SEEDS DONE — download /kaggle/working/*.json, *.npz and int8_* dirs")
