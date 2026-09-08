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
# ============================ CELL 1 of 2 =====================================
# RUN THIS CELL, THEN **RESTART THE KERNEL**, THEN RUN CELL 2.
#
# Why the restart is mandatory: pip cannot replace a package that the running kernel
# has already imported. The previous attempt reported `peft 0.19.1` while executing
# 0.20.0's `bnb.py` — a MIXED install, which crashed with
# `LoraConfig object has no attribute 'velora_config'` (that field exists in 0.20.0's
# LoraConfig and is referenced by 0.20.0's bnb.py; 0.19.1 has neither).
#
# Output is deliberately NOT suppressed. The previous line used `-q ... | tail -3`,
# which threw away the resolver errors that would have shown the pins failing.
#
# transformers is left at Kaggle's own version: trl 1.12.0 requires transformers
# >=4.56.2 with NO upper bound, and peft 0.20.0 has none either, so 5.0.0 is allowed.
# Downgrading it would mean uninstalling a preloaded package — exactly what produced
# the mixed install. ONLY peft is forced.
!pip install --upgrade --force-reinstall --no-deps "peft==0.20.0"
!pip install "trl==1.12.0" "datasets>=2.19" accelerate bitsandbytes sentencepiece scikit-learn
!pip check || echo "NOTE: pip check reported conflicts above — read them before continuing"
print("\n" + "=" * 70)
print("NOW RESTART THE KERNEL, THEN RUN CELL 2.")
print("  Kaggle: Run -> Restart & clear cell outputs   (or the session's Restart button)")
print("Without a restart the old peft stays loaded and cell 2 will crash in bnb.py.")
print("=" * 70)

# ============================ CELL 2 of 2 =====================================
# Everything below goes in a SECOND cell, run AFTER the kernel restart.

# ONE GPU, PINNED BEFORE TORCH IS IMPORTED. This must be the first executable line of
# the cell: torch reads CUDA_VISIBLE_DEVICES when it initialises CUDA, and setting it
# after `import torch` is a no-op that leaves no trace.
#
# Kaggle's "GPU T4 x2" gives two devices, and two things go wrong if both are visible:
#
#   1. transformers 5.0.0 Trainer._wrap_model, line ~1665:
#          if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
#              model = nn.DataParallel(model)
#      The guard tests is_loaded_in_8bit ONLY. Our model is loaded in 4-BIT, so the
#      guard does not fire and a bitsandbytes 4-bit model gets wrapped in DataParallel,
#      which replicates modules across devices. device_map={"": 0} does not prevent
#      this — it places the weights, it does not set n_gpu.
#   2. TrainingArguments: train_batch_size = per_device_train_batch_size * max(1, n_gpu).
#      With two devices the EFFECTIVE BATCH SILENTLY BECOMES 32 (4 x 4 x 2), not the
#      registered 16. That is a change to the experiment reported by nothing.
#
# Pinning to one device makes both impossible. A 1.5B model in 4-bit is ~1.1GB and was
# never going to need the second T4.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =============================================================================
# PREFLIGHT — runs FIRST, before the dataset downloads and before any weights load.
# A signature error must surface in ~10 seconds, not two minutes in, and never
# after hours of training.
#
# The rule this enforces: NEVER assume a third-party kwarg name. `SFTConfig` imports
# fine but renamed `max_seq_length` to `max_length`; checking that the class imports
# verified the wrong thing. Every kwarg below is checked against the INSTALLED
# signature, and the sequence-length name is RESOLVED at runtime rather than guessed.
# =============================================================================
import inspect

def _params(obj):
    """Accepted parameter names, covering dataclasses and **kwargs-style classes."""
    target = obj.__init__ if inspect.isclass(obj) else obj
    names = set(inspect.signature(target).parameters)
    names |= set(getattr(obj, "__dataclass_fields__", {}))
    return names

def _resolve(cls, candidates, what):
    """Pick whichever alias this version actually accepts. Raise if none do."""
    have = _params(cls)
    for c in candidates:
        if c in have:
            print(f"  PREFLIGHT: {cls.__name__}.{what} -> {c!r}")
            return c
    raise RuntimeError(
        f"{cls.__name__} accepts none of {candidates} for {what}. Available "
        f"parameters: {sorted(n for n in have if not n.startswith('_'))}")

def _accepts_var_kwargs(obj):
    target = obj.__init__ if inspect.isclass(obj) else obj
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in inspect.signature(target).parameters.values())

def _require(cls, kwargs, label):
    """Fail if a kwarg is not accepted. If the callable takes **kwargs, say so —
    introspection CANNOT verify it, and claiming otherwise would be the same
    mistake as checking that SFTConfig imports."""
    if _accepts_var_kwargs(cls):
        print(f"  PREFLIGHT: {label} takes **kwargs — NOT VERIFIABLE by introspection "
              f"({len(kwargs)} kwargs unchecked)")
        return
    missing = [k for k in kwargs if k not in _params(cls)]
    if missing:
        raise RuntimeError(
            f"{label} does not accept {missing}. Available: "
            f"{sorted(n for n in _params(cls) if not n.startswith('_'))}")
    print(f"  PREFLIGHT: {label} accepts all {len(kwargs)} kwargs")

import os, json, gc, hashlib, random, re, shutil, unicodedata
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
# max_seq_length lives on SFTConfig in current TRL, NOT TrainingArguments — passing it
# to TrainingArguments raises TypeError. Import defensively.
from trl import SFTConfig

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # record the actual model in PREREGISTRATION (E4)
# A 1.5B model in 4-bit is ~1GB and fits a single T4, so device_map="auto" would place
# it entirely on GPU 0 and leave GPU 1 idle. Pin it so the behaviour is explicit.
DEVICE_MAP = {"": 0}
SEEDS = [1, 2, 3]
# MAX_LEN bounded over the FULL 57,000-row fit set, not a sample. A 300-row head
# sample said max 1154; the true max is 2378 — LEDGAR is chronologically ordered so a
# head sample is not representative. Counts over the full set:
#   >1024: 754 rows (1.32%)   >1536: 17   >2048: 2   >2560: 0
# The LABEL sits at the end of every training example, so any truncation removes the
# answer. 2560 is the smallest bound with zero truncation; dynamic padding means the
# cost is paid only on the rare long batches (mean length is 607).
# PRECISION, stated once and asserted below. NOTHING here may be left to a library
# default: transformers 5.0.0 changed from_pretrained's dtype default from fp32 to
# "auto", which reads Qwen2.5's config torch_dtype (bfloat16). T4 is sm_75 and has no
# bf16 support, so every bf16 path on this hardware is a latent failure.
MODEL_DTYPE = torch.float16     # passed explicitly to from_pretrained; never "auto"
COMPUTE_DTYPE = torch.float16   # bnb_4bit_compute_dtype; must equal MODEL_DTYPE
USE_FP16, USE_BF16 = True, False        # AMP mode. bf16 is impossible on T4.

# MEMORY, on ONE 16GB T4 with gradient checkpointing on:
#   4-bit base ~1.1GB | LoRA r=16 over 7 modules = 18.5M params, fp32 weights+grads
#   +Adam states ~0.3GB | checkpointed layer boundaries at BS 4 x 2560 ~0.9GB.
#   The term that actually decides it is the lm_head logits: BS x seq x 151,936 vocab.
#   At the mean length (607) that is ~0.7GB; at a batch padded to 2378 (the longest
#   fit row) it is ~2.9GB in fp16 plus its fp32 upcast for cross-entropy, ~8.7GB total
#   on top of everything else. Dynamic padding means this only bites when a long row
#   lands in a batch: only 17 of 57,000 rows exceed 1536 tokens and 2 exceed 2048, so
#   the common case fits with room and the tail case is the risk.
#   FALLBACK IF IT OOMs: BS 2 / GA 8. NOT BS 2 alone — the effective batch is
#   BS x GA and must stay 16, because that is what E4 is registered at.
MAX_LEN, EPOCHS, LR, BS, GA = 2560, 1, 2e-4, 4, 4
REGISTERED_EFFECTIVE_BATCH = 16   # PREREGISTRATION 3t, E4 — not merely this file's value
assert BS * GA == REGISTERED_EFFECTIVE_BATCH, (
    f"effective batch is {BS * GA}, but E4 is registered at "
    f"{REGISTERED_EFFECTIVE_BATCH} in PREREGISTRATION 3t. The documented OOM fallback is "
    f"BS 2 / GA 8, which preserves it; changing the product is an amendment.")
EVAL_BS = 8            # halved automatically on OOM; 2560-token prompts are large
MAX_NEW_TOKENS = 16                    # longest label = 5 tokens + eos; 16 is 2.7x margin
EVAL_N = 3000
HOLDOUT_SHA12, TEST3000_SHA12 = "97eebc04d0c7", "e719c1109069"
import transformers as _tf, trl as _trl, peft as _peft
_ACTUAL = {"transformers": _tf.__version__, "trl": _trl.__version__, "peft": _peft.__version__}
# EXACT pins for what must be controlled. transformers is a FLOOR, not an equality:
# trl 1.12.0 requires >=4.56.2 with no upper bound, and Kaggle ships 5.0.0.
_EXACT = {"trl": "1.12.0", "peft": "0.20.0"}
_MIN = {"transformers": (4, 56, 2)}
print(f"versions: transformers {_ACTUAL['transformers']} | trl {_ACTUAL['trl']} | "
      f"peft {_ACTUAL['peft']}")
_bad = [f"{k}: expected {v}, running {_ACTUAL[k]}" for k, v in _EXACT.items()
        if _ACTUAL[k] != v]
for k, floor in _MIN.items():
    got = tuple(int(x) for x in _ACTUAL[k].split(".")[:3] if x.isdigit())
    if got < floor:
        _bad.append(f"{k}: need >= {'.'.join(map(str, floor))}, running {_ACTUAL[k]}")
if _bad:
    raise RuntimeError(
        "VERSION MISMATCH — the pins did not take. " + "; ".join(_bad) +
        ". Did you run CELL 1 and then RESTART THE KERNEL? pip cannot replace a "
        "package the kernel has already imported, and a half-replaced package is how "
        "the 'velora_config' crash happened. PREFLIGHT previously passed while running "
        "versions that were never introspected — this check exists so that cannot recur.")
print("  PREFLIGHT: versions match the pins")

# The gate above reads `<module>.__version__`. THAT IS THE VALUE THAT LIED: it reported
# peft 0.19.1 while 0.20.0's bnb.py executed. `__version__` is a string baked into the
# package at build time, so in a mixed install it describes whichever __init__.py won,
# not the modules actually imported. If the mismatch runs the other way — new metadata
# over stale code, or the reverse — an equality check against it passes on a broken
# environment. The three checks below test the install itself, not its self-report.
# Ported from kaggle_probe_qlora.py step 2 so Tier 1 is not dependent on the probe
# having been run.
import importlib.metadata as _md, pathlib as _pl

# (a) Distribution metadata vs the imported module's own string. These come from
#     different places on disk; a half-replaced install can disagree.
for _k, _mod in (("transformers", _tf), ("trl", _trl), ("peft", _peft)):
    _meta = _md.version(_k)
    if _meta != _mod.__version__:
        raise RuntimeError(
            f"MIXED INSTALL — {_k}: dist-info says {_meta}, imported module says "
            f"{_mod.__version__}. One of pip's writes did not complete, or the kernel "
            f"holds an older module than the metadata on disk. Restart the kernel; if "
            f"it persists, start a fresh session. Module file: {_mod.__file__}")
print(f"  PREFLIGHT: dist-info == __version__ for transformers, trl, peft")

# (b) The crash site itself. peft 0.20.0's tuners/lora/bnb.py references
#     LoraConfig.velora_config; 0.19.1's LoraConfig does not define it. Comparing the
#     SOURCE of the module that crashed against the FIELDS of the class it crashed on
#     detects the mix in either direction, and does not consult any version string.
_bnb_path = _pl.Path(_peft.__file__).parent / "tuners" / "lora" / "bnb.py"
if not _bnb_path.exists():
    raise RuntimeError(f"peft layout unexpected: {_bnb_path} missing — cannot verify "
                       "the mixed-install crash site. Do not proceed blind.")
_uses = "velora_config" in _bnb_path.read_text()
_has = "velora_config" in LoraConfig.__dataclass_fields__
if _uses != _has:
    raise RuntimeError(
        f"MIXED PEFT INSTALL — bnb.py references velora_config: {_uses}, but "
        f"LoraConfig defines it: {_has}. These come from different peft versions. This "
        f"is the exact AttributeError that killed two Kaggle sessions. Re-run CELL 1 "
        f"and RESTART THE KERNEL.")
print(f"  PREFLIGHT: peft crash site consistent (bnb.py uses velora_config={_uses}, "
      f"LoraConfig defines it={_has})")

# (c) Print where peft is actually imported from. Kaggle has more than one site-packages
#     on sys.path; if a stale copy shadows the installed one, this is what shows it.
print(f"  PREFLIGHT: peft.__file__ = {_peft.__file__}")
print("PREFLIGHT — validating every third-party signature before anything expensive")

# 1. Resolve the sequence-length kwarg by introspection, never by assumption.
SEQ_LEN_KW = _resolve(SFTConfig, ["max_length", "max_seq_length"], "sequence length")

# 2. Every other kwarg this cell passes, checked against the installed signature.
# This list must name EVERY kwarg the cell passes to SFTConfig. "bf16" was passed
# without being listed: it would not have failed, because bf16 is a dataclass field, but
# an allowlist that happens to pass is not doing its job — it is the same shape as a
# check that verifies the wrong property. The list is the claim; a kwarg missing from it
# is unchecked whether or not it happens to be valid.
_SFTCONFIG_KWARGS = ["output_dir","seed","num_train_epochs","learning_rate",
    "per_device_train_batch_size","gradient_accumulation_steps","fp16","bf16",
    "logging_steps","save_strategy","save_steps","save_total_limit","report_to",
    "gradient_checkpointing"]
_require(SFTConfig, _SFTCONFIG_KWARGS, "SFTConfig")
_require(SFTTrainer, ["model","train_dataset","peft_config","args"], "SFTTrainer")
_require(LoraConfig, ["r","lora_alpha","lora_dropout","bias","task_type",
    "target_modules"], "LoraConfig")
_require(BitsAndBytesConfig, ["load_in_4bit","bnb_4bit_quant_type",
    "bnb_4bit_compute_dtype","bnb_4bit_use_double_quant"], "BitsAndBytesConfig")
_require(AutoModelForCausalLM.from_pretrained, ["device_map","quantization_config"],
         "AutoModelForCausalLM.from_pretrained")

# 3. Defaults that change SEMANTICS, asserted rather than trusted.
_f = SFTConfig.__dataclass_fields__
assert _f["packing"].default is False, (
    "SFTConfig.packing defaults True in this version — packing concatenates examples "
    "into fixed blocks, so the LABEL would no longer sit at the end of its own prompt.")
assert _f["dataset_text_field"].default == "text", (
    f"dataset_text_field default is {_f['dataset_text_field'].default!r}, not 'text'")
print(f"  PREFLIGHT: packing={_f['packing'].default}, "
      f"dataset_text_field={_f['dataset_text_field'].default!r}, "
      f"{SEQ_LEN_KW} default={_f[SEQ_LEN_KW].default} (we override it)")
# 4. PRECISION AND DEVICE, resolved and cross-checked BEFORE any weights load.
# Three runs were each lost to a different T4-specific defect that only surfaced deep
# into the run. Every one of them was a disagreement between values that are knowable
# in the first ten seconds, so they are resolved and compared here.
_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
_bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
_cfg_dtype = AutoConfig.from_pretrained(MODEL).dtype
print(f"  PREFLIGHT: devices visible={torch.cuda.device_count()} "
      f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
print(f"  PREFLIGHT: gpu={torch.cuda.get_device_name(0) if _cap else 'NONE'} "
      f"sm_{_cap[0]}{_cap[1]}" if _cap else "  PREFLIGHT: NO GPU")
print(f"  PREFLIGHT: model config dtype={_cfg_dtype} -> OVERRIDDEN with "
      f"MODEL_DTYPE={MODEL_DTYPE} | COMPUTE_DTYPE={COMPUTE_DTYPE} | "
      f"fp16={USE_FP16} bf16={USE_BF16} | bf16 supported here={_bf16_ok}")
assert torch.cuda.is_available(), "no GPU — set Accelerator to GPU"
assert torch.cuda.device_count() == 1, (
    f"{torch.cuda.device_count()} GPUs visible. CUDA_VISIBLE_DEVICES must pin exactly "
    f"one BEFORE torch is imported, or Trainer wraps the 4-bit model in DataParallel "
    f"(its guard only tests is_loaded_in_8bit) and doubles the effective batch to 32.")
assert MODEL_DTYPE == COMPUTE_DTYPE, (
    f"MODEL_DTYPE {MODEL_DTYPE} != COMPUTE_DTYPE {COMPUTE_DTYPE}; bnb dequantises into "
    f"the compute dtype and the mismatch shows up as silent numerical drift, not an error")
assert not USE_BF16, "bf16 is selected but T4 is sm_75 and has no bf16 support"
assert USE_FP16 != USE_BF16, "exactly one of fp16/bf16 must be set"
if USE_BF16 and not _bf16_ok:
    raise RuntimeError("bf16 requested on hardware that does not support it")
if MODEL_DTYPE is torch.bfloat16 and not _bf16_ok:
    raise RuntimeError(f"MODEL_DTYPE is bf16 but this GPU (sm_{_cap[0]}{_cap[1]}) "
                       f"cannot do bf16 — this is the dtype='auto' trap")
print("  PREFLIGHT: precision and device agree")
print("PREFLIGHT PASSED\n")

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
# NOTE: this is only the initial value. The eval loop switches to "left", and nothing
# would switch it back — so seed 2's training assert would fire after seed 1's 5-6
# hours. Each seed therefore SETS the side it needs inside the loop rather than
# inheriting it. Do not rely on this line.
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
        bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True)

    if adapter.exists():
        # RESUME GAP FIX: eval can take >1h. If the session died during eval, the
        # adapter already exists and retraining the whole seed would waste hours.
        print(f"\n=== seed {seed}: adapter exists, SKIPPING TRAINING, going to eval ===")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(MODEL, device_map=DEVICE_MAP,
                                                    quantization_config=quant,
                                                    dtype=MODEL_DTYPE)
        model = PeftModel.from_pretrained(base, str(adapter))
    else:
        resume = ck.exists() and any(ck.glob("checkpoint-*"))
        print(f"\n=== seed {seed} ({'RESUMING' if resume else 'fresh'}) ===")
        base = prepare_model_for_kbit_training(
            AutoModelForCausalLM.from_pretrained(MODEL, device_map=DEVICE_MAP,
                                                 quantization_config=quant,
                                                 dtype=MODEL_DTYPE))
        # NOTE: TRL's collator pads via its own trl.trainer.utils.pad(), whose
        # padding_side defaults to "right", and SFTTrainer never reads
        # tok.padding_side — so for TRAINING this is belt-and-braces, not the
        # mechanism. It matters for the EVAL loop below, which calls tok() directly.
        # Kept because a future TRL version could start honouring it.
        # ESTABLISH the invariant for this seed — never inherit it. The previous
        # seed's eval left it as "left"; without this line seed 2 onward would fail
        # the assert below after hours of work, and a re-run would only shift the
        # failure to seed 3.
        tok.padding_side = "right"
        assert tok.padding_side == "right", (
            "training requires RIGHT padding; left padding corrupts the loss on the "
            "first real token of every padded sequence")
        cfg_kw = dict(output_dir=str(ck), seed=seed, num_train_epochs=EPOCHS,
            learning_rate=LR, per_device_train_batch_size=BS,
            gradient_accumulation_steps=GA, logging_steps=100,
            save_strategy="steps", save_steps=500, save_total_limit=2,
            report_to=[], gradient_checkpointing=True,
            fp16=USE_FP16, bf16=USE_BF16)   # both explicit; bf16 default is None, not False
        cfg_kw[SEQ_LEN_KW] = MAX_LEN   # name resolved by PREFLIGHT, not assumed
        trainer = SFTTrainer(model=base, train_dataset=train_txt,
            peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj","k_proj","v_proj","o_proj",
                                "gate_proj","up_proj","down_proj"]),
            args=SFTConfig(**cfg_kw))

        # UNDO TRL's UNCONDITIONAL bf16 CAST. trl 1.12.0 SFTTrainer.__init__, ~line 1154:
        #     if _is_quantized_model:
        #         for param in model.parameters():
        #             if param.requires_grad:
        #                 param.data = param.data.to(torch.bfloat16)
        # It casts every trainable parameter to bf16 whenever the model is 4-bit, with
        # NO GPU capability check and no reference to args.fp16. On Ampere+ that follows
        # the QLoRA paper; on a T4 (sm_75, no bf16) it is fatal: fp16 AMP installs a
        # GradScaler, and unscale_ raises
        #     NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
        #     not implemented for 'BFloat16'
        # after the first backward pass.
        #
        # Passing dtype= to from_pretrained does NOT fix this — measured. The order is:
        # from_pretrained sets the base dtype, prepare_model_for_kbit_training then casts
        # every non-Params4bit parameter to fp32, and TRL's cast runs last and wins. See
        # PREREGISTRATION 3r for the trace. fp32 is the correct target, not fp16: AMP
        # keeps master weights in fp32 and autocasts the matmuls.
        _recast = 0
        for _p in trainer.model.parameters():
            if _p.requires_grad and _p.dtype != torch.float32:
                _p.data = _p.data.to(torch.float32); _recast += 1
        _tdt = {str(_p.dtype) for _p in trainer.model.parameters() if _p.requires_grad}
        print(f"  trainable param dtypes after re-cast: {_tdt} ({_recast} tensors changed)")
        assert _tdt == {"torch.float32"}, (
            f"trainable params are {_tdt}, not fp32. fp16 AMP's GradScaler cannot unscale "
            f"anything but fp32; if this fires, TRL changed the cast at SFTTrainer "
            f"__init__ line ~1154 and the fix above no longer reaches it.")

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

    # switch to LEFT for generation. Set here, per seed; the training branch above
    # sets "right" back for the next seed rather than depending on this being undone.
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
