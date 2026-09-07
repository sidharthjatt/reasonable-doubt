"""Tier 0 — DeBERTa-v3-base LEDGAR classifier. Standalone Kaggle training script.

    python tier0_encoder.py --loss ce --seed 1 --max-length 512

Runs as a self-contained Kaggle notebook/script: it imports nothing from this repo
(Kaggle will not have it) and reconstructs the frozen manifests deterministically from
the dataset, then asserts their sha256 against values recorded here. If a hash does not
match, it RAISES — a manifest that has drifted means the rows are not the rows every
other run used.

SPLIT DISCIPLINE, enforced in code (see `assert_selection_split` below)
----------------------------------------------------------------------
  train (60k, minus holdout) : fitting weights
  train_holdout_3000         : SELECTION — epoch, hyperparameters, loss arm, signal
  dev_2000                   : router thresholds ONLY (hard rule 1) — untouched here
  test_3000                  : touched once, for the reported number — never here

Anything that picks calls `assert_selection_split`, which raises on dev or test. That
is why selection cannot silently drift onto the reporting split.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
* It does not choose the loss. Class weighting is a REGISTERED ARM, not a config
  convenience: at 137x imbalance it is the largest single lever on macro-F1, so
  picking it after seeing results is exactly what hard rule 6 forbids. Each `--loss`
  value is a separate preregistered arm with its own accept rule.
* It does not calibrate a router threshold. That is dev-only and lives elsewhere.
* It does not fit temperature for ROUTING. Temperature scaling is a monotone rescale
  of the margin — threshold t on margin/T routes identically to t*T on margin — so it
  is provably a no-op for a margin router, and is fitted here for REPORTING only, so
  Tier 0 confidence can be compared against LLM verbalized confidence on an
  interpretable scale. (The no-op guarantee does NOT hold for max-softmax, whose
  ranking can reorder under scaling.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Split guard — duplicated from src/train/splits.py so the script stays standalone.
# Kept byte-identical in intent; see that module for the rationale.
# ---------------------------------------------------------------------------
SELECTION_SPLITS = frozenset({"train_holdout_3000", "train"})
THRESHOLD_SPLITS = frozenset({"dev_2000", "validation", "dev"})
REPORTING_SPLITS = frozenset({"test_3000", "test_stratified_764", "test"})


class ForbiddenSplitError(RuntimeError):
    """A split was used for a purpose it is not permitted to serve."""


def assert_selection_split(name: str) -> str:
    """Permit only selection splits. Raises on dev or test."""
    if name in REPORTING_SPLITS:
        raise ForbiddenSplitError(
            f"REFUSING: {name!r} is a REPORTING split and must never be used to select "
            f"anything. Selection runs on {sorted(SELECTION_SPLITS)}."
        )
    if name in THRESHOLD_SPLITS:
        raise ForbiddenSplitError(
            f"REFUSING: {name!r} is reserved for router THRESHOLD calibration only "
            f"(hard rule 1). Use {sorted(SELECTION_SPLITS)} to select."
        )
    if name not in SELECTION_SPLITS:
        raise ForbiddenSplitError(f"unknown selection split {name!r}")
    return name


# ---------------------------------------------------------------------------
# Frozen manifest reconstruction. Same algorithm and seeds as src/data/manifest.py.
# ---------------------------------------------------------------------------
DATASET, CONFIG = "coastalcph/lex_glue", "ledgar"
EXPECTED_SPLIT_SIZES = {"train": 60_000, "validation": 10_000, "test": 10_000}

# sha256 of the concatenated texts, from the committed manifests. A mismatch means
# the dataset or the sampling changed and this run is not comparable to any other.
EXPECTED_SHA = {
    "train_holdout_3000": "97eebc04d0c7",  # 12-char prefix; full value checked below
}
MANIFEST_SPEC = {"train_holdout_3000": {"split": "train", "n": 3000, "seed": 20260907}}


def hash_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        b = t.encode("utf-8")
        h.update(f"{len(b)}\n".encode("ascii"))
        h.update(b)
    return h.hexdigest()


def stratified_indices(labels, n, seed):
    """Proportional allocation, largest remainder, seeded draw within class.
    Ascending output preserves the split's chronological row order."""
    import numpy as np

    total = len(labels)
    by_class = {}
    for i, lab in enumerate(labels):
        by_class.setdefault(int(lab), []).append(i)
    quotas = {c: len(r) * n / total for c, r in by_class.items()}
    alloc = {c: int(q) for c, q in quotas.items()}
    order = sorted(quotas, key=lambda c: (-(quotas[c] - alloc[c]), c))
    for c in order[: n - sum(alloc.values())]:
        alloc[c] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for c in sorted(by_class):
        if alloc[c]:
            rows = by_class[c]
            picked.extend(rows[i] for i in rng.choice(len(rows), alloc[c], replace=False))
    picked.sort()
    return picked


def load_data(max_length: int, holdout_name: str):
    """Load LEDGAR, assert shape, rebuild the holdout, verify its hash."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, CONFIG)
    for name, expected in EXPECTED_SPLIT_SIZES.items():
        if len(ds[name]) != expected:
            raise AssertionError(
                f"{name}: {len(ds[name])} rows, expected {expected}. The chronological "
                "60k/10k/10k split is load-bearing; do NOT proceed."
            )
    names = list(ds["train"].features["label"].names)
    if len(names) != 100:
        raise AssertionError(f"expected 100 labels, got {len(names)}")

    spec = MANIFEST_SPEC[holdout_name]
    idx = stratified_indices(list(ds[spec["split"]]["label"]), spec["n"], spec["seed"])
    sha = hash_texts(list(ds[spec["split"]].select(idx)["text"]))
    if not sha.startswith(EXPECTED_SHA[holdout_name]):
        raise AssertionError(
            f"{holdout_name}: rebuilt sha256 {sha[:12]}… != committed "
            f"{EXPECTED_SHA[holdout_name]}…. The rows this run would select on are not "
            "the rows every other run used. Refusing to proceed."
        )
    print(f"  {holdout_name} verified: {sha[:16]}… ({len(idx)} rows)")
    return ds, names, idx


# ---------------------------------------------------------------------------
# Loss arms. Each is a REGISTERED ARM, not a tuning knob.
# ---------------------------------------------------------------------------
LOSS_ARMS = {
    "ce": "unweighted cross-entropy — the preregistered BASELINE",
    "sqrt_inv_freq": "weights ∝ 1/sqrt(freq) — the moderate middle option",
    "effective_number": "Cui et al. effective-number weighting, beta configurable",
    "inv_freq": "weights ∝ 1/freq — aggressive at 137x imbalance; may destabilise",
}


def class_weights(counts, arm: str, num_labels: int, beta: float):
    """Per-class loss weights, normalised to mean 1 so the effective LR is comparable."""
    import numpy as np

    freq = np.array([max(1, counts.get(c, 0)) for c in range(num_labels)], dtype=np.float64)
    if arm == "ce":
        return None
    if arm == "inv_freq":
        w = 1.0 / freq
    elif arm == "sqrt_inv_freq":
        w = 1.0 / np.sqrt(freq)
    elif arm == "effective_number":
        w = (1.0 - beta) / (1.0 - np.power(beta, freq))
    else:
        raise ValueError(f"unknown loss arm {arm!r}; have {sorted(LOSS_ARMS)}")
    return w / w.mean()


# ---------------------------------------------------------------------------
# Routing signals. All three come from the SAME logits — one model, no extra compute.
# ---------------------------------------------------------------------------
def routing_signals(logits):
    """max_softmax, entropy and margin for every row.

    Registered together rather than chosen, so the comparison is preregistered and not
    discovered. Margin is the only one invariant to temperature scaling.
    """
    import numpy as np

    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    srt = np.sort(logits, axis=1)
    return {
        "max_softmax": p.max(axis=1),
        "entropy": -(p * np.log(p + 1e-12)).sum(axis=1),
        "margin": srt[:, -1] - srt[:, -2],
    }


def fit_temperature(logits, labels):
    """Temperature for REPORTING ONLY — never in the routing path.

    Exists so Tier 0's confidence can be compared against LLM verbalized confidence on
    an interpretable scale. For a margin router this is provably a no-op, so applying
    it to routing would be meaningless rather than merely harmless.
    """
    import numpy as np
    from scipy.optimize import minimize_scalar

    def nll(logT):
        T = np.exp(logT)
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        return float(-np.mean(z[np.arange(len(labels)), labels]
                              - np.log(np.exp(z).sum(axis=1))))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--loss", choices=sorted(LOSS_ARMS), default="ce",
                    help="REGISTERED ARM, not a tuning knob — see LOSS_ARMS")
    ap.add_argument("--effective-number-beta", type=float, default=0.999)
    ap.add_argument("--max-length", type=int, default=512,
                    help="512 default; 256 is the registered latency ablation")
    ap.add_argument("--seed", type=int, required=True,
                    help="required: results are mean±std over >=3 seeds (hard rule 2)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--selection-split", default="train_holdout_3000",
                    help="split used to PICK the epoch. Guarded: dev/test are refused.")
    ap.add_argument("--out", type=Path, default=Path("tier0_out"))
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    # The guard runs BEFORE anything expensive, so a forbidden split fails in seconds.
    assert_selection_split(args.selection_split)

    import numpy as np
    import torch
    from torch import nn
    from sklearn.metrics import f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("=" * 72)
    print(f"TIER 0 — {args.model}")
    print(f"  loss arm        : {args.loss}  ({LOSS_ARMS[args.loss]})")
    print(f"  max_length      : {args.max_length}")
    print(f"  seed            : {args.seed}")
    print(f"  selection split : {args.selection_split}  (dev/test refused by guard)")
    print("=" * 72)

    ds, names, holdout_idx = load_data(args.max_length, args.selection_split)

    # Fit on train MINUS the selection holdout, so selection is never on fitted rows.
    holdout = set(holdout_idx)
    fit_idx = [i for i in range(len(ds["train"])) if i not in holdout]
    print(f"  fitting on {len(fit_idx):,} train rows; selecting on {len(holdout_idx):,}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def prep(split, idx):
        sub = ds[split].select(idx)
        enc = tok(list(sub["text"]), truncation=True, max_length=args.max_length,
                  padding=False)
        enc["labels"] = [int(x) for x in sub["label"]]
        return [dict(zip(enc, v)) for v in zip(*enc.values())]

    train_ds = prep("train", fit_idx)
    sel_ds = prep("train", holdout_idx)

    counts = Counter(int(ds["train"][i]["label"]) for i in fit_idx)
    weights = class_weights(counts, args.loss, len(names), args.effective_number_beta)
    if weights is not None:
        print(f"  class weights: min {weights.min():.3f} max {weights.max():.3f} "
              f"(ratio {weights.max() / weights.min():.1f}x)")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)},
    )

    wt = None if weights is None else torch.tensor(weights, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = nn.CrossEntropyLoss(
                weight=None if wt is None else wt.to(out.logits.device)
            )(out.logits, labels)
            return (loss, out) if return_outputs else loss

    def metrics(pred):
        y = pred.label_ids
        p = pred.predictions.argmax(-1)
        return {"macro_f1": f1_score(y, p, average="macro", zero_division=0),
                "accuracy": (p == y).mean()}

    from transformers import DataCollatorWithPadding

    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "hf"), seed=args.seed,
            num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            eval_strategy="epoch", save_strategy="epoch",
            # Epoch selection uses macro-F1 on the SELECTION split only.
            load_best_model_at_end=True, metric_for_best_model="macro_f1",
            greater_is_better=True, save_total_limit=1,
            fp16=torch.cuda.is_available(), report_to=[], logging_steps=200,
        ),
        train_dataset=train_ds, eval_dataset=sel_ds,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
    )
    trainer.train()

    sel = trainer.predict(sel_ds)
    logits, y = sel.predictions, sel.label_ids
    sig = routing_signals(logits)
    temp = fit_temperature(logits, y)

    result = {
        "model": args.model, "loss_arm": args.loss, "max_length": args.max_length,
        "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "selection_split": args.selection_split,
        "selection_macro_f1": float(f1_score(y, logits.argmax(-1), average="macro",
                                             zero_division=0)),
        "selection_accuracy": float((logits.argmax(-1) == y).mean()),
        "reporting_temperature": temp,
        "temperature_note": "REPORTING ONLY — never applied in the routing path",
        "routing_signal_summary": {k: {"mean": float(v.mean()), "std": float(v.std())}
                                   for k, v in sig.items()},
        "precision": "fp32",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"selection_logits_seed{args.seed}.npz", logits=logits, labels=y)
    print(f"\n  selection macro-F1 : {result['selection_macro_f1']:.4f}   <-- PRIMARY")
    print(f"  selection accuracy : {result['selection_accuracy']:.4f}")
    print(f"  reporting temperature (NOT used for routing): {temp:.3f}")

    trainer.save_model(str(args.out / "fp32")); tok.save_pretrained(str(args.out / "fp32"))

    if not args.skip_onnx:
        # INT8 is what DEPLOYS. Reporting an FP32 number as the system's accuracy would
        # describe something that does not exist, so both are measured and the delta
        # is stated. Registered comparison, not an afterthought.
        print("\n  exporting ONNX + INT8 …")
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            ort = ORTModelForSequenceClassification.from_pretrained(
                str(args.out / "fp32"), export=True)
            ort.save_pretrained(str(args.out / "onnx"))
            q = ORTQuantizer.from_pretrained(str(args.out / "onnx"))
            q.quantize(save_dir=str(args.out / "int8"),
                       quantization_config=AutoQuantizationConfig.avx512_vnni(
                           is_static=False, per_channel=True))
            result["onnx_int8_path"] = str(args.out / "int8")
            print("  INT8 written. Evaluate it with the same selection split and record"
                  " the FP32->INT8 macro-F1 delta before any reported number.")
        except Exception as exc:
            # Hard rule 11: do NOT fall back to reporting the FP32 number as if it
            # were the deployed system. Fail, and say what is missing.
            raise RuntimeError(
                f"ONNX/INT8 export failed: {type(exc).__name__}: {exc}. INT8 is the "
                "deployed precision; an FP32-only result must not be reported as the "
                "system's accuracy. Fix the export or pass --skip-onnx and mark the "
                "run explicitly FP32-only."
            ) from exc

    (args.out / f"result_seed{args.seed}_{args.loss}.json").write_text(
        json.dumps(result, indent=2))
    print(f"\n  wrote {args.out}/result_seed{args.seed}_{args.loss}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
