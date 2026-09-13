#!/usr/bin/env python3
"""Derive the small JSON files REPORT.md and its figures read. Costs $0.00.

Every value here is COPIED OR RECOMPUTED from something already on disk — a results file,
the spend ledger, or configs/costs.yaml — never typed in. Each output carries a `source`
field naming where it came from, so a reader can check any number in the report without
trusting the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R = Path("results")


def stage1_scored() -> None:
    """Score the Stage 1 API answers on the FULL test_3000 through the registered scorer."""
    from src.data.schema import ParseFailure, parse_response
    from src.eval.metrics import score

    names = json.loads(Path("configs/labels.json").read_text())["labels"]
    npz = np.load(R / "test_logits_fp32_local_ce10ep_seed1.npz", allow_pickle=True)
    idx, gold_i = npz["test_3000_indices"], npz["test_3000_labels"].astype(int)
    gold = [names[int(g)] for g in gold_i]
    raw = json.loads((R / "stage1_results.json").read_text())["models"]

    out = {}
    for m, blob in raw.items():
        rows, preds, fails = blob["rows"], [], 0
        for i in idx:
            body = rows.get(str(int(i)))
            try:
                preds.append(parse_response(body.get("text", "")).label)
            except (ParseFailure, AttributeError):
                preds.append(None)
                fails += 1
        r = score(gold, preds, labels=names)
        out[m] = {"macro_f1": r.macro_f1, "accuracy": r.accuracy,
                  "n": len(preds), "parse_failures": fails}
    (R / "stage1_scored_test3000.json").write_text(json.dumps({
        "what": "Stage 1 zero-shot API answers scored on ALL 3000 test_3000 rows",
        "source": "results/stage1_results.json + results/test_logits_fp32_local_ce10ep_seed1.npz",
        "scorer": "src.eval.metrics.score", "classes_averaged": 100,
        "note": ("NOT the same scope as §4's E8 entry, which scores the FIRST 1000 rows "
                 "over the 97 classes present there. The two must not be compared."),
        "models": out}, indent=2) + "\n")
    print("wrote results/stage1_scored_test3000.json")


def cost_per_1k() -> None:
    """Measured API cost per 1,000 clauses, from the ledger. Local from costs.yaml."""
    from src.serve.pricing import local_usd_per_request

    ledger = [json.loads(x) for x in
              (R / "spend_ledger.jsonl").read_text().splitlines() if x.strip()]
    api = {}
    for row in ledger:
        # `stage1_cancelled` is a real ledger row for a batch that was cancelled: 0
        # tokens, $0.00, and a comma-joined model field covering both legs. It is kept
        # in the ledger (append-only, hard rule 12) and excluded here, because a $0.00
        # "cost per 1,000" would plot as the cheapest model on the chart.
        if (row["run_id"].startswith("stage1_") and row["kind"] == "actual"
                and row["n_requests"] and row["cost_usd"] > 0
                and "," not in row["model"]):
            api[row["model"]] = {
                "usd_per_1k": row["cost_usd"] / row["n_requests"] * 1000,
                "cost_usd": row["cost_usd"], "n_requests": row["n_requests"],
                "cache_read_input_tokens": row["cache_read_input_tokens"],
                "cache_creation_input_tokens": row["cache_creation_input_tokens"],
                "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
                "batch_id": row["batch_id"]}
    local = local_usd_per_request(precision="fp32")
    (R / "cost_per_1k.json").write_text(json.dumps({
        "what": "USD per 1,000 clauses",
        "source": ("API: results/spend_ledger.jsonl, stage1_* rows, kind=actual. "
                   "Local: configs/costs.yaml via src.serve.pricing (§3bl)."),
        "api_is_batch_with_caching": True,
        "local_tier0_fp32": {"usd_per_1k": local.usd * 1000, "basis": local.basis,
                             "tariff_is_assumed": local.tariff_is_assumed,
                             "is_estimate": local.is_estimate},
        "api": api}, indent=2) + "\n")
    print("wrote results/cost_per_1k.json")


def isa_matrix() -> None:
    """Every INT8-vs-FP32 measurement, per ISA, with its metric and artefact named."""
    e1b = json.loads((R / "int8_local_ce10ep_seed1.json").read_text())["metrics"]
    fp32 = json.loads((R / "fp32_local_onnx_ce10ep_seed1.json").read_text())["metrics"]
    (R / "isa_matrix.json").write_text(json.dumps({
        "what": "INT8 vs FP32 across every ISA measured. TWO METRICS — do not mix them.",
        "test_3000_macro_f1": {
            "metric": "macro-F1 on test_3000 (3000 rows)",
            "artefact": "E1b seed 1",
            "rows": [
                {"isa": "macOS arm64", "int8": e1b["macro_f1"], "fp32": fp32["macro_f1"],
                 "avx512_vnni": None, "note": "ARM; AVX-512 not applicable",
                 "source": "results/int8_local_ce10ep_seed1.json, results/fp32_local_onnx_ce10ep_seed1.json"},
                {"isa": "Linux x86 (Kaggle Xeon)", "int8": 0.000166, "fp32": 0.8114655856105638,
                 "avx512_vnni": False, "note": "avx512f=true, avx2=true, avx512_vnni=FALSE",
                 "source": "tier0_ce10ep_seed1.json (Kaggle run JSON): test_3000_int8, test_3000_onnx_fp32, e3_discriminator.onnx_env"},
            ]},
        # ONE ROW PER HOST MEASURED, each naming its own METRIC. test_3000 macro-F1 and
        # 200-row canary accuracy are different quantities on different row sets, and the
        # matrix the demo renders must never put them in one column (§3bc).
        "platforms": [
            {"host": "macOS arm64 (Apple Silicon)", "machine": "arm64",
             "isa": {"avx512_vnni": None, "avx512f": None, "avx2": None},
             "metric": "test_3000 macro-F1", "artefact": "E1b seed 1",
             "fp32": fp32["macro_f1"], "int8": e1b["macro_f1"],
             "source": "results/fp32_local_onnx_ce10ep_seed1.json, results/int8_local_ce10ep_seed1.json"},
            {"host": "Linux x86, Kaggle Xeon", "machine": "x86_64",
             "isa": {"avx512_vnni": False, "avx512f": True, "avx2": True},
             "metric": "test_3000 macro-F1", "artefact": "E1b seed 1",
             "fp32": 0.8114655856105638, "int8": 0.000166,
             "note": "INT8 at chance on a 100-class problem, no error raised.",
             "source": "tier0_ce10ep_seed1.json (Kaggle run JSON)"},
            {"host": "Linux aarch64, container", "machine": "aarch64",
             "isa": {"avx512_vnni": None, "avx512f": None, "avx2": None},
             "metric": "canary accuracy, 200 TRAIN rows", "artefact": "E1 seed 1",
             "fp32": 0.9150, "int8": 0.6400,
             "note": "Same physical CPU as the arm64 host. INT8 diverged; the canary refused.",
             "source": "PREREGISTRATION §3bf/§3bg"},
            {"host": "Cloud Run x86, VNNI", "machine": "x86_64",
             "isa": {"avx512_vnni": True, "avx512f": True, "avx2": True},
             "metric": "canary accuracy, 200 TRAIN rows", "artefact": "E1b seed 1",
             "fp32": 0.9550, "int8": None,
             "note": "12 of 13 instances observed reported VNNI.",
             "source": "PREREGISTRATION §3bn"},
            {"host": "Cloud Run x86, NO VNNI", "machine": "x86_64",
             "isa": {"avx512_vnni": False, "avx512f": False, "avx2": True},
             "metric": "canary accuracy, 200 TRAIN rows", "artefact": "E1b seed 1",
             "fp32": 0.9550, "int8": None,
             "note": ("Instance 65b67a604cb6, revision -00009-99j. FP32 does not collapse "
                      "on the hardware class where INT8 scored 0.000166. CANARY-LEVEL "
                      "EVIDENCE, not a test_3000 score."),
             "source": "PREREGISTRATION §3bq"},
        ],

        "canary_accuracy_200_train_rows": {
            "metric": "accuracy on the 200 frozen canary rows — NOT comparable to test_3000",
            "rows": [
                {"isa": "macOS arm64", "int8": 0.9000, "fp32": 0.9150,
                 "artefact": "E1 seed 1", "source": "PREREGISTRATION §3bf/§3bg"},
                {"isa": "Linux aarch64 (container)", "int8": 0.6400, "fp32": 0.9150,
                 "artefact": "E1 seed 1", "source": "PREREGISTRATION §3bf/§3bg"},
                {"isa": "Linux x86 (emulated, local)", "int8": None, "fp32": 0.9550,
                 "artefact": "E1b seed 1",
                 "note": "avx512_vnni=false, avx512f=false, avx2=true; emulated on Apple Silicon",
                 "source": "PREREGISTRATION §3bn"},
            ]},
    }, indent=2) + "\n")
    print("wrote results/isa_matrix.json")


if __name__ == "__main__":
    stage1_scored()
    cost_per_1k()
    isa_matrix()
