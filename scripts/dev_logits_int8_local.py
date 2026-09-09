"""Produce dev_2000 INT8 logits ON THE MAC (arm64), for E5 on the deployed precision.

    python scripts/dev_logits_int8_local.py --int8-dir models/int8_ce_1 --seed 1

**This is inference, not a re-score, and the distinction matters.** The
`dev_logits_int8_ce_seed*.npz` produced by the Kaggle dev run were computed by the same
non-VNNI x86 kernel that scored test_3000 at chance (§3ah): ONNX-FP32 vs INT8 argmax
agreement was 1.2-2.0% there. Those arrays are noise, and no re-scoring turns noise into
signal — the forward pass has to be run again on the ISA that deploys.

Writes the npz in `src/router/load_logits.py`'s `dev` convention so `scripts/e5_sweep.py`
consumes it unchanged, PLUS a `provenance` array recording precision and ISA. E5's guard
reads that provenance rather than guessing from the filename: after this run, "is this
INT8?" stops being the question and "was it measured on a kernel that works?" starts.

dev_2000 is calibration-ONLY (hard rule 1). This script therefore writes LOGITS and
deliberately prints NO macro-F1 for the split: computing dev accuracy here would make a
reporting number out of a threshold split. The router consumes the logits; nothing else
should.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data.loading import get_split, label_names, load_ledgar  # noqa: E402
from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest, verify_manifest  # noqa: E402
from src.router.calibrate import assert_threshold_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--int8-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--manifest", default="dev_2000")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"results/dev_logits_int8_local_ce_seed{args.seed}.npz")

    # The split guard runs on the NAME before any data is touched, so this script cannot
    # be pointed at test_3000 or train_holdout_3000 to manufacture a reporting number.
    assert_threshold_split(args.manifest)

    import onnxruntime as ort
    from transformers import AutoTokenizer

    ds = load_ledgar()
    man = load_manifest(args.manifest, DEFAULT_MANIFEST_DIR)
    verify_manifest(man, ds, DEFAULT_MANIFEST_DIR)
    names = label_names(ds)
    split = get_split(ds, man.split)
    texts = list(split.select(man.indices)["text"])
    labels = np.array([int(x) for x in split.select(man.indices)["label"]])
    absent = np.array(sorted(set(range(len(names))) - set(labels.tolist())), dtype=int)

    tok = AutoTokenizer.from_pretrained(str(args.int8_dir))
    sess = ort.InferenceSession(str(next(args.int8_dir.glob("*.onnx"))),
                                providers=["CPUExecutionProvider"])
    wanted = {i.name for i in sess.get_inputs()}
    print(f"manifest {man.name} sha={man.text_sha256[:16]}… ({len(texts)} rows)")
    print(f"artefact {args.int8_dir}  providers={sess.get_providers()}")

    chunks = []
    for i in range(0, len(texts), args.batch_size):
        enc = tok(texts[i:i + args.batch_size], truncation=True,
                  max_length=args.max_length, padding=True, return_tensors="np")
        chunks.append(sess.run(None, {k: v.astype(np.int64)
                                      for k, v in enc.items() if k in wanted})[0])
        if i % 500 == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    logits = np.concatenate(chunks, 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        dev_logits=logits, dev_labels=labels,
        dev_2000_indices=np.array(man.indices),
        dev_absent_classes=absent,
        # Provenance travels IN the artefact. A filename can be copied or renamed; this
        # cannot be separated from the numbers it describes.
        provenance=np.array(json.dumps({
            "precision": "int8", "isa": "arm64_local",
            "artefact": str(args.int8_dir), "seed": args.seed,
            "manifest": man.name, "manifest_sha256": man.text_sha256,
            "max_length": args.max_length, "batch_size": args.batch_size,
        })),
    )
    print(f"\nwrote {out}  [precision=int8, isa=arm64_local, {len(logits)} rows]")
    print("NO macro-F1 printed: dev_2000 is calibration-only (hard rule 1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
