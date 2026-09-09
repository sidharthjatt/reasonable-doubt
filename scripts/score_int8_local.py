"""Score the downloaded INT8 artefact ON THE MAC — the ISA that actually deploys it.

    python scripts/score_int8_local.py --int8-dir <downloaded int8_ce_1> --seed 1

Kaggle quantises for arm64 but EVALUATES on its own x86 CPU. INT8 kernels are
ISA-specific and their numerics can differ, so the accuracy Kaggle reports for the
INT8 artefact is measured on hardware that never serves it.

**E1's accept rule attaches to the INT8 number measured HERE.** If the two disagree,
the Mac figure is authoritative and the gap is reported, because it is a real property
of the deployed system.
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
from src.eval.metrics import score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--int8-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kaggle-result", type=Path, default=None,
                    help="tier0_<arm>_seed<N>.json from Kaggle, to compare against")
    ap.add_argument("--manifest", default="test_3000")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    # The frontier needs the LOGITS, not just the summary metrics. Saved on request
    # rather than always, so a scoring run does not silently write a second artefact.
    ap.add_argument("--save-logits", type=Path, default=None,
                    help="npz path for the raw logits + row indices (for E6/E5 joins)")
    args = ap.parse_args()

    import onnxruntime as ort
    from transformers import AutoTokenizer

    ds = load_ledgar()
    man = load_manifest(args.manifest, DEFAULT_MANIFEST_DIR)
    verify_manifest(man, ds, DEFAULT_MANIFEST_DIR)
    names = label_names(ds)
    split = get_split(ds, man.split)
    texts = list(split.select(man.indices)["text"])
    gold = [names[int(x)] for x in split.select(man.indices)["label"]]

    tok = AutoTokenizer.from_pretrained(str(args.int8_dir))
    sess = ort.InferenceSession(str(next(args.int8_dir.glob("*.onnx"))),
                                providers=["CPUExecutionProvider"])
    wanted = {i.name for i in sess.get_inputs()}
    print(f"manifest {man.name} sha={man.text_sha256[:16]}… ({len(texts)} rows)")
    print(f"artefact {args.int8_dir}  providers={sess.get_providers()}")

    logits = []
    for i in range(0, len(texts), args.batch_size):
        enc = tok(texts[i:i + args.batch_size], truncation=True,
                  max_length=args.max_length, padding=True, return_tensors="np")
        logits.append(sess.run(None, {k: v.astype(np.int64)
                                      for k, v in enc.items() if k in wanted})[0])
        if i % 500 == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    logits = np.concatenate(logits, 0)
    pred = [names[i] for i in logits.argmax(-1)]

    rep = score(gold, pred, labels=sorted(set(gold)))
    print("\n" + "=" * 62)
    print(f"INT8 on THIS machine (arm64) — {args.manifest}")
    print("=" * 62)
    print(rep.render())

    if args.save_logits:
        args.save_logits.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_logits,
            test_3000_logits=logits,
            test_3000_indices=np.array(man.indices),
            test_3000_labels=np.array([int(x) for x in
                                       split.select(man.indices)["label"]]),
            provenance=np.array(json.dumps({
                "precision": "int8", "isa": "arm64_local",
                "artefact": str(args.int8_dir), "seed": args.seed,
                "manifest": man.name, "manifest_sha256": man.text_sha256})))
        print(f"wrote logits -> {args.save_logits}  [precision=int8, isa=arm64_local]")

    payload = {"seed": args.seed, "artefact": str(args.int8_dir),
               "manifest": man.name, "manifest_sha256": man.text_sha256,
               "isa": "arm64_local", "metrics": rep.as_dict()}

    if args.kaggle_result and args.kaggle_result.exists():
        k = json.loads(args.kaggle_result.read_text())
        kf = k.get("test_3000_int8", {}).get("macro_f1")
        if kf is not None:
            d = rep.macro_f1 - kf
            payload["kaggle_x86_macro_f1"] = kf
            payload["local_minus_kaggle"] = d
            print(f"\n  Kaggle (x86) INT8 macro-F1 : {kf:.4f}")
            print(f"  local  (arm64) INT8 macro-F1: {rep.macro_f1:.4f}")
            print(f"  difference                  : {d:+.4f}")
            print("  -> E1 uses the LOCAL figure; it is the deployed system."
                  + ("" if abs(d) < 1e-9 else
                     "\n  -> non-zero: INT8 kernels differ across ISAs. Report the gap."))

    out = args.out or Path(f"results/int8_local_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
