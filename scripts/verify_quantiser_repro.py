"""Is an INT8 artefact byte-reproducible under THIS machine's quantiser?

    python scripts/verify_quantiser_repro.py \
        --onnx-fp32 <onnx_ce10ep_1> --reference-int8 <int8_ce10ep_1>

WHY. E1b commit 1 installed onnxruntime 1.29.0 and commit 2 installed 1.30.0 — the
Kaggle install cell pins only `transformers`, so ORT, onnx and optimum all float.
Training is torch-only, so the drift touches ONLY commit 2's ONNX-export + INT8-quantise
tail. That tail is exactly what the gate reads: the gate compares E1b INT8 against the
HOST BASELINE INT8, and the host baseline was quantised under ORT 1.29.0
(results/tier0_ce_hostB_seed1.json -> train_env.onnxruntime). A quantiser version change
is therefore a SECOND VARIABLE inside a difference that is supposed to isolate epochs.

WHAT THIS ANSWERS.
  1. Do the reference artefact's quantised tensors match what this machine's quantiser
     produces from the same FP32 ONNX? Byte-for-byte, per initializer.
  2. Can this machine's onnxruntime LOAD the reference artefact at all?

If (1) is identical there is no confound and the gate proceeds unchanged. If it is not,
the locally re-quantised bytes are written next to the reference, the gate reads THOSE
(matching the host baseline's 1.29.0), and the difference is disclosed rather than
absorbed.

WHAT IT DOES NOT DO. It does not attribute a mismatch to onnxruntime specifically.
`ORTQuantizer` is optimum's front end over `onnxruntime.quantization`, and optimum is
unpinned too — and its version was never recorded, because `train_env()` reads
`optimum.__version__`, which does not exist (it lives at `optimum.version.__version__`).
So a mismatch means "the toolchain moved", and isolating which half moved needs a second
run holding one of them fixed. That is stated rather than guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tool_versions() -> dict:
    import onnx
    import onnxruntime as ort

    try:
        from optimum.version import __version__ as optimum_version
    except Exception:
        optimum_version = None
    return {"onnxruntime": ort.__version__, "onnx": onnx.__version__,
            "optimum": optimum_version}


def initializer_digests(path: Path) -> dict[str, str]:
    """sha256 per initializer tensor, keyed by name.

    Compared per-tensor rather than over the whole file because a whole-file hash
    conflates a real weight difference with a metadata or ordering difference, and those
    have completely different consequences for the gate.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    return {init.name: hashlib.sha256(init.SerializeToString()).hexdigest()
            for init in model.graph.initializer}


def quantise(onnx_dir: Path, out_dir: Path) -> Path:
    """Re-quantise with the SAME config the notebook uses. Any deviation here would
    manufacture a difference and blame it on the version."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    ORTQuantizer.from_pretrained(str(onnx_dir)).quantize(
        save_dir=str(out_dir),
        quantization_config=AutoQuantizationConfig.arm64(is_static=False,
                                                         per_channel=True))
    return next(out_dir.glob("*.onnx"))


def can_load(path: Path) -> tuple[bool, str | None]:
    import onnxruntime as ort

    try:
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return True, None
    except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx-fp32", type=Path, required=True,
                    help="the FP32 ONNX export the reference was quantised from")
    ap.add_argument("--reference-int8", type=Path, required=True,
                    help="the downloaded INT8 artefact to check")
    ap.add_argument("--work", type=Path, default=None,
                    help="where to write the local re-quantisation")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    versions = tool_versions()
    ref_onnx = next(args.reference_int8.glob("*.onnx"))

    print("=" * 70)
    print("QUANTISER REPRODUCIBILITY CHECK")
    print("=" * 70)
    print(f"  this machine : ort {versions['onnxruntime']}  onnx {versions['onnx']}  "
          f"optimum {versions['optimum']}")
    print(f"  fp32 onnx    : {args.onnx_fp32}")
    print(f"  reference    : {ref_onnx}")

    loadable, load_err = can_load(ref_onnx)
    print(f"\n  reference loads under ort {versions['onnxruntime']}: {loadable}")
    if not loadable:
        print(f"    {load_err}")

    work = args.work or (args.reference_int8.parent /
                         f"{args.reference_int8.name}__requantised_local")
    work.mkdir(parents=True, exist_ok=True)
    print(f"\n  re-quantising locally -> {work}")
    local_onnx = quantise(args.onnx_fp32, work)

    ref_d, loc_d = initializer_digests(ref_onnx), initializer_digests(local_onnx)
    only_ref, only_loc = sorted(set(ref_d) - set(loc_d)), sorted(set(loc_d) - set(ref_d))
    shared = sorted(set(ref_d) & set(loc_d))
    differing = [n for n in shared if ref_d[n] != loc_d[n]]

    identical = not (only_ref or only_loc or differing)
    print("\n" + "-" * 70)
    print(f"  initializers: {len(ref_d)} reference / {len(loc_d)} local, "
          f"{len(shared)} shared")
    print(f"  name-only-in-reference : {len(only_ref)}")
    print(f"  name-only-in-local     : {len(only_loc)}")
    print(f"  shared but DIFFERING   : {len(differing)}")
    print("-" * 70)
    verdict = ("IDENTICAL — no quantiser confound" if identical
               else "NOT IDENTICAL — the toolchain moved")
    print(f"  VERDICT: {verdict}")
    if not identical:
        for n in differing[:10]:
            print(f"    differs: {n}")
        if len(differing) > 10:
            print(f"    … and {len(differing) - 10} more")
        print("\n  The locally re-quantised model is at:")
        print(f"    {local_onnx}")
        print("  Score the GATE against those bytes (they match the host baseline's "
              "toolchain) and disclose the difference.")

    payload = {"versions_this_machine": versions,
               "onnx_fp32": str(args.onnx_fp32),
               "reference_int8": str(ref_onnx),
               "reference_loads_locally": loadable,
               "reference_load_error": load_err,
               "n_initializers_reference": len(ref_d),
               "n_initializers_local": len(loc_d),
               "n_only_in_reference": len(only_ref),
               "n_only_in_local": len(only_loc),
               "n_shared": len(shared),
               "n_differing": len(differing),
               "differing_names": differing,
               "identical": identical,
               "local_requantised": str(local_onnx),
               "attribution_note": (
                   "A mismatch means the TOOLCHAIN moved, not onnxruntime specifically: "
                   "ORTQuantizer is optimum's front end over onnxruntime.quantization "
                   "and optimum is unpinned and unrecorded. Isolating which half moved "
                   "needs a second run holding one fixed.")}
    out = args.out or Path("results/quantiser_repro.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0 if identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
