"""Export torch FP32 weights to ONNX FP32 — the artefact the service actually loads.

    python scripts/export_fp32_onnx.py --fp32-dir ~/Downloads/tier0_e1b/fp32_ce10ep_1 \
                                       --out models/onnx_ce10ep_1_fp32

§3bg served FP32 for the first time but exported `models/onnx_ce_1_fp32` by hand, so the
one step between the trained weights and the served bytes had no script. This is that
step, and it delegates to `verify_quantiser_repro.export_fp32_onnx` — the SAME function
§3be's byte-for-byte control validated — rather than re-implementing the call, so the
served export path and the validated export path cannot drift apart.

It REFUSES to overwrite an existing export. A silent re-export would replace bytes that a
calibrated threshold and a measured canary already describe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Loaded by path, not as a package: scripts/ is deliberately NOT a package (adding an
# __init__.py there changes test collection), and the point is to reuse §3be's validated
# function rather than copy it.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_vqr", Path(__file__).resolve().parent / "verify_quantiser_repro.py")
_vqr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_vqr)
export_fp32_onnx = _vqr.export_fp32_onnx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32-dir", type=Path, required=True,
                    help="directory of trained torch FP32 weights")
    ap.add_argument("--out", type=Path, required=True,
                    help="destination ONNX directory, e.g. models/onnx_ce10ep_1_fp32")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.glob("*.onnx")) and not args.force:
        print(f"REFUSING: {args.out} already holds an ONNX export.\n"
              f"Re-exporting would replace bytes that a calibrated router threshold and "
              f"a measured canary already describe. Pass --force only if you intend to "
              f"invalidate both.")
        return 1

    if not args.fp32_dir.exists():
        raise SystemExit(f"no such weights directory: {args.fp32_dir}")

    export_fp32_onnx(args.fp32_dir.expanduser(), args.out)
    onnx = sorted(args.out.glob("*.onnx"))
    if not onnx:
        raise SystemExit(f"export produced no .onnx in {args.out}")
    for f in onnx:
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
