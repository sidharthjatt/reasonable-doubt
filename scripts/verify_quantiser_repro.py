"""Is an INT8 artefact byte-reproducible under a DIFFERENT quantiser version?

TWO PHASES, AND THE FIRST IS A POSITIVE CONTROL. Run them in this order:

    # PHASE 1 — CONTROL. Env pinned to the SAME toolchain that made the reference.
    python scripts/verify_quantiser_repro.py --role control \
        --onnx-fp32 onnx_ce10ep_1 --reference-int8 int8_ce10ep_1 --reference-ort 1.30.0

    # PHASE 2 — CANDIDATE. Env pinned to the HOST BASELINE's toolchain.
    python scripts/verify_quantiser_repro.py --role candidate \
        --onnx-fp32 onnx_ce10ep_1 --reference-int8 int8_ce10ep_1 \
        --reference-ort 1.30.0 --pinned-ort 1.29.0

WHY THE CONTROL IS NOT OPTIONAL. The candidate phase asks "do 1.29.0's tensors differ
from 1.30.0's?" A bare NOT-IDENTICAL answers that only if the comparison would have said
IDENTICAL when nothing differed. It would not, unless quantisation is reproducible across
everything ELSE that differs between here and Kaggle — and three things do:

  * ISA. Kaggle quantised on x86; this check runs on arm64.
  * optimum. `ORTQuantizer` is optimum's front end over `onnxruntime.quantization`.
  * onnx. The serializer that writes the tensors out.

The control holds the ORT version FIXED at the reference's own and re-quantises. If it
reproduces the reference exactly, all three are ruled out at once and a later difference
is attributable to ORT. IF THE CONTROL FAILS, THE INSTRUMENT IS BROKEN AND THE CANDIDATE
RESULT MEANS NOTHING — so `--role candidate` REFUSES TO RUN until a passing control for
the same two artefacts is on disk. That refusal is the point: a NOT-IDENTICAL from a
broken instrument looks exactly like a real quantiser difference.

WHY ANY OF THIS. E1b commit 1 installed onnxruntime 1.29.0 and commit 2 installed 1.30.0
— the Kaggle install cell pinned only `transformers`. Training is torch-only, so the drift
touches ONLY commit 2's ONNX-export + INT8-quantise tail. That tail is what the gate
reads: the gate compares E1b INT8 against the HOST BASELINE INT8, and the host baseline
was quantised under ORT 1.29.0 (results/tier0_ce_hostB_seed1.json -> train_env). A
quantiser version change is therefore a SECOND VARIABLE inside a difference that is
supposed to isolate epochs.

OUTCOMES.
  control NOT identical -> instrument unusable. Do not run the candidate; report that the
      confound could not be tested and the gate carries it as a stated limitation.
  control identical, candidate identical -> no confound. Gate proceeds unchanged.
  control identical, candidate NOT identical -> real ORT difference. The locally
      re-quantised 1.29.0 bytes are written beside the reference; the gate reads THOSE,
      matching the host baseline's quantiser, and the difference is disclosed.
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


CONTROL_PATH = Path("results/quantiser_repro_control.json")
CANDIDATE_PATH = Path("results/quantiser_repro.json")


def _artefact_identity(onnx_fp32: Path, reference: Path) -> dict:
    """What a control must have covered for it to license a candidate run."""
    return {"onnx_fp32": str(onnx_fp32.resolve()),
            "reference_int8": str(reference.resolve())}


def _require_passing_control(identity: dict) -> dict:
    """A candidate run is only meaningful behind a control that PASSED on these files."""
    if not CONTROL_PATH.exists():
        raise SystemExit(
            f"REFUSING to run the candidate phase: no control result at {CONTROL_PATH}.\n"
            f"The candidate asks whether a DIFFERENT onnxruntime produces different "
            f"tensors. That question is only answerable if this machine reproduces the "
            f"reference when the onnxruntime is the SAME — otherwise a NOT-IDENTICAL "
            f"result is indistinguishable from arm64-vs-x86, an optimum difference, or "
            f"an onnx serializer difference.\n"
            f"Run --role control first.")
    control = json.loads(CONTROL_PATH.read_text())
    if not control.get("identical"):
        raise SystemExit(
            f"REFUSING to run the candidate phase: the control at {CONTROL_PATH} did NOT "
            f"reproduce the reference ({control.get('n_differing')} initializers differ) "
            f"with the onnxruntime held at the reference's own version "
            f"{control.get('reference_ort')}.\n"
            f"THE INSTRUMENT IS BROKEN, not the quantiser. Something other than the ORT "
            f"version already prevents byte reproduction here — ISA, optimum or onnx. A "
            f"candidate result would be uninterpretable.\n"
            f"Report that the confound COULD NOT BE TESTED and let the gate carry it as "
            f"a stated limitation.")
    if control.get("artefact_identity") != identity:
        raise SystemExit(
            f"REFUSING: the control on disk covered different files.\n"
            f"  control  : {control.get('artefact_identity')}\n"
            f"  requested: {identity}\n"
            f"A control licenses the artefacts it actually ran on and no others.")
    return control


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", required=True, choices=("control", "candidate"),
                    help="control: same ORT as the reference. candidate: the pinned ORT.")
    ap.add_argument("--onnx-fp32", type=Path, required=True,
                    help="the FP32 ONNX export the reference was quantised from")
    ap.add_argument("--reference-int8", type=Path, required=True,
                    help="the downloaded INT8 artefact to check")
    ap.add_argument("--reference-ort", required=True,
                    help="the ORT version that MADE the reference (E1b commit 2: 1.30.0)")
    ap.add_argument("--pinned-ort", default=None,
                    help="candidate role: the host baseline's ORT (1.29.0)")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    versions = tool_versions()
    local_ort = versions["onnxruntime"]
    identity = _artefact_identity(args.onnx_fp32, args.reference_int8)

    # THE ENV MUST MATCH THE ROLE, and this is asserted rather than trusted: a control
    # accidentally run under the pinned ORT tests nothing and would "pass" the instrument
    # by measuring the wrong pair.
    if args.role == "control":
        expected = args.reference_ort
        why = ("the control must hold onnxruntime at the reference's OWN version, so "
               "that anything it finds is NOT an ORT difference")
    else:
        if not args.pinned_ort:
            raise SystemExit("--pinned-ort is required for --role candidate")
        expected = args.pinned_ort
        why = "the candidate must run under the host baseline's quantiser"
    if local_ort != expected:
        raise SystemExit(
            f"REFUSING: --role {args.role} needs onnxruntime {expected}, but this "
            f"environment has {local_ort}. {why.capitalize()}.\n"
            f"Build the throwaway env for this role and re-run.")

    control = _require_passing_control(identity) if args.role == "candidate" else None

    ref_onnx = next(args.reference_int8.glob("*.onnx"))
    print("=" * 70)
    print(f"QUANTISER REPRODUCIBILITY — {args.role.upper()} PHASE")
    print("=" * 70)
    print(f"  this machine : ort {local_ort}  onnx {versions['onnx']}  "
          f"optimum {versions['optimum']}")
    print(f"  reference made by ort {args.reference_ort}")
    print(f"  fp32 onnx    : {args.onnx_fp32}")
    print(f"  reference    : {ref_onnx}")
    if control:
        print(f"  control      : PASSED (ort {control['local_ort']}, "
              f"{control['n_shared']} initializers reproduced)")

    loadable, load_err = can_load(ref_onnx)
    print(f"\n  reference loads under ort {local_ort}: {loadable}")
    if not loadable:
        print(f"    {load_err}")

    work = args.work or (args.reference_int8.parent /
                         f"{args.reference_int8.name}__requant_{args.role}_{local_ort}")
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

    if args.role == "control":
        if identical:
            print("  CONTROL PASSED — this machine reproduces the reference byte for "
                  "byte with the ORT version held fixed.")
            print("  arm64-vs-x86, optimum and onnx are all ruled out. A difference in "
                  "the candidate phase is attributable to onnxruntime.")
        else:
            print("  CONTROL FAILED — THE INSTRUMENT IS BROKEN.")
            print("  With onnxruntime held at the reference's own version, this machine "
                  "still does not reproduce it. The cause is one of ISA (arm64 here, "
                  "x86 on Kaggle), optimum, or onnx — this check cannot say which.")
            print("  DO NOT run the candidate phase. Report that the confound could not "
                  "be tested and let the gate carry it as a stated limitation.")
    else:
        if identical:
            print(f"  CANDIDATE IDENTICAL — ort {args.pinned_ort} and "
                  f"{args.reference_ort} produce the same tensors. NO CONFOUND; the "
                  f"gate proceeds unchanged.")
        else:
            print(f"  CANDIDATE NOT IDENTICAL — ort {args.pinned_ort} and "
                  f"{args.reference_ort} differ on {len(differing)} initializers.")
            for n in differing[:10]:
                print(f"    differs: {n}")
            if len(differing) > 10:
                print(f"    … and {len(differing) - 10} more")
            print(f"\n  Behind a PASSING control, this is a real onnxruntime difference.")
            print(f"  The {args.pinned_ort}-quantised model is at:")
            print(f"    {local_onnx}")
            print("  Score the GATE against those bytes — they match the host "
                  "baseline's quantiser — and disclose the difference.")

    payload = {"role": args.role,
               "artefact_identity": identity,
               "versions_this_machine": versions,
               "local_ort": local_ort,
               "reference_ort": args.reference_ort,
               "pinned_ort": args.pinned_ort,
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
               "control_result": None if control is None else {
                   "identical": control["identical"], "local_ort": control["local_ort"]},
               "interpretation": (
                   "CONTROL: identical means arm64-vs-x86, optimum and onnx are ruled "
                   "out, so the candidate phase isolates onnxruntime. Not identical "
                   "means the instrument cannot test the confound at all."
                   if args.role == "control" else
                   "CANDIDATE, valid only behind a passing control: identical means the "
                   "two onnxruntime versions agree and the gate is unconfounded; not "
                   "identical means they disagree and the gate must read the pinned-ORT "
                   "bytes with the difference disclosed.")}
    out = args.out or (CONTROL_PATH if args.role == "control" else CANDIDATE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0 if identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
