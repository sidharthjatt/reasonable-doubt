# =============================================================================
# TIER 0 SMOKE PROBE — run this BEFORE kaggle_tier0.py. ~90 seconds, GPU not needed.
#
# Why Tier 0 needs a probe MORE than Tier 1, not less:
#
#   * Tier 0 is the notebook that DOWNGRADES a preloaded package. Kaggle ships
#     transformers 5.0.0; optimum-onnx declares transformers<4.58.0, so CELL 1 pulls
#     4.57.6 over the top of it. Replacing a preloaded package is precisely the
#     operation that produced the mixed peft install (PREREGISTRATION 3p) — Tier 1
#     avoids it by forcing only peft, Tier 0 cannot avoid it at all.
#   * Its ONNX export runs for the FIRST TIME 3-5 hours in, after training. An export
#     or quantiser failure there costs the whole session, and the seed's fp32
#     checkpoint is on a disk that goes away with it.
#
# So this exercises the export path end to end, on an UNTRAINED model, in ~90 seconds:
#   untrained DeBERTa-v3-base -> ORTModelForSequenceClassification export
#   -> ORTQuantizer arm64 INT8 -> one onnxruntime forward pass on 4 rows.
#
# Run CELL 1 of kaggle_tier0.py first, RESTART THE KERNEL, then run this.
# If it passes, the export path works and Tier 0 can start.
# IF IT FAILS, DO NOT START TIER 0 — you would meet the same failure after training.
#
# The 90 s target assumes the DeBERTa weights are already in the HF cache. A cold
# first download of ~370 MB is additional and depends on Kaggle's network.
# =============================================================================
# ONE GPU, PINNED BEFORE TORCH IS IMPORTED — the same pin kaggle_tier0.py uses.
# A probe that runs on a different device configuration than the notebook it certifies
# is not certifying that notebook. This file was written before the DataParallel /
# effective-batch defect was known and did not have the pin; see PREREGISTRATION 3t.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import inspect, sys, time, traceback

T0 = time.time()
def step(n, what):
    print(f"\n[{n}] {what}   (+{time.time()-T0:.0f}s)", flush=True)

ok = True
try:
    step(1, "versions and install consistency")
    import importlib.metadata as md, pathlib
    import transformers, optimum, onnx, onnxruntime
    mods = {"transformers": transformers, "onnx": onnx, "onnxruntime": onnxruntime}
    for k, m in mods.items():
        print(f"    {k:<14} {m.__version__:<10} {m.__file__}")
    for k in ("optimum", "optimum-onnx"):
        try: print(f"    {k:<14} {md.version(k)}")
        except md.PackageNotFoundError:
            raise AssertionError(f"{k} is not installed — CELL 1 did not complete")

    # dist-info vs the imported module's own string. These are two records written to
    # two places on disk; a half-completed pip write leaves them disagreeing, and
    # __version__ alone cannot report that. This is the check that peft's version gate
    # lacked when it certified 0.19.1 while running 0.20.0's code.
    for k, m in mods.items():
        meta = md.version(k)
        assert meta == m.__version__, (
            f"MIXED INSTALL — {k}: dist-info says {meta}, imported module says "
            f"{m.__version__}. Restart the kernel; if it persists, start a fresh "
            f"session. Module file: {m.__file__}")
    print("    dist-info == __version__ for all three")

    # The downgrade is the whole risk here, so assert the range optimum-onnx declares.
    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
    assert (4, 36) <= tv < (4, 58), (
        f"transformers {transformers.__version__} is outside optimum-onnx's declared "
        f"range (>=4.36,<4.58). The downgrade in CELL 1 did not take — almost always "
        f"a missing kernel restart.")
    print(f"    transformers {transformers.__version__} is inside [4.36, 4.58)")

    step(2, "optimum signatures — the calls kaggle_tier0.py makes after training")
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    assert hasattr(AutoQuantizationConfig, "arm64"), \
        "AutoQuantizationConfig.arm64 missing in this optimum"
    for fn, need, label in ((AutoQuantizationConfig.arm64, ["is_static", "per_channel"],
                             "AutoQuantizationConfig.arm64"),
                            (ORTQuantizer.quantize, ["save_dir", "quantization_config"],
                             "ORTQuantizer.quantize")):
        params = set(inspect.signature(fn).parameters)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in inspect.signature(fn).parameters.values()):
            print(f"    {label}: takes **kwargs — NOT VERIFIABLE by introspection")
            continue
        missing = [k for k in need if k not in params]
        assert not missing, f"{label} does not accept {missing}; has {sorted(params)}"
        print(f"    {label}: accepts {need}")

    step(3, "untrained DeBERTa-v3-base, 100 labels (same construction as Tier 0)")
    import numpy as np, tempfile
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    MODEL, N_LABELS, MAX_LENGTH = "microsoft/deberta-v3-base", 100, 512
    work = pathlib.Path(tempfile.mkdtemp(prefix="t0probe_"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=N_LABELS)
    fp32 = work / "fp32"; model.save_pretrained(str(fp32)); tok.save_pretrained(str(fp32))
    print(f"    saved fp32 to {fp32} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    step(4, "ONNX export (this is the call that first runs 3-5h into Tier 0)")
    onnx_dir = work / "onnx"
    ORTModelForSequenceClassification.from_pretrained(str(fp32), export=True
        ).save_pretrained(str(onnx_dir))
    tok.save_pretrained(str(onnx_dir))
    graphs = sorted(onnx_dir.glob("*.onnx"))
    assert graphs, f"export produced no .onnx in {onnx_dir}: {sorted(onnx_dir.iterdir())}"
    print(f"    exported {[g.name for g in graphs]} "
          f"({sum(g.stat().st_size for g in graphs)/1e6:.0f} MB)")

    step(5, "ORTQuantizer arm64 INT8 (NOT avx512 — deployment target is Apple Silicon)")
    int8_dir = work / "int8"
    ORTQuantizer.from_pretrained(str(onnx_dir)).quantize(save_dir=str(int8_dir),
        quantization_config=AutoQuantizationConfig.arm64(is_static=False, per_channel=True))
    tok.save_pretrained(str(int8_dir))
    q = sorted(int8_dir.glob("*.onnx"))
    assert q, f"quantiser produced no .onnx in {int8_dir}: {sorted(int8_dir.iterdir())}"
    fp32_mb = sum(g.stat().st_size for g in graphs) / 1e6
    int8_mb = sum(g.stat().st_size for g in q) / 1e6
    print(f"    quantised {[g.name for g in q]} ({int8_mb:.0f} MB, "
          f"{fp32_mb/int8_mb:.1f}x smaller than fp32)")
    # A "quantised" graph the same size as the input means the quantiser silently did
    # nothing — a failure that otherwise presents as success (PREREGISTRATION 3e).
    assert int8_mb < 0.6 * fp32_mb, (
        f"INT8 graph is {int8_mb:.0f} MB against fp32's {fp32_mb:.0f} MB — that is not "
        f"a quantised model. The quantiser reported success without quantising.")

    step(6, "onnxruntime forward pass on 4 rows (the deployed inference path)")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(q[0]), providers=["CPUExecutionProvider"])
    names_in = {i.name for i in sess.get_inputs()}
    print(f"    graph inputs: {sorted(names_in)}")
    rows = ["This Agreement shall be governed by the laws of the State of New York.",
            "The Company shall indemnify and hold harmless each Indemnified Party.",
            "Neither party may assign this Agreement without prior written consent.",
            "Employee shall receive an annual base salary of $250,000."]
    enc = tok(rows, truncation=True, max_length=MAX_LENGTH, padding=True,
              return_tensors="np")
    dropped = sorted(set(enc) - names_in)
    if dropped:
        print(f"    NOTE: tokenizer produced {dropped}, not in the graph — not fed")
    feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names_in}
    assert feed, f"no tokenizer output matched the graph inputs {sorted(names_in)}"
    logits = sess.run(None, feed)[0]
    print(f"    logits {logits.shape} dtype={logits.dtype} "
          f"range [{logits.min():.3f}, {logits.max():.3f}]")
    assert logits.shape == (len(rows), N_LABELS), \
        f"expected {(len(rows), N_LABELS)}, got {logits.shape}"
    # An all-zero or non-finite output is the quantiser having destroyed the graph while
    # still returning one. The head is randomly initialised, so the ARGMAX is
    # meaningless here and is deliberately not asserted — only that real numbers came out.
    assert np.isfinite(logits).all(), "non-finite logits from the INT8 graph"
    assert logits.std() > 1e-6, \
        f"INT8 logits are constant (std={logits.std():.2e}) — the graph is degenerate"
    print(f"    finite, non-constant (std={logits.std():.3f}); argmax not checked "
          f"(untrained head)")

    import shutil; shutil.rmtree(work, ignore_errors=True)

except Exception:
    ok = False
    traceback.print_exc()

print("\n" + "=" * 70)
print(f"TIER 0 PROBE: {'PASSED — the export path works, Tier 0 can start' if ok else 'FAILED — DO NOT START TIER 0'}"
      f"   ({time.time()-T0:.0f}s)")
print("=" * 70)
if not ok:
    sys.exit(1)
