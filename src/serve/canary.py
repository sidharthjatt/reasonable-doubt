"""Startup canary for the deployed Tier 0, and the CPU/runtime facts that explain it.

WHY THIS EXISTS, and it is not a general health check
-----------------------------------------------------
Kaggle's x86 CPU scored the INT8 artefact at macro-F1 **0.0037** against FP32's 0.7636
on the same weights — chance, with **no error raised anywhere**. INT8 kernels are
ISA-specific: without AVX-512 VNNI the quantised path can degrade silently rather than
refuse. HF Spaces is x86, so the deployment target is exactly the hardware class that
produced that failure, and a service that starts anyway would serve chance-level labels
with a plausible-looking confidence on every request.

So the canary is a REFUSAL, not a warning. If measured accuracy on a fixed row set falls
below the floor, the app does not start (hard rule 11: no silent degradation).

WHY THE ROWS COME FROM TRAIN
----------------------------
The canary is a smoke test, not a measurement, and it must never consume an evaluation
role. `test_3000` is the reporting set, `dev_2000` is reserved for router thresholds
(hard rule 1), and `train_holdout_3000` is the checkpoint-selection set. All three are
off limits. The canary therefore draws from TRAIN, explicitly excluding the rows already
claimed by `train_holdout_3000` and `exemplars_8` so it overlaps no other role at all.

Because these are rows Tier 0 was fitted on, canary accuracy is HIGHER than test
accuracy and is NOT comparable to it. It is a fixed reference point for "are the INT8
kernels on this host producing what they produced on the host we measured", nothing more.

WHY IT IS NOT IN `MANIFEST_SPECS`
---------------------------------
`build_manifest` is pure in ``(ds, spec)``. The canary's row set depends on the contents
of two OTHER manifests, so it cannot satisfy that contract. Rather than weaken the
contract for a non-evaluation set, the canary carries its own file with the same
integrity fields and the same drift check.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.loading import get_split, hash_texts, label_names, resolve_split
from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest

__all__ = [
    "CANARY_PATH",
    "CANARY_SEED",
    "CANARY_N",
    "CanarySet",
    "CanaryFailure",
    "CanaryResult",
    "build_canary",
    "cpu_isa_flags",
    "load_canary",
    "onnxruntime_version",
    "run_canary",
]

CANARY_PATH = Path(__file__).resolve().parents[2] / "configs" / "canary_200.json"
CANARY_SEED = 20260910
CANARY_N = 200

# Train-split manifests whose rows already carry a role. The canary excludes them.
RESERVED_TRAIN_MANIFESTS = ("train_holdout_3000", "exemplars_8")


class CanaryFailure(RuntimeError):
    """The deployed Tier 0 did not reproduce its reference accuracy on this host."""


@dataclass(frozen=True)
class CanarySet:
    """A frozen set of TRAIN rows, verifiable against the dataset like a manifest."""

    split: str
    n: int
    seed: int
    indices: list[int]
    text_sha256: str
    labels_sha256: str
    excluded_manifests: list[str]
    purpose: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "split": self.split,
            "n": self.n,
            "seed": self.seed,
            "excluded_manifests": self.excluded_manifests,
            "text_sha256": self.text_sha256,
            "labels_sha256": self.labels_sha256,
            "indices": self.indices,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CanarySet":
        return cls(
            split=d["split"], n=d["n"], seed=d["seed"], indices=list(d["indices"]),
            text_sha256=d["text_sha256"], labels_sha256=d["labels_sha256"],
            excluded_manifests=list(d["excluded_manifests"]), purpose=d["purpose"],
        )


def _reserved_indices(manifest_dir: Path) -> set[int]:
    reserved: set[int] = set()
    for name in RESERVED_TRAIN_MANIFESTS:
        m = load_manifest(name, manifest_dir)
        if resolve_split(m.split) != resolve_split("train"):
            raise ValueError(
                f"{name!r} is on split {m.split!r}, not train; the canary's exclusion "
                f"list assumes train-split manifests and refuses to guess")
        reserved |= set(m.indices)
    return reserved


def build_canary(ds, *, n: int = CANARY_N, seed: int = CANARY_SEED,
                 manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> CanarySet:
    """Draw ``n`` TRAIN rows that no other manifest claims.

    Uniform over eligible rows, i.e. a draw from the class prior — the canary should
    resemble traffic, not a stratified evaluation sample. At n=200 over 100 long-tailed
    classes a stratified draw would be dominated by quotas below 1 anyway.
    """
    import numpy as np

    split_name = resolve_split("train")
    split = get_split(ds, split_name)
    reserved = _reserved_indices(manifest_dir)
    eligible = np.array([i for i in range(len(split)) if i not in reserved])
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} eligible train rows for n={n}")

    rng = np.random.default_rng(seed)
    picked = sorted(int(eligible[i]) for i in
                    rng.choice(len(eligible), size=n, replace=False))
    if set(picked) & reserved:
        raise AssertionError("canary overlaps a reserved manifest after exclusion")

    rows = split.select(picked)
    names = label_names(ds)
    return CanarySet(
        split=split_name, n=n, seed=seed, indices=picked,
        text_sha256=hash_texts(list(rows["text"])),
        labels_sha256=hash_texts(names),
        excluded_manifests=list(RESERVED_TRAIN_MANIFESTS),
        purpose=("startup smoke test for the deployed INT8 Tier 0; TRAIN rows, so "
                 "accuracy here is NOT comparable to test_3000"),
    )


def load_canary(path: Path = CANARY_PATH) -> CanarySet:
    if not path.exists():
        raise FileNotFoundError(
            f"no canary set at {path}. Build it with scripts/build_canary.py. The "
            f"service refuses to start without one rather than skipping the check.")
    return CanarySet.from_dict(json.loads(path.read_text()))


def verify_canary(canary: CanarySet, ds) -> None:
    """Re-hash the rows and refuse on drift, exactly as `verify_manifest` does."""
    split = get_split(ds, canary.split)
    rows = split.select(canary.indices)
    if hash_texts(list(rows["text"])) != canary.text_sha256:
        raise CanaryFailure(
            "CANARY TEXT DRIFT: the rows at these indices no longer hash to the "
            "recorded sha256. The dataset changed under a frozen row set.")
    if hash_texts(label_names(ds)) != canary.labels_sha256:
        raise CanaryFailure("CANARY LABEL-SPACE DRIFT: the 100 labels changed.")


# --------------------------------------------------------------- hardware disclosure

def onnxruntime_version() -> str | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    return getattr(ort, "__version__", None)


def cpu_isa_flags() -> dict[str, bool | None]:
    """avx512_vnni / avx512f / avx2 for this CPU. ``None`` means UNREADABLE, not absent.

    The distinction is the whole point. On arm64 these x86 flags are meaningless and
    come back None; on x86 Linux (HF Spaces) they are read from /proc/cpuinfo and a
    False on avx512_vnni is the condition under which INT8 degraded to chance on
    Kaggle. Reporting None as False would turn "we could not tell" into "it is absent",
    which is the substitution hard rule 11 forbids.
    """
    wanted = ("avx512_vnni", "avx512f", "avx2")
    flags: set[str] | None = None

    if platform.system() == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text()
        except OSError:
            text = ""
        for line in text.splitlines():
            if line.startswith("flags") and ":" in line:
                flags = set(line.split(":", 1)[1].split())
                break
    elif platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                flags = set(re.split(r"\s+", out.stdout.lower().strip()))
        except (OSError, subprocess.SubprocessError):
            flags = None

    if flags is None:
        return {k: None for k in wanted}
    return {k: (k in flags) for k in wanted}


def hardware_report() -> dict[str, Any]:
    return {
        "onnxruntime_version": onnxruntime_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_isa_flags": cpu_isa_flags(),
    }


# ------------------------------------------------------------------- the check itself

@dataclass(frozen=True)
class CanaryResult:
    accuracy: float
    n: int
    n_correct: int
    floor: float
    passed: bool
    hardware: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"accuracy": self.accuracy, "n": self.n, "n_correct": self.n_correct,
                "floor": self.floor, "passed": self.passed, "hardware": self.hardware}


def run_canary(tier0, ds, canary: CanarySet, *, floor: float) -> CanaryResult:
    """Classify every canary row and compare accuracy against ``floor``.

    Raises:
        CanaryFailure: if accuracy is below the floor. The caller must not catch this
            and start anyway — that is the silent-degradation path this guards.
    """
    verify_canary(canary, ds)
    split = get_split(ds, canary.split)
    rows = split.select(canary.indices)
    names = label_names(ds)
    gold = [names[int(x)] for x in rows["label"]]

    preds = [tier0.classify(t).label for t in rows["text"]]
    n_correct = sum(1 for p, g in zip(preds, gold) if p == g)
    acc = n_correct / len(gold)
    result = CanaryResult(accuracy=acc, n=len(gold), n_correct=n_correct, floor=floor,
                          passed=acc >= floor, hardware=hardware_report())
    if not result.passed:
        raise CanaryFailure(
            f"TIER 0 CANARY FAILED: accuracy {acc:.4f} on {len(gold)} frozen TRAIN rows "
            f"is below the floor {floor:.4f}.\n"
            f"hardware: {result.hardware}\n"
            f"This is the Kaggle failure mode: INT8 kernels can degrade to chance on a "
            f"CPU without AVX-512 VNNI and raise nothing. REFUSING TO START rather than "
            f"serve chance-level labels with plausible confidences.")
    return result
