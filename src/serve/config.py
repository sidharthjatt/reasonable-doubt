"""Service configuration, loaded from configs/serve.yaml.

Nothing about the deployed system is hardcoded here: the Tier 0 artefact, the router
threshold file, the escalation model and the canary floor all come from the file, so
swapping E1b's 10-epoch weights in is a config change plus a recalibration rather than
a code change.

THE ONE CROSS-CHECK THAT MATTERS. `router_threshold.json` records the artefact it was
calibrated against. If serve.yaml points Tier 0 somewhere else, the threshold describes
a margin distribution that is not the one being served, and the escalation rate silently
becomes whatever the new weights happen to produce. That is refused, not warned about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["PROJECT_ROOT", "RouterThreshold", "ServiceConfig", "ThresholdArtefactMismatch"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVE_CONFIG = PROJECT_ROOT / "configs" / "serve.yaml"


class ThresholdArtefactMismatch(RuntimeError):
    """The router threshold was calibrated against different weights than are served."""


@dataclass(frozen=True)
class RouterThreshold:
    """A threshold produced by dev_2000 calibration, with its full provenance."""

    signal: str
    threshold: float
    calibrated_on: str
    artefact: str
    seed: int
    target_escalation_rate: float
    achieved_escalation_rate_on_dev: float
    retained_tier0_accuracy_on_dev: float
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "RouterThreshold":
        if not path.exists():
            raise FileNotFoundError(
                f"no router threshold at {path}. Produce it with "
                f"scripts/calibrate_router_threshold.py (dev_2000 only, hard rule 1). "
                f"The service will not invent one at request time.")
        d = json.loads(path.read_text())
        if d["calibrated_on"] != "dev_2000":
            raise ValueError(
                f"threshold file says it was calibrated on {d['calibrated_on']!r}. "
                f"Hard rule 1 permits dev only; refusing to serve it.")
        return cls(
            signal=d["signal"], threshold=float(d["threshold"]),
            calibrated_on=d["calibrated_on"], artefact=d["artefact"],
            seed=int(d["seed"]),
            target_escalation_rate=float(d["target_escalation_rate"]),
            achieved_escalation_rate_on_dev=float(d["achieved_escalation_rate_on_dev"]),
            retained_tier0_accuracy_on_dev=float(d["retained_tier0_accuracy_on_dev"]),
            raw=d,
        )

    def as_dict(self) -> dict[str, Any]:
        return {"signal": self.signal, "threshold": self.threshold,
                "calibrated_on": self.calibrated_on, "artefact": self.artefact,
                "seed": self.seed,
                "target_escalation_rate": self.target_escalation_rate,
                "achieved_escalation_rate_on_dev":
                    self.achieved_escalation_rate_on_dev,
                "retained_tier0_accuracy_on_dev":
                    self.retained_tier0_accuracy_on_dev}


@dataclass(frozen=True)
class ServiceConfig:
    tier0_model_dir: Path
    tier0_max_length: int
    threshold: RouterThreshold
    tier2_model: str
    tier2_max_output_tokens: int
    tier2_temperature: float
    tier2_batch: bool
    canary_row_set: Path
    canary_min_accuracy: float
    canary_measured_accuracy: float
    canary_enabled: bool
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_SERVE_CONFIG,
             *, root: Path | None = None) -> "ServiceConfig":
        import yaml

        path = Path(path)
        root = root or PROJECT_ROOT
        d = yaml.safe_load(path.read_text())

        threshold = RouterThreshold.load(root / d["router"]["threshold_file"])
        model_dir = root / d["tier0"]["model_dir"]

        # The cross-check. Compare on the artefact's basename: the calibration npz
        # records a repo-relative path, serve.yaml may name a different prefix, but the
        # WEIGHTS must be the same ones.
        served, calibrated = model_dir.name, Path(threshold.artefact).name
        if served != calibrated:
            raise ThresholdArtefactMismatch(
                f"REFUSING TO START. Tier 0 serves {served!r} but the router threshold "
                f"in {d['router']['threshold_file']} was calibrated on {calibrated!r}.\n"
                f"The margin distribution is a property of the weights, so this "
                f"threshold does not describe the model being served and the escalation "
                f"rate would be whatever the new weights happen to produce.\n"
                f"Re-run scripts/calibrate_router_threshold.py against {served!r} on "
                f"dev_2000 before serving it.")

        if threshold.signal != d["router"]["signal"]:
            raise ThresholdArtefactMismatch(
                f"serve.yaml routes on {d['router']['signal']!r} but the threshold file "
                f"was calibrated for {threshold.signal!r}")

        return cls(
            tier0_model_dir=model_dir,
            tier0_max_length=int(d["tier0"]["max_length"]),
            threshold=threshold,
            tier2_model=d["tier2"]["model"],
            tier2_max_output_tokens=int(d["tier2"]["max_output_tokens"]),
            tier2_temperature=float(d["tier2"]["temperature"]),
            tier2_batch=bool(d["tier2"]["batch"]),
            canary_row_set=root / d["canary"]["row_set"],
            canary_min_accuracy=float(d["canary"]["min_accuracy"]),
            canary_measured_accuracy=float(d["canary"]["measured_accuracy"]),
            canary_enabled=bool(d["canary"]["enabled"]),
            raw=d,
        )
