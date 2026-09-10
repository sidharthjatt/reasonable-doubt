"""Tier 0: the fine-tuned DeBERTa-v3-base encoder, ONNX INT8, served locally.

The model directory is CONFIGURED, never hardcoded. E1b may replace `int8_ce_1` with a
10-epoch artefact, and a path baked into code would keep serving the superseded weights
while the config, the threshold file and the report all described the new ones.

THE CONFIDENCE SIGNAL IS IMPORTED, NOT REIMPLEMENTED. `src.router.signals.margin` is the
function E5 calibrated the threshold with. A local softmax-and-subtract here would be a
second implementation of the deployed signal, free to drift from the calibrated one
while every test still passed — the scorer-bypass class this project has already paid
for twice. There is exactly one margin function and this serves it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.router.signals import margin as margin_signal
from src.serve.tiers import TierResult

__all__ = ["Tier0Encoder"]


@dataclass
class Tier0Encoder:
    """ONNX INT8 encoder. Loads once at construction; `classify` is inference only."""

    model_dir: Path
    labels: list[str]
    max_length: int = 512
    name: str = "tier0"
    is_stub: bool = False

    def __post_init__(self) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_dir = Path(self.model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Tier 0 model directory {self.model_dir} does not exist. It is set from "
                f"configs/serve.yaml; the service refuses to start rather than fall back "
                f"to an arbitrary artefact.")
        onnx_files = sorted(self.model_dir.glob("*.onnx"))
        if len(onnx_files) != 1:
            raise FileNotFoundError(
                f"expected exactly one .onnx in {self.model_dir}, found "
                f"{[f.name for f in onnx_files]}; refusing to guess which to serve")

        self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._sess = ort.InferenceSession(str(onnx_files[0]),
                                          providers=["CPUExecutionProvider"])
        self._wanted = {i.name for i in self._sess.get_inputs()}
        self._onnx_path = onnx_files[0]

    def logits(self, text: str) -> np.ndarray:
        enc = self._tok([text], truncation=True, max_length=self.max_length,
                        padding=True, return_tensors="np")
        feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self._wanted}
        return self._sess.run(None, feeds)[0]

    def classify(self, text: str) -> TierResult:
        lg = self.logits(text)
        idx = int(lg.argmax(-1)[0])
        if idx >= len(self.labels):
            raise IndexError(
                f"Tier 0 predicted class index {idx} but only {len(self.labels)} labels "
                f"are configured; the artefact and the label space disagree")
        return TierResult(
            label=self.labels[idx],
            confidence=float(margin_signal(lg)[0]),
            tier=self.name,
            is_stub=False,
        )

    def describe(self) -> dict:
        return {"model_dir": str(self.model_dir), "onnx": self._onnx_path.name,
                "max_length": self.max_length, "n_labels": len(self.labels)}
