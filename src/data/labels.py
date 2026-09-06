"""Canonical label space and output normalization.

Deliberately strict: normalization fixes case, whitespace and trivial surrounding
punctuation, and nothing else. There is NO fuzzy matching and NO nearest-neighbour
guessing. An output that does not map to a canonical label is a *format failure* —
a real, reportable failure mode of the LLM tiers — not something to paper over with
a best guess. See :class:`FormatFailureCounter`; the resulting rate is a metric.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DEFAULT_LABELS_PATH",
    "FormatFailureCounter",
    "LabelNormalizer",
    "normalize_key",
    "load_labels",
    "save_labels",
]

DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[2] / "configs" / "labels.json"

# Characters stripped from the ends of a candidate only: quotes, brackets, and
# sentence punctuation a model may append. Never stripped from the interior.
_EDGE_CHARS = " \t\r\n\"'`“”‘’*_.,;:!?()[]{}<>"
_WHITESPACE = re.compile(r"\s+")


def normalize_key(text: str) -> str:
    """Fold a string to its lookup key: NFKC, edge punctuation off, collapsed
    whitespace, casefolded. Interior punctuation and hyphens are preserved, so
    ``Anti-Corruption Laws`` and ``Anti Corruption Laws`` remain DISTINCT keys."""
    key = unicodedata.normalize("NFKC", text)
    key = key.strip(_EDGE_CHARS)
    key = _WHITESPACE.sub(" ", key)
    return key.casefold()


def load_labels(path: str | Path = DEFAULT_LABELS_PATH) -> list[str]:
    """Load the canonical ordered label list written by :func:`save_labels`."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["labels"])


def save_labels(labels: list[str], path: str | Path = DEFAULT_LABELS_PATH) -> Path:
    """Write the canonical ordered label list. Index order IS the dataset's order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "coastalcph/lex_glue:ledgar",
        "num_labels": len(labels),
        "note": "Order is the dataset ClassLabel order; index i is class i. Do not sort.",
        "labels": list(labels),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


class LabelNormalizer:
    """Maps a model output string onto a canonical label, or ``None``.

    ``None`` means "this output is not a label we recognise" and is a measured
    outcome, not an error to be recovered from.
    """

    def __init__(self, labels: list[str]) -> None:
        self.labels = list(labels)
        self._lookup: dict[str, str] = {}
        collisions: dict[str, list[str]] = {}
        for label in self.labels:
            key = normalize_key(label)
            if key in self._lookup and self._lookup[key] != label:
                collisions.setdefault(key, [self._lookup[key]]).append(label)
            self._lookup[key] = label
        if collisions:
            raise ValueError(
                "labels collide under normalization, so a normalized output could not "
                f"be attributed to one class: {collisions}"
            )

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_LABELS_PATH) -> "LabelNormalizer":
        return cls(load_labels(path))

    def normalize(self, output: str | None) -> str | None:
        """Return the canonical label for ``output``, or ``None`` if unmatched."""
        if output is None:
            return None
        return self._lookup.get(normalize_key(output))

    def index_of(self, label: str) -> int:
        """Canonical index of an exact canonical label."""
        return self.labels.index(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, output: object) -> bool:
        return isinstance(output, str) and self.normalize(output) is not None


@dataclass
class FormatFailureCounter:
    """Counts unmatched model outputs. ``failure_rate`` is a reported metric.

    Keeps a bounded sample of the offending strings so the report can show *how* the
    model failed, not just how often.
    """

    total: int = 0
    unmatched: int = 0
    max_samples: int = 25
    samples: list[str] = field(default_factory=list)

    def record(self, raw_output: str | None, matched: str | None) -> str | None:
        """Record one outcome and pass ``matched`` back through."""
        self.total += 1
        if matched is None:
            self.unmatched += 1
            if len(self.samples) < self.max_samples:
                self.samples.append("<None>" if raw_output is None else raw_output)
        return matched

    @property
    def matched(self) -> int:
        return self.total - self.unmatched

    @property
    def failure_rate(self) -> float:
        """Fraction of outputs that did not map to a canonical label. 0.0 if empty."""
        return self.unmatched / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "matched": self.matched,
            "unmatched": self.unmatched,
            "format_failure_rate": self.failure_rate,
            "samples": list(self.samples),
        }
