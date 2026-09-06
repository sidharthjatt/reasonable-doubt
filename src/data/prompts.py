"""Prompt template loading and rendering.

Templates live in ``configs/prompts/`` as files, never as inline strings, and are
identified downstream by their sha256 rather than a hand-maintained version number —
the hash is what a cache key and a results log can actually be audited against.

Cacheable-prefix design
-----------------------
A rendered template is the **entire static system prompt**: task description, the 100
label names, and (for few-shot) the exemplars. It is one contiguous block that does
not vary per request. The clause under test is NEVER part of it — it goes in the user
message. So the cache breakpoint sits cleanly at the end of the rendered system
prompt, and every request in a run shares a byte-identical prefix.

``n_exemplars`` is a parameter, not a constant: stage 2's budget gate may force it
from 8 down to 4 (see ``budget.stages.stage_2`` in ``configs/costs.yaml``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from src.data.schema import SCHEMA_JSON

if TYPE_CHECKING:
    from datasets import Dataset

__all__ = [
    "DEFAULT_PROMPT_DIR",
    "PromptTemplate",
    "RenderedPrompt",
    "format_labels_block",
    "load_template",
    "render_fewshot",
    "render_zeroshot",
    "select_exemplars",
]

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptTemplate:
    """A template file plus the hash that identifies this exact wording."""

    name: str
    text: str
    sha256: str

    @property
    def short_sha(self) -> str:
        return self.sha256[:12]


@dataclass(frozen=True)
class RenderedPrompt:
    """A fully rendered, per-request-invariant system prompt.

    ``sha256`` identifies the exact cacheable prefix; ``template_sha256`` identifies
    the wording it came from.
    """

    text: str
    sha256: str
    template_name: str
    template_sha256: str
    n_exemplars: int
    exemplar_indices: tuple[int, ...] = ()
    exemplar_seed: int | None = None

    def as_metadata(self) -> dict:
        """Provenance to record alongside any run using this prompt."""
        return {
            "template_name": self.template_name,
            "template_sha256": self.template_sha256,
            "prompt_sha256": self.sha256,
            "n_exemplars": self.n_exemplars,
            "exemplar_indices": list(self.exemplar_indices),
            "exemplar_seed": self.exemplar_seed,
        }


def load_template(name: str, directory: str | Path = DEFAULT_PROMPT_DIR) -> PromptTemplate:
    """Load ``<name>.txt`` from the prompt directory."""
    path = Path(directory) / f"{name}.txt"
    text = path.read_text(encoding="utf-8")
    return PromptTemplate(name=name, text=text, sha256=_sha256(text))


def format_labels_block(labels: Sequence[str]) -> str:
    """The 100 labels, one per line, in canonical index order. Order is never sorted
    or shuffled: it is part of the cacheable prefix and must be byte-stable."""
    return "\n".join(f"- {label}" for label in labels)


def render_zeroshot(
    labels: Sequence[str],
    *,
    template: PromptTemplate | None = None,
    directory: str | Path = DEFAULT_PROMPT_DIR,
) -> RenderedPrompt:
    """Render the zero-shot system prompt."""
    tpl = template or load_template("zeroshot", directory)
    text = tpl.text.format(labels=format_labels_block(labels), schema=SCHEMA_JSON)
    return RenderedPrompt(
        text=text,
        sha256=_sha256(text),
        template_name=tpl.name,
        template_sha256=tpl.sha256,
        n_exemplars=0,
    )


def select_exemplars(
    train: "Dataset",
    labels: Sequence[str],
    n_exemplars: int,
    seed: int,
) -> list[int]:
    """Pick ``n_exemplars`` TRAIN row indices, deterministically, one per distinct class.

    Exemplars come from the train split only — never dev, never test (hard rule 1).
    Classes are drawn without replacement so the exemplar block shows variety rather
    than repeating a frequent class.
    """
    import numpy as np

    if n_exemplars < 0:
        raise ValueError(f"n_exemplars must be >= 0, got {n_exemplars}")
    if n_exemplars == 0:
        return []

    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(train["label"]):
        by_class.setdefault(int(label), []).append(idx)

    present = sorted(by_class)
    if n_exemplars > len(present):
        raise ValueError(
            f"n_exemplars={n_exemplars} exceeds the {len(present)} classes in train"
        )

    rng = np.random.default_rng(seed)
    chosen_classes = rng.choice(len(present), size=n_exemplars, replace=False)
    picked = [
        by_class[present[c]][int(rng.integers(len(by_class[present[c]])))]
        for c in chosen_classes
    ]
    return sorted(picked)


def render_fewshot(
    labels: Sequence[str],
    train: "Dataset",
    *,
    n_exemplars: int,
    seed: int,
    template: PromptTemplate | None = None,
    directory: str | Path = DEFAULT_PROMPT_DIR,
    max_exemplar_chars: int | None = 1200,
) -> RenderedPrompt:
    """Render the few-shot system prompt with ``n_exemplars`` worked examples.

    Exemplars are formatted in exactly the request/response shape the model must
    produce, so the demonstration and the instruction cannot disagree.
    """
    tpl = template or load_template("fewshot", directory)
    indices = select_exemplars(train, labels, n_exemplars, seed)

    blocks = []
    for i, idx in enumerate(indices, start=1):
        row = train[idx]
        text = row["text"]
        if max_exemplar_chars is not None and len(text) > max_exemplar_chars:
            text = text[:max_exemplar_chars].rstrip() + " […]"
        # NOTE: a fixed demonstration confidence anchors the model's verbalized
        # confidence toward this value, which directly affects router R2's signal.
        # Flagged as a design decision to revisit, not an oversight.
        answer = f'{{"label": "{labels[int(row["label"])]}", "confidence": 0.9}}'
        blocks.append(f"Example {i}\nProvision:\n{text}\n\nAnswer:\n{answer}")

    rendered = tpl.text.format(
        labels=format_labels_block(labels),
        schema=SCHEMA_JSON,
        exemplars="\n\n---\n\n".join(blocks),
    )
    return RenderedPrompt(
        text=rendered,
        sha256=_sha256(rendered),
        template_name=tpl.name,
        template_sha256=tpl.sha256,
        n_exemplars=n_exemplars,
        exemplar_indices=tuple(indices),
        exemplar_seed=seed,
    )
