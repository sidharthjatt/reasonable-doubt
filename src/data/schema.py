"""Output schema for the LLM tiers.

The model returns a single JSON object. ``confidence`` is the verbalized-confidence
signal consumed by router R2. We fully expect it to be poorly calibrated — measuring
that miscalibration is a planned finding, not a bug, so nothing here rescales,
clips-with-a-warning, or otherwise launders the value.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "ClauseClassification",
    "ParseFailure",
    "SCHEMA_JSON",
    "parse_response",
]


class ClauseClassification(BaseModel):
    """``{"label": "<one of the 100 canonical labels>", "confidence": <float 0-1>}``"""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Exactly one of the 100 canonical LEDGAR labels.")
    confidence: float = Field(ge=0.0, le=1.0, description="Verbalized confidence, 0-1.")


SCHEMA_JSON = '{"label": "<label>", "confidence": <number between 0 and 1>}'

# Fallback extractor for outputs wrapped in prose or ```json fences. This recovers
# JSON *shape* only; it never guesses a label — an unrecognised label string still
# fails normalization downstream and is counted as a format failure.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ParseFailure(ValueError):
    """The model output was not a usable ClauseClassification object."""


def parse_response(raw: str | None, *, strict: bool = False) -> ClauseClassification:
    """Parse a model output into a :class:`ClauseClassification`.

    Args:
        raw: The model's text output.
        strict: If True, ``raw`` must be exactly one bare JSON object — no code
            fences, no surrounding prose. Use this when measuring how well a model
            follows the "return only JSON" instruction.

    Raises:
        ParseFailure: on any malformed or non-conforming output. Callers route this
            into :class:`~src.data.labels.FormatFailureCounter` rather than retrying
            silently.
    """
    if raw is None:
        raise ParseFailure("output was None")

    candidates = [raw.strip()]
    if not strict:
        for pattern in (_FENCE, _OBJECT):
            match = pattern.search(raw)
            if match:
                candidates.append(match.group(1 if pattern is _FENCE else 0).strip())

    last: Exception | None = None
    for candidate in candidates:
        try:
            return ClauseClassification.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last = exc
    raise ParseFailure(f"could not parse output as {SCHEMA_JSON}: {last}") from last
