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

# Reasoning models emit a delimited thinking block before the answer. Some expose it
# as a separate response field (gpt-oss on Groq); others inline it in `content`
# wrapped in <think>...</think> (Qwen3). The thinking block routinely contains draft
# JSON objects, so the ANSWER is what follows it — never the first object in the text.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Remove a delimited thinking block. An UNCLOSED block means the model was cut
    off mid-thought and never reached its answer — that is a truncation, and stripping
    it correctly leaves nothing to parse rather than salvaging a draft."""
    text = _THINK_BLOCK.sub("", text)
    return _UNCLOSED_THINK.sub("", text)


def _json_objects(text: str) -> list[str]:
    """Every balanced top-level ``{...}`` span, in order of appearance.

    A greedy ``\{.*\}`` would span from the first brace to the last and match nothing
    parseable when several objects are present, which is exactly the case inside a
    reasoning block. Brace counting is string-aware so a brace inside a JSON string
    literal cannot unbalance the scan.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    spans.append(text[start : i + 1])
    return spans


class ParseFailure(ValueError):
    """The model output was not a usable ClauseClassification object."""


def parse_response(raw: str | None, *, strict: bool = False) -> ClauseClassification:
    """Parse a model output into a :class:`ClauseClassification`.

    Args:
        raw: The model's text output.
        strict: If True, ``raw`` must be exactly one bare JSON object — no code
            fences, no reasoning block, no surrounding prose. Use this when measuring
            how well a model follows the "return only JSON" instruction.

    Note:
        Non-strict parsing recovers JSON *shape* from reasoning blocks, code fences and
        surrounding prose. It never guesses a label: an unrecognised label string still
        fails normalization downstream and is counted as a format failure. Extracting a
        model's stated answer from its own delimited reasoning is reading the output
        format correctly, not papering over a bad answer.

    Raises:
        ParseFailure: on any malformed or non-conforming output. Callers route this
            into :class:`~src.data.labels.FormatFailureCounter` rather than retrying
            silently.
    """
    if raw is None:
        raise ParseFailure("output was None")

    candidates = [raw.strip()]
    if not strict:
        body = _strip_reasoning(raw)
        fenced = _FENCE.search(body)
        if fenced:
            candidates.append(fenced.group(1).strip())
        # Last object first: the answer follows the reasoning, it does not precede it.
        candidates.extend(reversed(_json_objects(body)))

    last: Exception | None = None
    for candidate in candidates:
        try:
            return ClauseClassification.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last = exc
    raise ParseFailure(f"could not parse output as {SCHEMA_JSON}: {last}") from last
