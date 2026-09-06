"""Output schema parsing and cacheable-prefix prompt construction."""

from __future__ import annotations

import pytest

from src.data.labels import LabelNormalizer, load_labels
from src.data.prompts import (
    load_template,
    render_fewshot,
    render_zeroshot,
    select_exemplars,
)
from src.data.schema import ClauseClassification, ParseFailure, parse_response

# ------------------------------------------------------------------------ schema


def test_parses_bare_json():
    out = parse_response('{"label": "Governing Laws", "confidence": 0.82}')
    assert out.label == "Governing Laws"
    assert out.confidence == 0.82


def test_parses_fenced_json_when_not_strict():
    raw = 'Here you go:\n```json\n{"label": "Waivers", "confidence": 0.4}\n```'
    assert parse_response(raw).label == "Waivers"


def test_strict_mode_rejects_prose_wrapped_output():
    raw = 'Sure!\n{"label": "Waivers", "confidence": 0.4}'
    assert parse_response(raw).label == "Waivers"
    with pytest.raises(ParseFailure):
        parse_response(raw, strict=True)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "Governing Laws",                                  # bare label, not JSON
        '{"label": "Waivers"}',                            # missing confidence
        '{"confidence": 0.5}',                             # missing label
        '{"label": "Waivers", "confidence": 1.5}',         # out of range
        '{"label": "Waivers", "confidence": -0.1}',        # out of range
        '{"label": "Waivers", "confidence": "high"}',      # not a number
        '{"label": "Waivers", "confidence": 0.5, "why": "x"}',  # extra field
    ],
)
def test_malformed_outputs_raise_parse_failure(raw):
    with pytest.raises(ParseFailure):
        parse_response(raw)


def test_confidence_is_not_silently_clipped():
    """Miscalibration is a finding to measure, so out-of-range values fail loudly."""
    with pytest.raises(Exception):
        ClauseClassification(label="Waivers", confidence=1.2)


def test_parsing_does_not_validate_the_label_string():
    """The schema checks shape; label legitimacy is the normalizer's job."""
    out = parse_response('{"label": "Not A Real Label", "confidence": 0.9}')
    assert out.label == "Not A Real Label"
    assert LabelNormalizer(load_labels()).normalize(out.label) is None


# ----------------------------------------------------------------------- prompts


@pytest.fixture(scope="module")
def labels():
    return load_labels()


def test_zeroshot_contains_every_label_and_the_schema(labels):
    p = render_zeroshot(labels)
    for label in labels:
        assert label in p.text
    assert '"label"' in p.text and '"confidence"' in p.text
    assert p.n_exemplars == 0


def test_rendered_prompt_excludes_any_clause_text(labels):
    """The cacheable prefix must be per-request invariant: no clause may leak into it."""
    p = render_zeroshot(labels)
    assert "{labels}" not in p.text and "{schema}" not in p.text
    assert "Provision:" not in p.text


def test_zeroshot_rendering_is_byte_stable(labels):
    """Byte-identical across calls, or prompt caching cannot hit."""
    a, b = render_zeroshot(labels), render_zeroshot(labels)
    assert a.text == b.text
    assert a.sha256 == b.sha256


def test_template_hash_identifies_the_wording():
    tpl = load_template("zeroshot")
    assert len(tpl.sha256) == 64
    assert load_template("zeroshot").sha256 == tpl.sha256
    assert load_template("fewshot").sha256 != tpl.sha256


def test_prompt_hash_changes_if_the_label_block_changes(labels):
    a = render_zeroshot(labels)
    b = render_zeroshot(labels[:-1])
    assert a.sha256 != b.sha256


def test_n_exemplars_is_a_parameter_not_a_constant(ledgar, labels):
    """Stage 2's budget gate may force 8 down to 4; both must render."""
    eight = render_fewshot(labels, ledgar["train"], n_exemplars=8, seed=1)
    four = render_fewshot(labels, ledgar["train"], n_exemplars=4, seed=1)
    assert eight.n_exemplars == 8 and four.n_exemplars == 4
    assert len(eight.exemplar_indices) == 8
    assert len(four.text) < len(eight.text)
    assert eight.sha256 != four.sha256


def test_fewshot_is_deterministic_and_byte_stable(ledgar, labels):
    a = render_fewshot(labels, ledgar["train"], n_exemplars=8, seed=42)
    b = render_fewshot(labels, ledgar["train"], n_exemplars=8, seed=42)
    assert a.text == b.text and a.sha256 == b.sha256
    assert a.exemplar_indices == b.exemplar_indices


def test_exemplars_come_from_train_only(ledgar, labels):
    """Hard rule 1: dev is for calibration, test is touched once. Neither seeds prompts."""
    p = render_fewshot(labels, ledgar["train"], n_exemplars=8, seed=42)
    train_texts = set(ledgar["train"].select(list(p.exemplar_indices))["text"])
    assert len(train_texts) == 8
    for text in train_texts:
        assert text[:200] in p.text


def test_exemplars_are_one_per_distinct_class(ledgar):
    idx = select_exemplars(ledgar["train"], [], n_exemplars=8, seed=3)
    classes = [int(ledgar["train"][i]["label"]) for i in idx]
    assert len(set(classes)) == 8


def test_too_many_exemplars_rejected(ledgar):
    with pytest.raises(ValueError, match="exceeds"):
        select_exemplars(ledgar["train"], [], n_exemplars=101, seed=1)


def test_zero_exemplars_allowed(ledgar):
    assert select_exemplars(ledgar["train"], [], n_exemplars=0, seed=1) == []


def test_prompt_metadata_is_recorded_for_provenance(ledgar, labels):
    meta = render_fewshot(labels, ledgar["train"], n_exemplars=4, seed=9).as_metadata()
    assert meta["n_exemplars"] == 4
    assert meta["exemplar_seed"] == 9
    assert len(meta["prompt_sha256"]) == 64
    assert len(meta["template_sha256"]) == 64
    assert len(meta["exemplar_indices"]) == 4
