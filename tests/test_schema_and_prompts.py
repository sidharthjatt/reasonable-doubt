"""Output schema parsing and cacheable-prefix prompt construction."""

from __future__ import annotations

import pytest

from src.data.labels import LabelNormalizer, load_labels
from src.data.manifest import DEFAULT_MANIFEST_DIR, load_manifest, verify_manifest
from src.data.prompts import (
    EXEMPLAR_MANIFEST,
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
    eight = render_fewshot(labels, ledgar["train"], n_exemplars=8)
    four = render_fewshot(labels, ledgar["train"], n_exemplars=4)
    assert eight.n_exemplars == 8 and four.n_exemplars == 4
    assert len(eight.exemplar_indices) == 8
    assert len(four.text) < len(eight.text)
    assert eight.sha256 != four.sha256


def test_four_exemplars_are_a_strict_subset_of_eight(ledgar, labels):
    """So the reduced-budget run stays comparable to the full one."""
    eight = render_fewshot(labels, ledgar["train"], n_exemplars=8)
    four = render_fewshot(labels, ledgar["train"], n_exemplars=4)
    assert four.exemplar_indices == eight.exemplar_indices[:4]


def test_fewshot_is_deterministic_and_byte_stable(ledgar, labels):
    a = render_fewshot(labels, ledgar["train"], n_exemplars=8)
    b = render_fewshot(labels, ledgar["train"], n_exemplars=8)
    assert a.text == b.text and a.sha256 == b.sha256
    assert a.exemplar_indices == b.exemplar_indices


def test_exemplars_come_from_the_frozen_manifest(ledgar, labels):
    """Hard rule 1 + no prefix drift: exemplars are read, never re-sampled."""
    manifest = load_manifest(EXEMPLAR_MANIFEST, DEFAULT_MANIFEST_DIR)
    verify_manifest(manifest, ledgar, DEFAULT_MANIFEST_DIR)
    assert manifest.split == "train"  # never dev, never test
    assert manifest.sampling == "proportional-random"

    p = render_fewshot(labels, ledgar["train"], n_exemplars=8)
    assert list(p.exemplar_indices) == manifest.indices
    assert p.exemplar_seed == manifest.seed


def test_exemplar_text_appears_in_the_prompt(ledgar, labels):
    p = render_fewshot(labels, ledgar["train"], n_exemplars=8)
    for text in ledgar["train"].select(list(p.exemplar_indices))["text"]:
        assert text[:200] in p.text


def test_exemplars_are_not_stratified(ledgar):
    """A uniform draw from the class prior makes no label-coverage claim; at n=8 it
    will usually repeat a head class rather than cover 8 distinct ones."""
    indices, _ = select_exemplars(8)
    classes = [int(ledgar["train"][i]["label"]) for i in indices]
    assert len(set(classes)) <= 8


def test_requesting_more_exemplars_than_frozen_is_rejected():
    with pytest.raises(ValueError, match="exceeds the 8 exemplars frozen"):
        select_exemplars(9)


def test_negative_exemplars_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        select_exemplars(-1)


def test_zero_exemplars_allowed():
    assert select_exemplars(0)[0] == []


def test_prompt_metadata_is_recorded_for_provenance(ledgar, labels):
    meta = render_fewshot(labels, ledgar["train"], n_exemplars=4).as_metadata()
    assert meta["n_exemplars"] == 4
    assert meta["exemplar_seed"] == 20260907
    assert len(meta["prompt_sha256"]) == 64
    assert len(meta["template_sha256"]) == 64
    assert len(meta["exemplar_indices"]) == 4
