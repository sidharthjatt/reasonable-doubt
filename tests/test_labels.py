"""Label normalizer: strict matching, no fuzzy rescue."""

import pytest

from src.data.labels import (
    DEFAULT_LABELS_PATH,
    FormatFailureCounter,
    LabelNormalizer,
    load_labels,
    normalize_key,
)

LABELS = [
    "Adjustments",
    "Anti-Corruption Laws",
    "Governing Laws",
    "Waivers",
    "No Waivers",
    "Waiver Of Jury Trials",
]


@pytest.fixture
def norm():
    return LabelNormalizer(LABELS)


def test_exact_match(norm):
    for label in LABELS:
        assert norm.normalize(label) == label


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("governing laws", "Governing Laws"),
        ("GOVERNING LAWS", "Governing Laws"),
        ("GoVeRnInG lAwS", "Governing Laws"),
        ("  Governing Laws  ", "Governing Laws"),
        ("\n\tGoverning Laws\n", "Governing Laws"),
        ("Governing   Laws", "Governing Laws"),
        ('"Governing Laws"', "Governing Laws"),
        ("'Governing Laws'", "Governing Laws"),
        ("Governing Laws.", "Governing Laws"),
        ("Governing Laws,", "Governing Laws"),
        ("**Governing Laws**", "Governing Laws"),
        ("(Governing Laws)", "Governing Laws"),
        ("anti-corruption laws", "Anti-Corruption Laws"),
    ],
)
def test_case_whitespace_and_trivial_punctuation(norm, raw, expected):
    assert norm.normalize(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Governing Law",          # singular — a different string, not a near-miss to fix
        "Anti Corruption Laws",   # hyphen removed; interior punctuation is significant
        "Choice of Law",          # plausible synonym; still not a canonical label
        "Governing Laws clause",  # extra interior words
        "Jury Trials",            # substring of a real label
        "Waiver",                 # prefix of a real label
        "label: Governing Laws",  # key included
        "Adjustment",             # singular of a real label
        "",
        "   ",
        "42",
        None,
    ],
)
def test_non_labels_return_none_and_are_not_coerced(norm, raw):
    assert norm.normalize(raw) is None


def test_ambiguity_is_not_resolved_by_guessing(norm):
    """'Waivers' must not absorb 'No Waivers' or 'Waiver Of Jury Trials'."""
    assert norm.normalize("waivers") == "Waivers"
    assert norm.normalize("no waivers") == "No Waivers"
    assert norm.normalize("waiver of jury trials") == "Waiver Of Jury Trials"


def test_contains_operator(norm):
    assert "governing laws" in norm
    assert "Governing Law" not in norm
    assert 42 not in norm


def test_colliding_labels_rejected_at_construction():
    with pytest.raises(ValueError, match="collide"):
        LabelNormalizer(["Governing Laws", "governing laws.", "Waivers"])


def test_normalize_key_preserves_interior_punctuation():
    assert normalize_key("Anti-Corruption Laws") != normalize_key("Anti Corruption Laws")


def test_committed_labels_file_is_the_100_class_space():
    labels = load_labels(DEFAULT_LABELS_PATH)
    assert len(labels) == 100
    assert len(set(labels)) == 100
    LabelNormalizer(labels)  # no collisions across the real label space


def test_format_failure_counter():
    c = FormatFailureCounter()
    assert c.failure_rate == 0.0  # empty is not a failure

    norm = LabelNormalizer(LABELS)
    for raw in ["Governing Laws", "waivers", "Governing Law", "banana"]:
        c.record(raw, norm.normalize(raw))

    assert (c.total, c.matched, c.unmatched) == (4, 2, 2)
    assert c.failure_rate == 0.5
    assert c.samples == ["Governing Law", "banana"]
    assert c.as_dict()["format_failure_rate"] == 0.5


def test_format_failure_counter_passes_value_through():
    c = FormatFailureCounter()
    assert c.record("x", "Waivers") == "Waivers"
    assert c.record("x", None) is None


def test_format_failure_counter_samples_are_bounded():
    c = FormatFailureCounter(max_samples=2)
    for _ in range(10):
        c.record("junk", None)
    assert c.unmatched == 10
    assert len(c.samples) == 2
