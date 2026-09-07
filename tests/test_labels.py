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


# ---------------------------------------------------------------------------
# Realistic messy model output.
#
# Rung 0 produced 98 verbatim matches and 0 normalizations, so the normalizer went
# completely unexercised against real output. Claude may not be so tidy. These pin
# the boundary between "trivially reformatted, accept" and "not a label, reject".
# ---------------------------------------------------------------------------

MESSY_BUT_VALID = [
    ("`Governing Laws`", "backticks"),
    ("**Governing Laws**", "markdown bold"),
    ("**`Governing Laws`**", "bold + backticks"),
    ("_Governing Laws_", "markdown italic"),
    ("'Governing Laws'", "single quotes"),
    ('"Governing Laws"', "double quotes"),
    ("“Governing Laws”", "smart double quotes"),
    ("‘Governing Laws’", "smart single quotes"),
    ("Governing Laws.", "trailing period"),
    ("Governing Laws,", "trailing comma"),
    ("Governing Laws;", "trailing semicolon"),
    ("Governing Laws:", "trailing colon"),
    ("Governing Laws!", "trailing exclamation"),
    ("Governing Laws?", "trailing question mark"),
    ('"Governing Laws".', "quotes then period"),
    ("(Governing Laws)", "parentheses"),
    ("[Governing Laws]", "square brackets"),
    ("{Governing Laws}", "braces"),
    ("<Governing Laws>", "angle brackets"),
    ("   Governing Laws   ", "surrounding spaces"),
    ("\n\tGoverning Laws\n\n", "surrounding newlines and tabs"),
    ("Governing   Laws", "collapsed inner whitespace"),
    ("Governing\tLaws", "inner tab"),
    ("Governing\nLaws", "inner newline"),
    ("governing laws", "all lower"),
    ("GOVERNING LAWS", "all upper"),
    ("gOvErNiNg LaWs", "mixed case"),
    ("Ｇoverning Laws", "NFKC: fullwidth G"),
]


@pytest.mark.parametrize("raw, why", MESSY_BUT_VALID, ids=[w for _, w in MESSY_BUT_VALID])
def test_messy_but_recoverable_output_normalizes(norm, raw, why):
    assert norm.normalize(raw) == "Governing Laws", why


NOT_A_LABEL = [
    ("The label is Governing Laws.", "label nested in prose"),
    ("This clause is about Governing Laws", "prose prefix"),
    ("Governing Laws clause", "trailing noise word"),
    ("Governing Laws (choice of law)", "parenthetical gloss"),
    ("label: Governing Laws", "key included"),
    ("Answer: Governing Laws", "answer prefix"),
    ("1. Governing Laws", "numbered list item"),
    ("- Governing Laws", "bullet list item"),
    ("Governing Law", "singular"),
    ("Governing-Laws", "hyphen substituted for space"),
    ("GoverningLaws", "space removed"),
    ("Choice of Law", "plausible synonym"),
    ("Applicable Law", "near-miss of a different real label"),
    ("Adjustment", "singular of a real label"),
    ("Anti Corruption Laws", "interior hyphen removed"),
    ("Jury Trials", "substring of a real label"),
    ("Waiver", "prefix of a real label"),
    ("Governing Laws / Venues", "two labels combined"),
    ("Governing Laws\nVenues", "two labels on separate lines"),
    ("", "empty"),
    ("   ", "whitespace only"),
    ("...", "punctuation only"),
    ("null", "literal null"),
    ("N/A", "not applicable"),
    ("I'm not sure", "refusal"),
    ("42", "number"),
]


@pytest.mark.parametrize("raw, why", NOT_A_LABEL, ids=[w for _, w in NOT_A_LABEL])
def test_output_that_is_not_a_label_returns_none(norm, raw, why):
    """These are FORMAT FAILURES to be counted, not near-misses to be rescued.

    Note the deliberate strictness: a bullet or numbered-list prefix is rejected. A
    model that lists rather than answers has not followed the instruction, and we
    want that in the format-failure rate rather than silently repaired.
    """
    assert norm.normalize(raw) is None, why


def test_messy_variants_never_cross_map_between_real_labels():
    """The dangerous failure: a mutation of label A resolving to label B.

    Run over the real 100-class space, not the toy fixture.
    """
    labels = load_labels(DEFAULT_LABELS_PATH)
    real = LabelNormalizer(labels)
    mutations = [
        lambda s: s.lower(),
        lambda s: s.upper(),
        lambda s: f"  {s}  ",
        lambda s: f"`{s}`",
        lambda s: f"**{s}**",
        lambda s: f'"{s}."',
        lambda s: f"({s})",
        lambda s: s.replace(" ", "  "),
        lambda s: f"\n{s}\n",
    ]
    for label in labels:
        for mutate in mutations:
            assert real.normalize(mutate(label)) == label


def test_every_real_label_survives_a_round_trip():
    labels = load_labels(DEFAULT_LABELS_PATH)
    real = LabelNormalizer(labels)
    assert [real.normalize(x) for x in labels] == labels


def test_failure_counter_reflects_realistic_mixed_output(norm):
    """A realistic batch: some clean, some reformatted, some genuinely wrong."""
    counter = FormatFailureCounter()
    outputs = [
        "Governing Laws",          # clean
        "`Governing Laws`",        # recoverable
        "**Waivers**",             # recoverable
        "The label is Waivers",    # format failure
        "Choice of Law",           # format failure
    ]
    for raw in outputs:
        counter.record(raw, norm.normalize(raw))

    assert counter.total == 5
    assert counter.matched == 3
    assert counter.unmatched == 2
    assert counter.failure_rate == pytest.approx(0.4)
    assert counter.samples == ["The label is Waivers", "Choice of Law"]
