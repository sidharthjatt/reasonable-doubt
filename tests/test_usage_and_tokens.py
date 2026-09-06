"""Usage ingestion (rule 10) and token counting (rules 9 and 11)."""

from __future__ import annotations

import json

import pytest

from src.api.cost import compute_cost
from src.api.tokens import TokenCountUnavailable, TokenCounter
from src.api.usage import (
    ANTHROPIC_USAGE_FIELDS,
    Usage,
    UsageFieldMissing,
    UsageNotCostable,
    parse_usage,
)

FULL = {
    "input_tokens": 120,
    "cache_creation_input_tokens": 2000,
    "cache_read_input_tokens": 4000,
    "output_tokens": 25,
}


def test_parses_four_independent_fields():
    u = parse_usage(FULL, model="claude-sonnet-5")
    assert u.input_tokens == 120
    assert u.cache_creation_input_tokens == 2000
    assert u.cache_read_input_tokens == 4000
    assert u.output_tokens == 25


@pytest.mark.parametrize("field", ANTHROPIC_USAGE_FIELDS)
def test_each_missing_field_raises_individually(field):
    """Hard rule 10: never default a missing field to zero."""
    block = {k: v for k, v in FULL.items() if k != field}
    with pytest.raises(UsageFieldMissing, match=field):
        parse_usage(block, model="claude-sonnet-5")


@pytest.mark.parametrize("field", ANTHROPIC_USAGE_FIELDS)
def test_null_field_raises(field):
    with pytest.raises(UsageFieldMissing, match="null"):
        parse_usage({**FULL, field: None}, model="claude-sonnet-5")


@pytest.mark.parametrize("bad", ["12", 12.5, True, -1])
def test_non_integer_or_negative_field_raises(bad):
    with pytest.raises(UsageFieldMissing):
        parse_usage({**FULL, "input_tokens": bad}, model="claude-sonnet-5")


def test_absent_usage_block_raises():
    with pytest.raises(UsageFieldMissing, match="no usage block"):
        parse_usage(None, model="claude-sonnet-5")


def test_the_ten_x_overbill_this_rule_prevents():
    """Summing input + cache_read as 'input' is the mistake rule 10 exists to stop."""
    u = parse_usage(FULL, model="claude-sonnet-5")
    correct = compute_cost("claude-sonnet-5", **u.as_cost_kwargs(), cache_ttl="1h", batch=True)
    naive = compute_cost(
        "claude-sonnet-5",
        input_tokens=u.input_tokens + u.cache_read_input_tokens,
        output_tokens=u.output_tokens,
        cache_write_tokens=u.cache_creation_input_tokens,
        cache_ttl="1h",
        batch=True,
    )
    assert naive > correct  # and plausible-looking, which is the danger


def test_sdk_style_object_is_accepted():
    class SDKUsage:
        input_tokens = 120
        cache_creation_input_tokens = 2000
        cache_read_input_tokens = 4000
        output_tokens = 25

    assert parse_usage(SDKUsage(), model="claude-sonnet-5").input_tokens == 120


def test_sdk_object_missing_a_field_still_raises():
    class Partial:
        input_tokens = 120
        output_tokens = 25

    with pytest.raises(UsageFieldMissing):
        parse_usage(Partial(), model="claude-sonnet-5")


# ------------------------------------------------------- providers without cache fields


def test_unreported_cache_fields_are_null_not_zero():
    u = parse_usage(
        {"input_tokens": 100, "output_tokens": 20},
        model="gemini-x",
        provider="gemini",
        cache_fields_reported=False,
    )
    assert u.cache_read_input_tokens is None
    assert u.cache_creation_input_tokens is None
    assert u.cache_fields_reported is False


def test_costing_unreported_cache_fields_raises():
    u = parse_usage(
        {"input_tokens": 100, "output_tokens": 20},
        model="gemini-x",
        provider="gemini",
        cache_fields_reported=False,
    )
    with pytest.raises(UsageNotCostable, match="null, not zero"):
        u.as_cost_kwargs()


def test_aggregation_keeps_fields_separate():
    a = parse_usage(FULL, model="claude-sonnet-5")
    b = parse_usage({**FULL, "input_tokens": 80}, model="claude-sonnet-5")
    total = a + b
    assert total.input_tokens == 200
    assert total.cache_read_input_tokens == 8000
    assert total.output_tokens == 50


def test_aggregating_an_unknown_yields_unknown_not_a_partial_sum():
    known = parse_usage(FULL, model="m")
    unknown = Usage(100, 20, None, None, model="m", provider="gemini")
    assert (known + unknown).cache_read_input_tokens is None


# -------------------------------------------------------------------- token counting


class FakeClient:
    def __init__(self, counts=None, fail=False):
        self.counts = counts or {}
        self.fail = fail
        self.calls = 0

    def count_tokens(self, *, model, system, messages):
        self.calls += 1
        if self.fail:
            raise ConnectionError("endpoint unreachable")
        return self.counts.get(model, 1234)


MSGS = [{"role": "user", "content": "a clause"}]


def test_counts_come_from_the_endpoint(tmp_path):
    tc = TokenCounter(FakeClient({"claude-sonnet-5": 950}), cache_root=tmp_path)
    assert tc.count_tokens("claude-sonnet-5", MSGS, system="sys") == 950


def test_counts_are_cached_per_model_and_content(tmp_path):
    client = FakeClient({"claude-sonnet-5": 950})
    tc = TokenCounter(client, cache_root=tmp_path)
    tc.count_tokens("claude-sonnet-5", MSGS, system="sys")
    tc.count_tokens("claude-sonnet-5", MSGS, system="sys")
    assert client.calls == 1
    assert tc.stats()["hits"] == 1


def test_different_models_do_not_share_a_count(tmp_path):
    """Hard rule 9: 4.7+ tokenizes differently; the same text has different counts."""
    client = FakeClient({"claude-sonnet-5": 950, "claude-haiku-4-5-20251001": 700})
    tc = TokenCounter(client, cache_root=tmp_path)
    assert tc.count_tokens("claude-sonnet-5", MSGS, system="sys") == 950
    assert tc.count_tokens("claude-haiku-4-5-20251001", MSGS, system="sys") == 700
    assert client.calls == 2


def test_different_content_does_not_share_a_count(tmp_path):
    client = FakeClient({"m": 10})
    tc = TokenCounter(client, cache_root=tmp_path)
    tc.count_tokens("m", MSGS, system="sys")
    tc.count_tokens("m", [{"role": "user", "content": "different"}], system="sys")
    assert client.calls == 2


def test_endpoint_failure_raises_and_never_estimates(tmp_path):
    """Hard rule 11: no local tokenizer fallback, no estimate."""
    tc = TokenCounter(FakeClient(fail=True), cache_root=tmp_path)
    with pytest.raises(TokenCountUnavailable, match="No fallback is permitted"):
        tc.count_tokens("claude-sonnet-5", MSGS, system="sys")


@pytest.mark.parametrize("bad", [0, -5, 12.5, True, None, "900"])
def test_implausible_count_raises(tmp_path, bad):
    class Weird(FakeClient):
        def count_tokens(self, *, model, system, messages):
            return bad

    tc = TokenCounter(Weird(), cache_root=tmp_path)
    with pytest.raises(TokenCountUnavailable):
        tc.count_tokens("m", MSGS, system="sys")


def test_missing_model_rejected(tmp_path):
    tc = TokenCounter(FakeClient(), cache_root=tmp_path)
    with pytest.raises(ValueError, match="per-model"):
        tc.count_tokens("", MSGS, system="sys")


def test_tampered_cached_count_for_the_wrong_model_raises(tmp_path):
    client = FakeClient({"claude-sonnet-5": 950})
    tc = TokenCounter(client, cache_root=tmp_path)
    tc.count_tokens("claude-sonnet-5", MSGS, system="sys")

    path = next((tmp_path / "token_counts").rglob("*.json"))
    data = json.loads(path.read_text())
    data["model"] = "claude-haiku-4-5-20251001"
    path.write_text(json.dumps(data))

    with pytest.raises(TokenCountUnavailable, match="never shared across models"):
        tc.count_tokens("claude-sonnet-5", MSGS, system="sys")
