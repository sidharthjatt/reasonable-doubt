"""Provider adapters: shared cache/usage plumbing, divergent cache reporting."""

import pytest

from src.api.providers import PROVIDERS, get_provider
from src.api.usage import UsageFieldMissing, UsageNotCostable

MSGS = [{"role": "user", "content": "clause"}]


def test_known_providers():
    assert set(PROVIDERS) == {"anthropic", "gemini", "groq"}


def test_unknown_provider_rejected():
    with pytest.raises(KeyError, match="unknown provider"):
        get_provider("openai")


def test_anthropic_supports_batch_and_reports_cache_tokens():
    p = get_provider("anthropic")
    assert p.supports_batch and p.reports_cache_tokens and not p.is_free_tier


@pytest.mark.parametrize("name", ["gemini", "groq"])
def test_free_tiers_are_marked_and_do_not_report_cache_tokens(name):
    p = get_provider(name)
    assert p.is_free_tier and not p.reports_cache_tokens


def test_provider_is_part_of_the_cache_key():
    """The same model id under two providers must not share cached responses."""
    a = get_provider("anthropic").cache_key("m", system="s", messages=MSGS, params={})
    g = get_provider("groq").cache_key("m", system="s", messages=MSGS, params={})
    assert a.digest != g.digest
    assert a.prompt_sha256 == g.prompt_sha256


def test_free_tier_usage_is_null_not_zero_and_refuses_costing():
    u = get_provider("groq").parse_usage(
        {"input_tokens": 100, "output_tokens": 20}, model="llama-x"
    )
    assert u.cache_read_input_tokens is None
    with pytest.raises(UsageNotCostable):
        u.as_cost_kwargs()


def test_anthropic_still_requires_all_four_fields():
    with pytest.raises(UsageFieldMissing, match="cache_creation_input_tokens"):
        get_provider("anthropic").parse_usage(
            {"input_tokens": 100, "output_tokens": 20}, model="claude-sonnet-5"
        )
