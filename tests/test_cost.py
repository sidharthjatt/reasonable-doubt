"""Tests for src/api/cost.py, anchored on Anthropic's own published worked example."""

import pytest

from src.api.cost import (
    UnknownModelError,
    compute_cost,
    cost_breakdown,
    effective_rates,
    load_rate_card,
)

# Costs are exact multiples of rates/1e6, so float error is far below a cent.
CENT = 1e-9


def test_docs_worked_example_opus5_with_prompt_caching():
    """Anthropic's published Opus 5 example, prompt caching active.

      10,000 uncached input tokens -> $0.05
      40,000 cache read tokens     -> $0.02
      15,000 output tokens         -> $0.375
      token subtotal               -> $0.445

    The docs' $0.525 headline adds an $0.08 Managed Agents session-runtime charge,
    which does not apply to us; this function prices tokens only.
    """
    parts = cost_breakdown(
        "claude-opus-5",
        input_tokens=10_000,
        output_tokens=15_000,
        cache_read_tokens=40_000,
    )
    assert parts["input"] == pytest.approx(0.05, abs=CENT)
    assert parts["cache_read"] == pytest.approx(0.02, abs=CENT)
    assert parts["output"] == pytest.approx(0.375, abs=CENT)
    assert parts["cache_write"] == 0.0
    assert parts["total"] == pytest.approx(0.445, abs=CENT)

    assert compute_cost(
        "claude-opus-5",
        input_tokens=10_000,
        output_tokens=15_000,
        cache_read_tokens=40_000,
    ) == pytest.approx(0.445, abs=CENT)


def test_sonnet5_batch_input_is_half_base():
    """The docs' batch table lists Sonnet 5 batch input at $1/MTok = 0.5x the $2 base."""
    base = effective_rates("claude-sonnet-5", batch=False)["input"] * 1e6
    batched = effective_rates("claude-sonnet-5", batch=True)["input"] * 1e6

    assert base == pytest.approx(2.0, abs=CENT)
    assert batched == pytest.approx(1.0, abs=CENT)
    assert batched == pytest.approx(0.5 * base, abs=CENT)

    assert compute_cost(
        "claude-sonnet-5", input_tokens=1_000_000, batch=True
    ) == pytest.approx(1.0, abs=CENT)


def test_batch_and_cache_multipliers_stack():
    """A cache-read token in a batch bills at 0.5 * 0.1 * base_input."""
    per_mtok = effective_rates("claude-sonnet-5", batch=True)["cache_read"] * 1e6
    assert per_mtok == pytest.approx(0.5 * 0.1 * 2.0, abs=CENT)


@pytest.mark.parametrize(
    "ttl, expected_multiplier",
    [("5m", 1.25), ("1h", 2.0)],
)
def test_cache_write_multipliers(ttl, expected_multiplier):
    per_mtok = effective_rates("claude-haiku-4-5-20251001", cache_ttl=ttl)["cache_write"] * 1e6
    assert per_mtok == pytest.approx(expected_multiplier * 1.0, abs=CENT)


def test_haiku_and_sonnet_base_rates():
    haiku = effective_rates("haiku-4.5")
    sonnet = effective_rates("sonnet-5")
    assert (haiku["input"] * 1e6, haiku["output"] * 1e6) == pytest.approx((1.0, 5.0), abs=CENT)
    # $2/$10 is the standard price; the planned $3/$15 increase does not occur.
    assert (sonnet["input"] * 1e6, sonnet["output"] * 1e6) == pytest.approx((2.0, 10.0), abs=CENT)


def test_alias_and_id_agree():
    kwargs = dict(input_tokens=1000, output_tokens=500)
    assert compute_cost("claude-opus-5", **kwargs) == compute_cost("opus-5", **kwargs)


def test_zero_tokens_is_free():
    assert compute_cost("claude-sonnet-5") == 0.0


def test_cache_ttl_is_irrelevant_without_cache_writes():
    kwargs = dict(input_tokens=1000, output_tokens=100, cache_read_tokens=5000)
    assert compute_cost("claude-sonnet-5", cache_ttl="5m", **kwargs) == compute_cost(
        "claude-sonnet-5", cache_ttl="1h", **kwargs
    )


def test_unknown_model_rejected():
    with pytest.raises(UnknownModelError):
        compute_cost("gpt-4o", input_tokens=100)


def test_unknown_cache_ttl_rejected():
    with pytest.raises(ValueError):
        compute_cost("claude-sonnet-5", input_tokens=100, cache_ttl="10m")


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        compute_cost("claude-sonnet-5", input_tokens=-1)


def test_batch_rejected_for_model_without_batch_support():
    with pytest.raises(ValueError):
        compute_cost("PLACEHOLDER-groq-model", input_tokens=100, batch=True)


def test_no_hardcoded_rates_module_reads_the_card():
    """Hard rule 5: prices come from the card, so patching the card moves the price."""
    card = load_rate_card()
    doubled = {
        **card,
        "providers": {
            **card["providers"],
            "anthropic": {
                **card["providers"]["anthropic"],
                "models": {
                    **card["providers"]["anthropic"]["models"],
                    "claude-sonnet-5": {
                        **card["providers"]["anthropic"]["models"]["claude-sonnet-5"],
                        "input": 4.0,
                    },
                },
            },
        },
    }
    assert compute_cost(
        "claude-sonnet-5", input_tokens=1_000_000, rate_card=doubled
    ) == pytest.approx(4.0, abs=CENT)
