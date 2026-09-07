"""Secrets must be blocked at the point of WRITE, not the point of display."""

from __future__ import annotations

import pytest

from src.api.cache import CacheKey, ResponseCache
from src.api.ledger import SpendLedger
from src.api.redaction import (
    REDACTED,
    SecretLeakError,
    assert_no_secrets,
    redact,
    secret_values,
)
from src.api.usage import parse_usage

FAKE_ANTHROPIC = "sk-ant-api03-" + "A" * 40
FAKE_GROQ = "gsk_" + "B" * 40
FAKE_HF = "hf_" + "C" * 34
FAKE_GEMINI = "AIza" + "D" * 32


@pytest.fixture
def env_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ)
    return {"ANTHROPIC_API_KEY": FAKE_ANTHROPIC, "GROQ_API_KEY": FAKE_GROQ}


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize("secret", [FAKE_ANTHROPIC, FAKE_GROQ, FAKE_HF, FAKE_GEMINI])
def test_known_key_shapes_are_detected_even_when_not_in_the_environment(secret):
    """Catches a key pasted into a prompt or echoed back by a provider."""
    with pytest.raises(SecretLeakError):
        assert_no_secrets({"text": f"my key is {secret}"}, context="test", env={})


def test_environment_value_is_detected_even_without_a_known_prefix(env_with_key):
    odd = "totally-unshaped-credential-value-1234"
    with pytest.raises(SecretLeakError):
        assert_no_secrets({"k": odd}, context="test", env={"MY_API_KEY": odd})


def test_detection_is_recursive_through_nested_structures():
    payload = {"a": [{"b": ("c", {"d": f"x {FAKE_GROQ} y"})}]}
    with pytest.raises(SecretLeakError):
        assert_no_secrets(payload, context="test", env={})


def test_detection_reaches_dict_keys_too():
    with pytest.raises(SecretLeakError):
        assert_no_secrets({FAKE_GROQ: "value"}, context="test", env={})


def test_error_message_does_not_reproduce_the_secret():
    """Reporting a leak must not itself leak."""
    with pytest.raises(SecretLeakError) as exc:
        assert_no_secrets({"k": FAKE_ANTHROPIC}, context="test", env={})
    assert FAKE_ANTHROPIC not in str(exc.value)
    assert "not reproduced here" in str(exc.value)


def test_clean_payload_passes():
    assert_no_secrets(
        {"response": "Governing Laws", "usage": {"input_tokens": 100}},
        context="test",
        env={},
    )


def test_short_env_values_are_not_treated_as_secrets():
    """An empty or placeholder env var must not redact ordinary text."""
    assert secret_values({"X_API_KEY": ""}) == []
    assert secret_values({"X_API_KEY": "todo"}) == []


# ------------------------------------------------------------------ redaction


def test_redact_masks_environment_values_and_known_shapes(env_with_key):
    text = f"Authorization: Bearer {FAKE_ANTHROPIC} and {FAKE_HF}"
    out = redact(text)
    assert FAKE_ANTHROPIC not in out
    assert FAKE_HF not in out
    assert out.count(REDACTED) == 2


def test_redact_is_a_noop_on_clean_text():
    assert redact("nothing secret here", env={}) == "nothing secret here"


# --------------------------------------------- point-of-write guards (the real test)


def test_cache_refuses_to_write_a_secret(tmp_path, env_with_key):
    """A provider echoing the key back must not land on disk."""
    cache = ResponseCache(tmp_path)
    key = CacheKey.build("anthropic", "claude-sonnet-5",
                         system="s", messages=[{"role": "user", "content": "c"}], params={})
    with pytest.raises(SecretLeakError, match="REFUSING TO WRITE"):
        cache.put(key, {"text": f"your key {FAKE_ANTHROPIC} is invalid"})

    # and nothing was written
    assert not cache.path_for(key).exists()
    assert cache.cache_stats().entries == 0


def test_cache_refuses_a_secret_hidden_in_request_params(tmp_path, env_with_key):
    cache = ResponseCache(tmp_path)
    key = CacheKey.build("groq", "llama", system="s", messages=[], params={})
    with pytest.raises(SecretLeakError):
        cache.put(key, "fine", request_params={"authorization": f"Bearer {FAKE_GROQ}"})
    assert not cache.path_for(key).exists()


def test_ledger_refuses_to_write_a_secret(tmp_path, env_with_key):
    """The ledger is version-controlled, so a leak here would be committed."""
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    u = parse_usage(
        {"input_tokens": 10, "cache_creation_input_tokens": 0,
         "cache_read_input_tokens": 0, "output_tokens": 5},
        model="claude-sonnet-5",
    )
    with pytest.raises(SecretLeakError, match="REFUSING TO WRITE"):
        ledger.record_actual("run1", u, provider="anthropic", model="claude-sonnet-5",
                             notes=f"debug: {FAKE_ANTHROPIC}")
    assert not ledger.path.exists()


def test_normal_cache_and_ledger_writes_still_work(tmp_path, env_with_key):
    """The guard must not block legitimate writes while a real key is in the env."""
    cache = ResponseCache(tmp_path)
    key = CacheKey.build("anthropic", "claude-sonnet-5", system="s", messages=[], params={})
    cache.put(key, {"text": '{"label": "Governing Laws", "confidence": 0.9}'})
    assert cache.get_or_none(key) is not None

    ledger = SpendLedger(tmp_path / "spend.jsonl")
    u = parse_usage(
        {"input_tokens": 10, "cache_creation_input_tokens": 0,
         "cache_read_input_tokens": 0, "output_tokens": 5},
        model="claude-sonnet-5",
    )
    ledger.record_actual("run1", u, provider="anthropic", model="claude-sonnet-5")
    assert len(ledger.entries()) == 1
