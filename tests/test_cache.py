"""Cache key correctness. A wrong key here produces cheap, plausible, wrong numbers."""

from __future__ import annotations

import json

import pytest

from src.api.cache import (
    CACHE_FORMAT_VERSION,
    CacheCollisionError,
    CacheKey,
    ResponseCache,
    content_sha256,
    params_sha256,
)

MSGS = [{"role": "user", "content": "This Agreement is governed by Delaware law."}]
SYS = "You are a legal contract analyst."
PARAMS = {"temperature": 0.0, "max_tokens": 64}


def key(**over):
    base = dict(provider="anthropic", model_id="claude-sonnet-5", system=SYS,
                messages=MSGS, params=PARAMS)
    base.update(over)
    provider = base.pop("provider")
    model = base.pop("model_id")
    return CacheKey.build(provider, model, **base)


# ------------------------------------------------------------------ key identity


def test_identical_requests_share_a_key():
    assert key() == key()
    assert key().digest == key().digest


def test_key_changes_when_the_model_changes():
    """Hard rule 9: Claude 4.7+ tokenizes differently, so one model's cached response
    and token counts must never be served for another model's request."""
    a, b = key(), key(model_id="claude-haiku-4-5-20251001")
    assert a.digest != b.digest
    assert a.prompt_sha256 == b.prompt_sha256  # same prompt, different key


def test_key_changes_when_the_provider_changes():
    a, b = key(), key(provider="groq")
    assert a.digest != b.digest


@pytest.mark.parametrize(
    "changed",
    [
        {"temperature": 1.0},
        {"max_tokens": 128},
        {"top_p": 0.9},
        {"top_k": 40},
        {"stop_sequences": ["\n\n"]},
        {"seed": 7},
    ],
)
def test_key_changes_when_decoding_params_change(changed):
    assert key().digest != key(params={**PARAMS, **changed}).digest


def test_key_changes_when_the_prompt_changes():
    other = [{"role": "user", "content": "This Agreement is governed by New York law."}]
    assert key().digest != key(messages=other).digest


def test_key_changes_when_the_system_prompt_changes():
    assert key().digest != key(system=SYS + " Be concise.").digest


def test_key_changes_when_message_order_changes():
    two = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert key(messages=two).digest != key(messages=list(reversed(two))).digest


def test_non_semantic_params_do_not_change_the_key():
    """Metadata must not fragment the cache and cause a needless re-call (rule 3)."""
    assert key().digest == key(params={**PARAMS, "metadata": {"user_id": "x"}}).digest


def test_absent_param_differs_from_explicitly_set_param():
    assert params_sha256({"temperature": 0.0}) != params_sha256({})


def test_params_hash_is_order_insensitive():
    assert params_sha256({"temperature": 0.0, "max_tokens": 64}) == params_sha256(
        {"max_tokens": 64, "temperature": 0.0}
    )


def test_provider_and_model_are_required():
    for bad in [("", "m"), ("anthropic", "")]:
        with pytest.raises(ValueError, match="required"):
            CacheKey.build(*bad, system=SYS, messages=MSGS, params=PARAMS)


def test_content_hash_covers_system_and_messages():
    assert content_sha256(None, MSGS) != content_sha256(SYS, MSGS)


# ---------------------------------------------------------------------- storage


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path)


def test_miss_then_hit(cache):
    k = key()
    assert cache.get_or_none(k) is None
    cache.put(k, {"content": "x"}, usage={"input_tokens": 1}, request_params=PARAMS)
    entry = cache.get_or_none(k)
    assert entry is not None
    assert entry.response == {"content": "x"}
    assert entry.usage == {"input_tokens": 1}
    assert entry.key == k


def test_stats_track_hits_misses_and_entries(cache):
    k = key()
    cache.get_or_none(k)
    cache.put(k, {"content": "x"})
    cache.get_or_none(k)
    cache.get_or_none(key(model_id="claude-opus-5"))

    s = cache.cache_stats()
    assert (s.hits, s.misses, s.writes, s.entries) == (1, 2, 1, 1)
    assert s.hit_rate == pytest.approx(1 / 3)


def test_rewriting_identical_content_is_a_noop(cache):
    k = key()
    cache.put(k, {"content": "x"}, usage={"input_tokens": 1}, request_params=PARAMS)
    cache.put(k, {"content": "x"}, usage={"input_tokens": 1}, request_params=PARAMS)
    assert cache.cache_stats().duplicate_writes == 1
    assert cache.cache_stats().entries == 1


def test_collision_with_differing_content_raises(cache):
    k = key()
    cache.put(k, {"content": "x"})
    with pytest.raises(CacheCollisionError, match="DIFFERENT content"):
        cache.put(k, {"content": "y"})


def test_collision_on_differing_usage_raises(cache):
    """Same text but different token counts means the key is missing something."""
    k = key()
    cache.put(k, {"content": "x"}, usage={"input_tokens": 100})
    with pytest.raises(CacheCollisionError):
        cache.put(k, {"content": "x"}, usage={"input_tokens": 900})


def test_entry_records_full_provenance(cache):
    k = key()
    cache.put(k, {"content": "x"}, usage={"input_tokens": 1}, request_params=PARAMS)
    data = json.loads(cache.path_for(k).read_text())
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-sonnet-5"
    assert data["format_version"] == CACHE_FORMAT_VERSION
    assert data["request_params"] == PARAMS
    assert data["timestamp_utc"].endswith("+00:00")
    assert set(data["key"]) == {"provider", "model_id", "prompt_sha256", "params_sha256"}


def test_unknown_format_version_raises_rather_than_missing(cache):
    """Hard rule 11: an unreadable cache must not degrade into a silent re-call."""
    k = key()
    cache.put(k, {"content": "x"})
    path = cache.path_for(k)
    data = json.loads(path.read_text())
    data["format_version"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(CacheCollisionError, match="format_version"):
        cache.get_or_none(k)


def test_two_models_coexist_without_clobbering(cache):
    """The scenario a prompt-only key would silently corrupt."""
    sonnet, haiku = key(), key(model_id="claude-haiku-4-5-20251001")
    cache.put(sonnet, {"content": "sonnet"}, usage={"input_tokens": 900})
    cache.put(haiku, {"content": "haiku"}, usage={"input_tokens": 700})
    assert cache.get_or_none(sonnet).response == {"content": "sonnet"}
    assert cache.get_or_none(haiku).response == {"content": "haiku"}
    assert cache.get_or_none(haiku).usage["input_tokens"] == 700
