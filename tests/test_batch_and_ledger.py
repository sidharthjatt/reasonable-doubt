"""Batch join semantics, budget enforcement, and an end-to-end offline dry run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.batch import (
    CACHE_TTL,
    MAX_BATCH_REQUESTS,
    BatchClient,
    BatchJoinError,
    BatchState,
    build_batch_requests,
    custom_id_for,
    join_on_custom_id,
    parse_custom_id,
)
from src.api.cost import compute_cost, load_rate_card
from src.api.ledger import (
    BudgetExceeded,
    LedgerEstimateError,
    SpendLedger,
    estimate_run_cost,
    estimate_run_cost_bracket,
)
from src.api.providers import get_provider
from src.api.usage import Usage, parse_usage

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def fixture_batch():
    return json.loads((FIXTURES / "batch_responses.json").read_text())


# ------------------------------------------------------------------- custom_id


def test_custom_id_is_derived_from_the_manifest_row():
    assert custom_id_for("test_3000", 45) == "test_3000-000045"
    assert parse_custom_id("test_3000-000045") == ("test_3000", 45)


def test_custom_id_round_trips_for_every_row_of_a_full_manifest():
    for i in (0, 1, 999, 9999, 59_999):
        assert parse_custom_id(custom_id_for("dev_2000", i))[1] == i


def test_custom_id_charset_is_valid():
    assert custom_id_for("test_3000", 0).replace("-", "").replace("_", "").isalnum()


# ------------------------------------------------------------- request building


def _requests(n=3, **kw):
    return build_batch_requests(
        "test_3000",
        [12, 45, 87][:n],
        ["clause a", "clause b", "clause c"][:n],
        model="claude-sonnet-5",
        system_prompt="SYSTEM PREFIX",
        max_tokens=64,
        **kw,
    )


def test_cache_ttl_is_one_hour_not_the_five_minute_default():
    """Batch cache hits are best-effort; 5m would expire before out-of-order
    requests get to read the prefix."""
    assert CACHE_TTL == "1h"
    block = _requests()[0].params["system"][0]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_system_prompt_is_one_contiguous_cacheable_block():
    for req in _requests():
        system = req.params["system"]
        assert len(system) == 1  # single block => clean cache breakpoint
        assert system[0]["text"] == "SYSTEM PREFIX"
        assert "clause" not in system[0]["text"]  # clause never enters the prefix


def test_only_the_clause_varies_between_requests():
    reqs = _requests()
    prefixes = {json.dumps(r.params["system"], sort_keys=True) for r in reqs}
    assert len(prefixes) == 1
    assert {r.params["messages"][0]["content"] for r in reqs} == {
        "clause a", "clause b", "clause c"
    }


def test_mismatched_rows_and_texts_rejected():
    with pytest.raises(ValueError, match="clause texts"):
        build_batch_requests("m", [1, 2], ["only one"], model="m",
                             system_prompt="s", max_tokens=8)


def test_batch_size_limit_enforced():
    n = MAX_BATCH_REQUESTS + 1
    with pytest.raises(ValueError, match="batch limit"):
        build_batch_requests("m", list(range(n)), ["x"] * n, model="m",
                             system_prompt="s", max_tokens=8)


def test_duplicate_row_indices_rejected():
    with pytest.raises(ValueError, match="duplicate custom_id"):
        build_batch_requests("m", [5, 5], ["a", "b"], model="m",
                             system_prompt="s", max_tokens=8)


def test_bad_cache_ttl_rejected():
    with pytest.raises(ValueError, match="cache_ttl"):
        _requests(cache_ttl="30m")


# ------------------------------------------------------------------ joining


def test_join_is_by_custom_id_not_position(fixture_batch):
    """Results come back in ANY ORDER; position carries no information."""
    submitted = [custom_id_for("test_3000", i) for i in fixture_batch["submitted_row_indices"]]
    returned_order = [r["custom_id"] for r in fixture_batch["results"]]
    assert returned_order != submitted  # fixture is deliberately shuffled

    joined = join_on_custom_id(submitted, fixture_batch["results"])
    by_row = joined.row_indices()
    assert by_row[12]["content"][0]["text"].startswith('{"label": "Governing Laws"')
    assert by_row[45]["content"][0]["text"].startswith('{"label": "Severability"')
    assert by_row[87]["content"][0]["text"].startswith('{"label": "Notices"')


def test_join_is_stable_under_any_shuffle(fixture_batch):
    import itertools

    submitted = [custom_id_for("test_3000", i) for i in fixture_batch["submitted_row_indices"]]
    reference = join_on_custom_id(submitted, fixture_batch["results"]).succeeded
    for perm in itertools.permutations(fixture_batch["results"]):
        assert join_on_custom_id(submitted, list(perm)).succeeded == reference


def test_partial_failure_is_surfaced_not_fatal(fixture_batch):
    submitted = [custom_id_for("test_3000", i) for i in fixture_batch["submitted_row_indices"]]
    joined = join_on_custom_id(submitted, fixture_batch["results"])
    assert joined.n_succeeded == 3
    assert joined.n_failed == 1
    assert joined.failed["test_3000-000091"]["error"]["type"] == "overloaded_error"


def test_missing_custom_id_raises(fixture_batch):
    submitted = [custom_id_for("test_3000", i) for i in fixture_batch["submitted_row_indices"]]
    truncated = fixture_batch["results"][:-1]
    with pytest.raises(BatchJoinError, match="did not come back"):
        join_on_custom_id(submitted, truncated)


def test_unexpected_custom_id_raises(fixture_batch):
    submitted = [custom_id_for("test_3000", i) for i in fixture_batch["submitted_row_indices"]]
    extra = fixture_batch["results"] + [
        {"custom_id": "test_3000-999999", "result": {"type": "succeeded", "message": {}}}
    ]
    with pytest.raises(BatchJoinError, match="UNEXPECTED"):
        join_on_custom_id(submitted, extra)


def test_duplicate_custom_id_in_results_raises(fixture_batch):
    submitted = [custom_id_for("test_3000", i) for i in fixture_batch["submitted_row_indices"]]
    dupes = fixture_batch["results"] + [fixture_batch["results"][0]]
    with pytest.raises(BatchJoinError, match="duplicate"):
        join_on_custom_id(submitted, dupes)


def test_result_without_custom_id_raises():
    with pytest.raises(BatchJoinError, match="no custom_id"):
        join_on_custom_id(["a"], [{"result": {"type": "succeeded"}}])


# --------------------------------------------------------- submission / resume


class FakeAPI:
    def __init__(self):
        self.created = 0

    def create_batch(self, requests):
        self.created += 1
        return {"id": f"batch_{self.created:03d}"}

    def retrieve_batch(self, batch_id):
        return {"id": batch_id, "processing_status": "ended"}

    def batch_results(self, batch_id):
        return []


def test_batch_id_persisted_immediately_on_submit(tmp_path):
    client = BatchClient(FakeAPI(), state_dir=tmp_path)
    state = client.submit("run1", _requests(), model="claude-sonnet-5",
                          manifest_name="test_3000")
    on_disk = json.loads((tmp_path / "run1.json").read_text())
    assert on_disk["batch_id"] == state.batch_id
    assert on_disk["custom_ids"] == state.custom_ids
    assert on_disk["cache_ttl"] == "1h"


def test_resubmitting_the_same_run_id_does_not_double_spend(tmp_path):
    api = FakeAPI()
    client = BatchClient(api, state_dir=tmp_path)
    first = client.submit("run1", _requests(), model="m", manifest_name="test_3000")
    second = client.submit("run1", _requests(), model="m", manifest_name="test_3000")
    assert api.created == 1
    assert first.batch_id == second.batch_id


def test_resume_false_refuses_rather_than_resubmitting(tmp_path):
    client = BatchClient(FakeAPI(), state_dir=tmp_path)
    client.submit("run1", _requests(), model="m", manifest_name="test_3000")
    with pytest.raises(RuntimeError, match="double-spend"):
        client.submit("run1", _requests(), model="m", manifest_name="test_3000", resume=False)


def test_warmup_submits_a_small_batch_marked_as_such(tmp_path):
    client = BatchClient(FakeAPI(), state_dir=tmp_path)
    state = client.warmup_batch("warm1", _requests(), model="m",
                                manifest_name="test_3000", n=2)
    assert state.is_warmup is True
    assert state.n_requests == 2


def test_empty_batch_refused(tmp_path):
    client = BatchClient(FakeAPI(), state_dir=tmp_path)
    with pytest.raises(ValueError, match="empty batch"):
        client.submit("run1", [], model="m", manifest_name="test_3000")


def test_poll_backs_off_and_records_completion(tmp_path):
    class Slow(FakeAPI):
        def __init__(self):
            super().__init__()
            self.polls = 0

        def retrieve_batch(self, batch_id):
            self.polls += 1
            status = "in_progress" if self.polls < 3 else "ended"
            return {"id": batch_id, "processing_status": status}

    slept = []
    client = BatchClient(Slow(), state_dir=tmp_path, sleep=slept.append)
    state = client.submit("run1", _requests(), model="m", manifest_name="test_3000")
    state = client.poll(state, initial_delay=1.0, max_delay=8.0)

    assert state.status == "ended"
    assert state.completed_at_utc is not None
    assert len(slept) == 2 and slept[1] > slept[0]  # backoff grew


def test_poll_raises_on_terminal_failure(tmp_path):
    class Failed(FakeAPI):
        def retrieve_batch(self, batch_id):
            return {"id": batch_id, "processing_status": "expired"}

    client = BatchClient(Failed(), state_dir=tmp_path, sleep=lambda _: None)
    state = client.submit("run1", _requests(), model="m", manifest_name="test_3000")
    with pytest.raises(RuntimeError, match="expired"):
        client.poll(state)


# ------------------------------------------------------------------- ledger


@pytest.fixture
def ledger(tmp_path):
    return SpendLedger(tmp_path / "spend_ledger.jsonl")


def test_empty_ledger_has_full_budget(ledger):
    assert ledger.cumulative_usd() == 0.0
    assert ledger.hard_stop_usd == 15.0
    assert ledger.remaining_usd() == 15.0


def test_records_actual_from_the_four_usage_fields(ledger):
    u = parse_usage(
        {"input_tokens": 325, "cache_creation_input_tokens": 2000,
         "cache_read_input_tokens": 4000, "output_tokens": 77},
        model="claude-sonnet-5",
    )
    entry = ledger.record_actual("run1", u, provider="anthropic",
                                 model="claude-sonnet-5", batch=True, cache_ttl="1h")
    assert entry.cost_usd == pytest.approx(0.00511, abs=1e-9)
    assert entry.cache_read_input_tokens == 4000
    assert ledger.cumulative_usd() == pytest.approx(0.00511, abs=1e-9)


def test_ledger_is_append_only(ledger):
    u = parse_usage({"input_tokens": 100, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 0, "output_tokens": 10},
                    model="claude-sonnet-5")
    for _ in range(3):
        ledger.record_actual("r", u, provider="anthropic", model="claude-sonnet-5")
    assert len(ledger.path.read_text().strip().splitlines()) == 3
    assert len(ledger.entries()) == 3


def test_refuses_a_run_that_would_exceed_the_hard_stop(ledger):
    with pytest.raises(BudgetExceeded, match="REFUSING TO RUN"):
        ledger.assert_within_budget(15.01)


def test_refusal_accounts_for_already_recorded_spend(ledger):
    u = parse_usage({"input_tokens": 7_000_000, "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 0, "output_tokens": 0},
                    model="claude-sonnet-5")
    ledger.record_actual("r1", u, provider="anthropic", model="claude-sonnet-5", batch=False)
    assert ledger.cumulative_usd() == pytest.approx(14.0)
    ledger.assert_within_budget(1.0)  # exactly at the stop is permitted
    with pytest.raises(BudgetExceeded, match="Remaining budget"):
        ledger.assert_within_budget(1.01)


def test_estimate_rows_do_not_count_as_spend(ledger):
    est = estimate_run_cost(
        "run1", provider="anthropic", model="claude-sonnet-5", n_requests=10,
        system_tokens=2000, per_request_input_tokens=[100] * 10,
        expected_output_tokens=25, assume_cache_hits=False,
    )
    ledger.record_estimate(est)
    assert ledger.cumulative_usd() == 0.0
    assert len(ledger.entries()) == 1


def test_corrupt_ledger_line_raises_rather_than_under_reporting(ledger):
    ledger.path.write_text('{"not": "an entry"}\n')
    with pytest.raises(ValueError, match="unreadable entry"):
        ledger.cumulative_usd()


def test_cannot_cost_a_response_without_cache_fields(ledger):
    u = Usage(100, 20, None, None, model="gemini-x", provider="gemini")
    with pytest.raises(Exception, match="null, not zero"):
        ledger.record_actual("r", u, provider="gemini", model="claude-sonnet-5")


# --------------------------------------------------------------- estimation


def test_optimistic_and_pessimistic_bounds_bracket_the_truth():
    kw = dict(provider="anthropic", model="claude-sonnet-5", n_requests=100,
              system_tokens=2000, per_request_input_tokens=[100] * 100,
              expected_output_tokens=25)
    optimistic = estimate_run_cost("r", assume_cache_hits=True, **kw)
    pessimistic = estimate_run_cost("r", assume_cache_hits=False, **kw)
    assert pessimistic.estimated_usd > optimistic.estimated_usd
    assert optimistic.cache_read_tokens == 2000 * 99
    assert pessimistic.cache_read_tokens == 0


def test_estimate_renders_both_bounds_and_names_the_gating_one():
    est = estimate_run_cost_bracket("run1", provider="anthropic", model="claude-sonnet-5",
                                    n_requests=3, system_tokens=2000,
                                    per_request_input_tokens=[100, 110, 120],
                                    expected_output_tokens=25)
    text = est.render()
    assert "ESTIMATED COST" in text
    assert "optimistic" in text.lower()
    assert "PESSIMISTIC" in text
    assert "gates spend" in text
    assert "best-effort" in text.lower()


# --- the gating rule: spend is guarded by the number that can actually happen ---


def test_gate_uses_the_pessimistic_figure(ledger):
    est = estimate_run_cost_bracket(
        "r", provider="anthropic", model="claude-sonnet-5", n_requests=100,
        system_tokens=2000, per_request_input_tokens=[100] * 100,
        expected_output_tokens=25,
    )
    assert est.gating_usd > est.optimistic_usd
    assert ledger.assert_within_budget(est) == pytest.approx(est.gating_usd)


def test_run_that_fits_optimistically_but_breaches_pessimistically_is_refused(ledger):
    """The whole point: a guard on the optimistic figure is not a guard.

    A large cacheable prefix over many requests is cheap if every request reads the
    cache, and ruinous if none do. Batch cache hits are best-effort, so the expensive
    outcome is live and must be what gates.
    """
    est = estimate_run_cost_bracket(
        "r", provider="anthropic", model="claude-opus-5", n_requests=3000,
        system_tokens=2000, per_request_input_tokens=[50] * 3000,
        expected_output_tokens=20,
    )
    assert est.optimistic_usd < 15.0 < est.gating_usd  # fits one way, not the other
    # The optimistic AMOUNT would have passed a naive guard...
    ledger.assert_within_budget(est.optimistic_usd)
    # ...and the optimistic ESTIMATE object is refused outright, so it cannot be
    # substituted by accident.
    with pytest.raises(LedgerEstimateError):
        ledger.assert_within_budget(est.optimistic)
    with pytest.raises(BudgetExceeded):
        ledger.assert_within_budget(est)


def test_optimistic_estimate_cannot_be_used_to_gate(ledger):
    optimistic = estimate_run_cost(
        "r", provider="anthropic", model="claude-sonnet-5", n_requests=10,
        system_tokens=2000, per_request_input_tokens=[100] * 10,
        expected_output_tokens=25, assume_cache_hits=True,
    )
    assert optimistic.is_gating is False
    with pytest.raises(LedgerEstimateError, match="OPTIMISTIC"):
        ledger.assert_within_budget(optimistic)


def test_estimate_rejects_mismatched_token_counts():
    with pytest.raises(ValueError, match="per-request token counts"):
        estimate_run_cost("r", provider="anthropic", model="claude-sonnet-5",
                          n_requests=5, system_tokens=100,
                          per_request_input_tokens=[10, 10], expected_output_tokens=5)


# ------------------------------------------------------- end-to-end offline dry run


def test_end_to_end_dry_run_matches_a_hand_checked_cost(fixture_batch, tmp_path):
    """Fixture responses -> join -> usage -> cost -> ledger, with no network.

    Hand-checked, Sonnet 5, batch (0.5x), 1h TTL (2.0x write), cache read 0.1x:
        input       325 * $2/MTok * 0.5              = $0.000325
        cache write 2000 * $2/MTok * 2.0 * 0.5       = $0.004
        cache read  4000 * $2/MTok * 0.1 * 0.5       = $0.0004
        output       77 * $10/MTok * 0.5             = $0.000385
        total                                        = $0.005110
    """
    submitted = [custom_id_for("test_3000", i) for i in fixture_batch["submitted_row_indices"]]
    joined = join_on_custom_id(submitted, fixture_batch["results"])

    provider = get_provider("anthropic")
    total = None
    for body in joined.succeeded.values():
        u = provider.parse_usage(body["usage"], model=fixture_batch["model"])
        total = u if total is None else total + u

    assert total.input_tokens == 325
    assert total.cache_creation_input_tokens == 2000
    assert total.cache_read_input_tokens == 4000
    assert total.output_tokens == 77

    ledger = SpendLedger(tmp_path / "spend.jsonl")
    est = estimate_run_cost_bracket("dryrun", provider="anthropic",
                                    model=fixture_batch["model"],
                                    n_requests=3, system_tokens=2000,
                                    per_request_input_tokens=[120, 95, 110],
                                    expected_output_tokens=26)
    ledger.assert_within_budget(est)
    ledger.record_estimate(est.pessimistic)

    entry = ledger.record_actual(
        "dryrun", total, provider="anthropic", model=fixture_batch["model"],
        batch_id="batch_001", batch=True, cache_ttl="1h",
        estimated_usd=est.gating_usd, n_requests=3,
        notes="1 of 4 requests failed individually (overloaded_error)",
    )

    assert entry.cost_usd == pytest.approx(0.005110, abs=1e-9)
    assert ledger.cumulative_usd() == pytest.approx(0.005110, abs=1e-9)

    comparison = ledger.estimate_vs_actual()
    assert len(comparison) == 1
    assert comparison[0]["actual_usd"] == pytest.approx(0.005110, abs=1e-9)
    assert comparison[0]["estimated_usd"] == est.gating_usd
    assert comparison[0]["error_usd"] == pytest.approx(
        0.005110 - est.gating_usd, abs=1e-9
    )


def test_dry_run_never_touches_the_network(fixture_batch, monkeypatch):
    """Belt and braces: fail loudly if any socket is opened during the dry run."""
    import socket

    def forbidden(*a, **k):
        raise AssertionError("the offline dry run attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    submitted = [custom_id_for("test_3000", i) for i in fixture_batch["submitted_row_indices"]]
    joined = join_on_custom_id(submitted, fixture_batch["results"])
    provider = get_provider("anthropic")
    for body in joined.succeeded.values():
        provider.parse_usage(body["usage"], model=fixture_batch["model"])
    compute_cost("claude-sonnet-5", input_tokens=1, rate_card=load_rate_card())


# ------------------------------------------------------- --confirm spend gate


def _estimate(usd_scale=10):
    return estimate_run_cost_bracket(
        "run1", provider="anthropic", model="claude-sonnet-5", n_requests=usd_scale,
        system_tokens=2000, per_request_input_tokens=[100] * usd_scale,
        expected_output_tokens=25,
    )


def test_without_confirm_the_entrypoint_refuses_to_send(ledger, capsys):
    from src.api.ledger import require_confirmation

    with pytest.raises(SystemExit, match="NOT SENDING"):
        require_confirmation(_estimate(), ledger, confirm=False)
    assert "ESTIMATED COST" in capsys.readouterr().out


def test_with_confirm_the_entrypoint_proceeds(ledger, capsys):
    from src.api.ledger import require_confirmation

    projected = require_confirmation(_estimate(), ledger, confirm=True)
    assert projected == pytest.approx(_estimate().gating_usd)
    assert "CONFIRMED" in capsys.readouterr().out


def test_confirm_cannot_override_the_hard_stop(ledger, capsys):
    """--confirm is consent to spend, not permission to breach the budget."""
    from src.api.ledger import require_confirmation

    huge = estimate_run_cost_bracket(
        "big", provider="anthropic", model="claude-opus-5", n_requests=1,
        system_tokens=0, per_request_input_tokens=[10_000_000],
        expected_output_tokens=0, batch=False,
    )
    assert huge.gating_usd > 15.0
    with pytest.raises(BudgetExceeded):
        require_confirmation(huge, ledger, confirm=True)


def test_unrecognised_batch_status_raises_instead_of_polling_forever(tmp_path):
    """Hard rule 11: an unknown status must not be read as 'still running'. Silently
    polling would burn the 24h timeout and then misreport it as a TimeoutError."""
    class Renamed(FakeAPI):
        def retrieve_batch(self, batch_id):
            return {"id": batch_id, "processing_status": "finished"}  # API renamed it

    client = BatchClient(Renamed(), state_dir=tmp_path, sleep=lambda _: None)
    state = client.submit("run1", _requests(), model="m", manifest_name="test_3000")
    with pytest.raises(RuntimeError, match="UNRECOGNISED processing_status"):
        client.poll(state)


def test_missing_status_field_also_raises(tmp_path):
    class NoStatus(FakeAPI):
        def retrieve_batch(self, batch_id):
            return {"id": batch_id}

    client = BatchClient(NoStatus(), state_dir=tmp_path, sleep=lambda _: None)
    state = client.submit("run1", _requests(), model="m", manifest_name="test_3000")
    with pytest.raises(RuntimeError, match="UNRECOGNISED"):
        client.poll(state)


def test_in_progress_statuses_still_poll(tmp_path):
    class Working(FakeAPI):
        def __init__(self):
            super().__init__()
            self.n = 0

        def retrieve_batch(self, batch_id):
            self.n += 1
            return {"id": batch_id,
                    "processing_status": "in_progress" if self.n < 3 else "ended"}

    client = BatchClient(Working(), state_dir=tmp_path, sleep=lambda _: None)
    state = client.submit("run1", _requests(), model="m", manifest_name="test_3000")
    assert client.poll(state, initial_delay=0.01).status == "ended"
