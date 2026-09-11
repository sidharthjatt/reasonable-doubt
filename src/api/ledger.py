"""Append-only spend ledger and budget enforcement (hard rule 8).

Every dollar the project spends is recorded in ``results/spend_ledger.jsonl``, one JSON
object per line, appended and never rewritten. The ledger is the authority on
cumulative spend; :meth:`SpendLedger.assert_within_budget` refuses — by raising, never
by warning — any run that would push cumulative spend past ``budget.hard_stop_usd`` in
``configs/costs.yaml``.

Estimate-vs-actual
------------------
An estimate is recorded before submission and the actual is written alongside it after
the run, from the returned usage blocks. That record is itself a reported result: it
tells us how well our own cost model predicts reality, which is the question this
project exists to answer.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.api.cost import compute_cost, load_rate_card
from src.api.redaction import assert_no_secrets
from src.api.usage import Usage

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "BudgetExceeded",
    "CostEstimate",
    "LedgerEstimateError",
    "LedgerEntry",
    "RunEstimate",
    "SpendLedger",
    "estimate_run_cost",
    "estimate_run_cost_bracket",
    "require_confirmation",
]

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parents[2] / "results" / "spend_ledger.jsonl"


class BudgetExceeded(RuntimeError):
    """The run would push cumulative spend past the hard stop. Refusal, not a warning."""


class LedgerEstimateError(ValueError):
    """An optimistic estimate was offered where a gating figure is required."""


@dataclass
class CostEstimate:
    """A pre-submission estimate, itemised so it can be argued with."""

    run_id: str
    provider: str
    model: str
    n_requests: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    batch: bool
    cache_ttl: str
    estimated_usd: float
    token_source: str
    assume_cache_hits: bool = True
    assumptions: list[str] = field(default_factory=list)

    @property
    def is_gating(self) -> bool:
        """Only the pessimistic (no-cache-hits) estimate may gate spend.

        Batch prompt-cache hits are BEST-EFFORT — requests process concurrently and
        out of order — so the no-hits case is a live outcome, not a pathological one.
        A guard that evaluates the optimistic figure is guarding against the number
        that cannot blow the budget.
        """
        return not self.assume_cache_hits

    def render(self) -> str:
        lines = [
            f"Cost estimate — run {self.run_id}",
            f"  provider/model : {self.provider} / {self.model}",
            f"  requests       : {self.n_requests:,}",
            f"  batch          : {self.batch}   cache TTL: {self.cache_ttl}",
            f"  token source   : {self.token_source}",
            "",
            f"  input   (uncached) : {self.input_tokens:>12,}",
            f"  cache write        : {self.cache_write_tokens:>12,}",
            f"  cache read         : {self.cache_read_tokens:>12,}",
            f"  output             : {self.output_tokens:>12,}",
            "",
            f"  ESTIMATED COST     : ${self.estimated_usd:,.4f}",
        ]
        if self.assumptions:
            lines += ["", "  assumptions:"] + [f"    - {a}" for a in self.assumptions]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LedgerEntry:
    """One recorded row. ``cumulative_usd`` is the total INCLUDING this entry."""

    timestamp_utc: str
    run_id: str
    provider: str
    model: str
    batch_id: str | None
    input_tokens: int
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    output_tokens: int
    cost_usd: float
    cumulative_usd: float
    kind: str = "actual"  # "actual" | "estimate"
    estimated_usd: float | None = None
    n_requests: int | None = None
    notes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_run_cost(
    run_id: str,
    *,
    provider: str,
    model: str,
    n_requests: int,
    system_tokens: int,
    per_request_input_tokens: Iterable[int],
    expected_output_tokens: int,
    batch: bool = True,
    cache_ttl: str = "1h",
    token_source: str = "count_tokens endpoint",
    assume_cache_hits: bool = True,
    rate_card: dict[str, Any] | None = None,
) -> CostEstimate:
    """Estimate a run's cost BEFORE submitting it.

    ``system_tokens`` is the cacheable prefix, counted once as a cache write and then
    assumed to be read on every subsequent request. Prompt-cache hits inside a batch
    are BEST-EFFORT (requests process concurrently and out of order), so
    ``assume_cache_hits=False`` gives the pessimistic bound where the prefix is billed
    at full price every time. Both are estimates; the actual comes from
    ``usage.cache_read_input_tokens`` after the run.

    Args:
        per_request_input_tokens: Per-request token counts for the variable part
            (the clause), from the target model's own ``count_tokens``.
    """
    per_request = list(per_request_input_tokens)
    if len(per_request) != n_requests:
        raise ValueError(
            f"got {len(per_request)} per-request token counts for {n_requests} requests"
        )
    if system_tokens < 0 or expected_output_tokens < 0:
        raise ValueError("token counts must be non-negative")

    variable_input = sum(per_request)
    if assume_cache_hits:
        cache_write = system_tokens
        cache_read = system_tokens * max(0, n_requests - 1)
        uncached_input = variable_input
        assumptions = [
            "cacheable prefix written once, then read on every later request",
            "batch cache hits are BEST-EFFORT; this is the OPTIMISTIC bound and may "
            "NOT be used to gate spend",
        ]
    else:
        cache_write = 0
        cache_read = 0
        uncached_input = variable_input + system_tokens * n_requests
        assumptions = [
            "PESSIMISTIC bound: NO cache hits; prefix billed in full on every request",
            "this is the figure that gates spend — it is what can actually happen",
        ]

    estimated = compute_cost(
        model,
        input_tokens=uncached_input,
        output_tokens=expected_output_tokens * n_requests,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        cache_ttl=cache_ttl,
        batch=batch,
        rate_card=rate_card,
    )
    return CostEstimate(
        run_id=run_id,
        provider=provider,
        model=model,
        n_requests=n_requests,
        input_tokens=uncached_input,
        output_tokens=expected_output_tokens * n_requests,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        batch=batch,
        cache_ttl=cache_ttl,
        estimated_usd=estimated,
        token_source=token_source,
        assume_cache_hits=assume_cache_hits,
        assumptions=assumptions,
    )


@dataclass
class RunEstimate:
    """Both bounds for one run. ``gating_usd`` is the pessimistic figure.

    Cache hits inside a batch are best-effort, so the true cost lands somewhere in
    ``[optimistic, pessimistic]``. We report both and gate on the upper one.
    """

    optimistic: CostEstimate
    pessimistic: CostEstimate

    @property
    def run_id(self) -> str:
        return self.pessimistic.run_id

    @property
    def gating_usd(self) -> float:
        return self.pessimistic.estimated_usd

    @property
    def optimistic_usd(self) -> float:
        return self.optimistic.estimated_usd

    def render(self) -> str:
        return (
            f"{self.pessimistic.render()}\n\n"
            f"  optimistic (all cache hits) : ${self.optimistic_usd:,.4f}\n"
            f"  PESSIMISTIC (no cache hits) : ${self.gating_usd:,.4f}  <-- gates spend\n"
            "  Actual cost lands between these; batch cache hits are best-effort."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "optimistic_usd": self.optimistic_usd,
            "pessimistic_usd": self.gating_usd,
            "gating_usd": self.gating_usd,
            "optimistic": self.optimistic.as_dict(),
            "pessimistic": self.pessimistic.as_dict(),
        }


def estimate_run_cost_bracket(run_id: str, **kwargs: Any) -> RunEstimate:
    """Build both bounds. This is what spending entrypoints should use."""
    kwargs.pop("assume_cache_hits", None)
    return RunEstimate(
        optimistic=estimate_run_cost(run_id, assume_cache_hits=True, **kwargs),
        pessimistic=estimate_run_cost(run_id, assume_cache_hits=False, **kwargs),
    )


class SpendLedger:
    """Append-only JSONL ledger. Reads are cheap; the file is small by construction."""

    def __init__(
        self,
        path: str | Path = DEFAULT_LEDGER_PATH,
        rate_card: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rate_card = rate_card if rate_card is not None else load_rate_card()

    # ------------------------------------------------------------------ reading

    def entries(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(LedgerEntry(**json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    # Hard rule 11: a corrupt ledger must not be silently skipped —
                    # under-reading it under-reports spend and defeats the hard stop.
                    raise ValueError(f"{self.path}:{lineno}: unreadable entry: {exc}") from exc
        return rows

    def cumulative_usd(self, run_id: str | None = None) -> float:
        """Total ACTUAL spend recorded. Estimate rows are excluded — they are not money.

        ``run_id`` scopes the total to ONE producer. Two different limits read this
        ledger and they are not the same question:

        * the **$15 hard stop** (hard rule 8) governs ALL spend on this project, so it
          reads the whole file — ``run_id=None``;
        * a **per-service cap** governs only what that service spent, so it must read
          only that service's rows.

        Conflating them is not a rounding error, it is a silent outage: the deployed
        service's $1.00 cap was compared against $3.58 of *experiment* spend, so the cap
        was permanently "reached" and every request returned
        ``escalation_skipped=true`` — a service that looks healthy and never escalates.
        """
        return sum(e.cost_usd for e in self.entries()
                   if e.kind == "actual" and (run_id is None or e.run_id == run_id))

    @property
    def hard_stop_usd(self) -> float:
        return float(self._rate_card["budget"]["hard_stop_usd"])

    def remaining_usd(self) -> float:
        return self.hard_stop_usd - self.cumulative_usd()

    # ------------------------------------------------------------------ enforcing

    def assert_within_budget(self, estimate: "RunEstimate | CostEstimate | float") -> float:
        """Raise :class:`BudgetExceeded` if this run would breach the hard stop.

        Evaluated against the PESSIMISTIC (no-cache-hits) figure. Passing an
        optimistic :class:`CostEstimate` is refused outright rather than silently
        gating on the wrong number.

        Returns the projected total if the run is permitted.
        """
        if isinstance(estimate, RunEstimate):
            amount = estimate.gating_usd
        elif isinstance(estimate, CostEstimate):
            if not estimate.is_gating:
                raise LedgerEstimateError(
                    f"refusing to gate spend on the OPTIMISTIC estimate for run "
                    f"{estimate.run_id!r}. Batch cache hits are best-effort, so the "
                    "no-cache-hits figure is what can actually happen. Pass a "
                    "RunEstimate (estimate_run_cost_bracket) or an estimate built "
                    "with assume_cache_hits=False."
                )
            amount = estimate.estimated_usd
        else:
            amount = float(estimate)
        if amount < 0:
            raise ValueError("estimate must be non-negative")

        spent = self.cumulative_usd()
        projected = spent + amount
        if projected > self.hard_stop_usd:
            raise BudgetExceeded(
                f"REFUSING TO RUN. Recorded spend ${spent:,.4f} + estimate "
                f"${amount:,.4f} = ${projected:,.4f}, which exceeds the hard stop of "
                f"${self.hard_stop_usd:,.2f} (budget.hard_stop_usd in configs/costs.yaml). "
                f"Remaining budget: ${self.remaining_usd():,.4f}."
            )
        return projected

    # ------------------------------------------------------------------ writing

    def _append(self, entry: LedgerEntry) -> LedgerEntry:
        # The ledger is version-controlled (hard rule 12), so a leak here would be
        # committed. Guard at the point of write.
        assert_no_secrets(entry.as_dict(), context=f"ledger entry for run {entry.run_id}")
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # spend must survive a crash
        return entry

    def record_estimate(self, estimate: CostEstimate) -> LedgerEntry:
        """Record the pre-submission estimate. Does NOT count toward cumulative spend."""
        return self._append(
            LedgerEntry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                run_id=estimate.run_id,
                provider=estimate.provider,
                model=estimate.model,
                batch_id=None,
                input_tokens=estimate.input_tokens,
                cache_creation_input_tokens=estimate.cache_write_tokens,
                cache_read_input_tokens=estimate.cache_read_tokens,
                output_tokens=estimate.output_tokens,
                cost_usd=0.0,
                cumulative_usd=self.cumulative_usd(),
                kind="estimate",
                estimated_usd=estimate.estimated_usd,
                n_requests=estimate.n_requests,
            )
        )

    def record_actual(
        self,
        run_id: str,
        usage: Usage,
        *,
        provider: str,
        model: str,
        batch_id: str | None = None,
        batch: bool = True,
        cache_ttl: str = "1h",
        estimated_usd: float | None = None,
        n_requests: int | None = None,
        notes: str | None = None,
    ) -> LedgerEntry:
        """Record the ACTUAL cost, computed from the four usage fields separately.

        ``estimated_usd`` is carried alongside so the estimate-vs-actual record
        accumulates without a second join.
        """
        cost = compute_cost(
            model,
            **usage.as_cost_kwargs(),  # raises if cache fields were never reported
            cache_ttl=cache_ttl,
            batch=batch,
            rate_card=self._rate_card,
        )
        return self._append(
            LedgerEntry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                run_id=run_id,
                provider=provider,
                model=model,
                batch_id=batch_id,
                input_tokens=usage.input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                cumulative_usd=self.cumulative_usd() + cost,
                kind="actual",
                estimated_usd=estimated_usd,
                n_requests=n_requests,
                notes=notes,
            )
        )

    def estimate_vs_actual(self) -> list[dict[str, Any]]:
        """Per-run estimate/actual pairs with the error, for the report."""
        out = []
        for e in self.entries():
            if e.kind != "actual" or e.estimated_usd is None:
                continue
            err = e.cost_usd - e.estimated_usd
            out.append(
                {
                    "run_id": e.run_id,
                    "model": e.model,
                    "estimated_usd": e.estimated_usd,
                    "actual_usd": e.cost_usd,
                    "error_usd": err,
                    "relative_error": err / e.estimated_usd if e.estimated_usd else None,
                }
            )
        return out


def require_confirmation(
    estimate: "RunEstimate | CostEstimate",
    ledger: SpendLedger,
    *,
    confirm: bool,
    stream: Any = None,
) -> float:
    """Gate for every spending entrypoint (hard rule 8).

    Prints the itemised estimate, checks it against the hard stop, and refuses to
    proceed unless ``--confirm`` was passed. Returns the projected cumulative total.

    Both bounds are printed; the PESSIMISTIC one gates. Order matters: the budget check
    runs BEFORE the confirmation check, so a run that breaches the hard stop is refused
    even if the operator passed ``--confirm``.
    """
    import sys

    out = stream or sys.stdout
    print(estimate.render(), file=out)
    print(
        f"\n  recorded spend : ${ledger.cumulative_usd():,.4f}"
        f"\n  hard stop      : ${ledger.hard_stop_usd:,.2f}"
        f"\n  remaining      : ${ledger.remaining_usd():,.4f}",
        file=out,
    )

    projected = ledger.assert_within_budget(estimate)  # raises BudgetExceeded
    gating = (
        estimate.gating_usd if isinstance(estimate, RunEstimate) else estimate.estimated_usd
    )

    if not confirm:
        raise SystemExit(
            "\nNOT SENDING. This is an estimate only. Re-run with --confirm to spend "
            f"up to ${gating:,.4f} (pessimistic bound)."
        )
    print(f"\n  CONFIRMED — projected cumulative: ${projected:,.4f}\n", file=out)
    return projected
