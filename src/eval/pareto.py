"""E5/E6 cost-accuracy Pareto frontier for the cascade.

Every price comes from `configs/costs.yaml` via `src.api.cost` (hard rule 5: no
hardcoded prices anywhere). Nothing here calls an API; the whole frontier is simulated
offline from cached Stage 1 results and Tier 0 logits, at zero spend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CascadePoint:
    escalation_rate: float
    threshold: float
    accuracy: float
    macro_f1: float
    usd_per_1k: float
    n_escalated: int


def cascade_points(
    *,
    score: np.ndarray,
    tier0_correct: np.ndarray,
    api_correct: np.ndarray,
    tier0_usd_per_1k: float,
    api_usd_per_1k: float,
    macro_f1_fn=None,
    thresholds: np.ndarray | None = None,
) -> list[CascadePoint]:
    """Sweep the escalation threshold and cost each operating point.

    Escalate when `score < threshold`; escalated rows take the API's answer, retained
    rows keep Tier 0's. Cost is a per-1k blend: Tier 0 pays for every row, the API only
    for escalated ones — which is the whole economic claim the cascade makes.
    """
    score = np.asarray(score, float)
    t0 = np.asarray(tier0_correct, bool)
    api = np.asarray(api_correct, bool)
    if not (len(score) == len(t0) == len(api)):
        raise ValueError(
            f"length mismatch: score {len(score)}, tier0 {len(t0)}, api {len(api)}. "
            f"These must be the SAME rows in the SAME order — see src/router/join.py."
        )
    if thresholds is None:
        thresholds = np.quantile(score, np.linspace(0.0, 1.0, 201))

    out: list[CascadePoint] = []
    for t in thresholds:
        esc = score < t
        combined = np.where(esc, api, t0)
        out.append(CascadePoint(
            escalation_rate=float(esc.mean()),
            threshold=float(t),
            accuracy=float(combined.mean()),
            macro_f1=float(macro_f1_fn(esc)) if macro_f1_fn else float("nan"),
            usd_per_1k=tier0_usd_per_1k + api_usd_per_1k * float(esc.mean()),
            n_escalated=int(esc.sum()),
        ))
    return out


def pareto_frontier(points: list[CascadePoint]) -> list[CascadePoint]:
    """Points not dominated on (cost down, accuracy up).

    A point is dominated when another is at least as accurate AND no more expensive, and
    strictly better on one of the two.
    """
    keep: list[CascadePoint] = []
    for p in sorted(points, key=lambda q: (q.usd_per_1k, -q.accuracy)):
        if not keep or p.accuracy > keep[-1].accuracy + 1e-12:
            keep.append(p)
    return keep


def oracle_point(*, tier0_correct: np.ndarray, api_correct: np.ndarray,
                 tier0_usd_per_1k: float, api_usd_per_1k: float) -> CascadePoint:
    """Upper bound: escalate exactly the rows Tier 0 gets wrong, and no others.

    Unreachable by any real signal — it needs the labels — but it is the ceiling every
    routing signal is measured against, and the gap to it is what E5 is actually
    reporting.
    """
    t0 = np.asarray(tier0_correct, bool)
    esc = ~t0
    combined = np.where(esc, np.asarray(api_correct, bool), t0)
    return CascadePoint(
        escalation_rate=float(esc.mean()), threshold=float("nan"),
        accuracy=float(combined.mean()), macro_f1=float("nan"),
        usd_per_1k=tier0_usd_per_1k + api_usd_per_1k * float(esc.mean()),
        n_escalated=int(esc.sum()),
    )
