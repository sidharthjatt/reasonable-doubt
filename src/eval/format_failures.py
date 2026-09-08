"""Break Tier 1's format failures down by CAUSE, so the cost of exact matching is visible.

Tier 1 emits a label STRING and it is matched EXACTLY (PREREGISTRATION 3ag). Every
near-miss is therefore both a format failure and a wrong answer. That is a deliberate
choice, and this module measures its price rather than arguing about it.

The categories are mutually exclusive and are tried in order:

  pluralisation   the emitted string differs from exactly one real label only by a
                  trailing 's' -- 'Governing Law' vs 'Governing Laws'
  case_or_edge    normalising already handles these, so a hit here means the normaliser
                  regressed; it should always be 0
  ambiguous       the string is a near-miss for MORE THAN ONE label, so no fold could
                  repair it without choosing arbitrarily -- this is the number that
                  justifies refusing to fold
  other           anything else: hallucinated labels, prose, truncation
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


def _depluralise_candidates(s: str, names: list[str], key) -> list[str]:
    """Labels that `s` would match if a trailing 's' were added or removed."""
    ks = key(s)
    out = []
    for n in names:
        kn = key(n)
        if ks == kn:
            continue
        if ks + "s" == kn or ks == kn + "s":
            out.append(n)
    return out


@dataclass(frozen=True)
class FailureBreakdown:
    n_rows: int
    n_failures: int
    pluralisation: int
    case_or_edge: int
    ambiguous: int
    other: int
    examples: dict[str, list[str]] = field(default_factory=dict)
    ambiguous_pairs: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.n_failures / self.n_rows if self.n_rows else 0.0

    @property
    def recoverable_by_plural_fold(self) -> int:
        """Failures a naive singular/plural fold WOULD repair. The cost of not folding."""
        return self.pluralisation

    def as_dict(self) -> dict:
        return {"n_rows": self.n_rows, "n_failures": self.n_failures,
                "failure_rate": self.failure_rate,
                "pluralisation": self.pluralisation, "case_or_edge": self.case_or_edge,
                "ambiguous": self.ambiguous, "other": self.other,
                "recoverable_by_plural_fold": self.recoverable_by_plural_fold,
                "ambiguous_pairs": self.ambiguous_pairs[:20],
                "examples": {k: v[:5] for k, v in self.examples.items()}}


def breakdown(raws: list[str], preds: list[str | None], names: list[str],
              key, normalize) -> FailureBreakdown:
    """Categorise every row whose output did not normalise."""
    cats: Counter[str] = Counter()
    ex: dict[str, list[str]] = {}
    amb: list[tuple[str, list[str]]] = []

    for raw, pred in zip(raws, preds):
        if pred is not None:
            continue
        cands = _depluralise_candidates(raw, names, key)
        if normalize(raw) is not None:
            cat = "case_or_edge"          # unreachable unless the normaliser regressed
        elif len(cands) == 1:
            cat = "pluralisation"
        elif len(cands) > 1:
            cat = "ambiguous"
            amb.append((raw, cands))
        else:
            cat = "other"
        cats[cat] += 1
        ex.setdefault(cat, []).append(raw)

    return FailureBreakdown(
        n_rows=len(raws), n_failures=sum(cats.values()),
        pluralisation=cats["pluralisation"], case_or_edge=cats["case_or_edge"],
        ambiguous=cats["ambiguous"], other=cats["other"], examples=ex,
        ambiguous_pairs=amb)


def collision_audit(names: list[str], key) -> list[tuple[str, str]]:
    """Pairs of REAL labels that a naive singular/plural fold would merge.

    This is the argument against folding, stated as data: if the label space itself
    contains pairs that differ only by a trailing 's', a fold does not repair
    near-misses, it destroys distinctions that LEDGAR draws.
    """
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ka, kb = key(a), key(b)
            if ka + "s" == kb or kb + "s" == ka:
                out.append((a, b))
    return out
