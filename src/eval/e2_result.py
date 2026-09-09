"""E2's result, and the §3ae band that may never be separated from it.

PREREGISTRATION §3ae resolved `seed_sd` to the POINT ESTIMATE, and registered a
**mandatory reporting condition** as the price of the permissive reading:

    the chi-square band must accompany E2's result EVERY TIME IT APPEARS

A margin quoted without the band overstates its precision by up to an order of
magnitude — the sample sd of three numbers is a 12x-wide estimate of the sd it
estimates, so a regime-A point estimate can have an interval reaching regime C. With
the measured `seed_sd = 0.0032` it does: the band's upper end is 0.0557, inside regime
C (> 0.050).

**This module exists so that condition is ENFORCED, not remembered.** There is no
accessor here that returns a margin, a delta or a verdict as text without the band
attached. `accepted` is a bool, which cannot be quoted as a precise-looking number;
every *textual* and *serialised* path goes through one formatter that emits the band
with it. `tests/test_e2_result.py` asserts that property over every public rendering
method, so a future method that forgets it fails the suite rather than shipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT2, Z = math.sqrt(2.0), 1.96

# chi-square(n-1) 95% multipliers on the sample sd. n=3 is the registered seed count.
# `tests/test_e2_result.py` pins these against scripts/c3_substitute.py so the two
# cannot drift apart — the substituted rule and the reported result must use one band.
_CHI2_95 = {3: (0.5207, 6.2847)}


def margin_for(seed_sd: float) -> float:
    """E2's preregistered formula: sqrt(2) * 1.96 * seed_sd."""
    return SQRT2 * Z * seed_sd


@dataclass(frozen=True)
class E2Band:
    """The interval E2's margin inherits from estimating an sd on 3 seeds."""
    seed_sd: float
    n_seeds: int

    def __post_init__(self) -> None:
        if self.n_seeds not in _CHI2_95:
            raise ValueError(
                f"no registered chi-square band for n_seeds={self.n_seeds}; §3ae is "
                f"written for 3 seeds. Adding a seed count is an amendment, not a "
                f"default — the band multipliers change with it.")
        if not (self.seed_sd > 0):
            raise ValueError(f"seed_sd must be positive, got {self.seed_sd!r}")

    @property
    def _mult(self) -> tuple[float, float]:
        return _CHI2_95[self.n_seeds]

    @property
    def margin(self) -> float:
        return margin_for(self.seed_sd)

    @property
    def sd_lo(self) -> float: return self._mult[0] * self.seed_sd

    @property
    def sd_hi(self) -> float: return self._mult[1] * self.seed_sd

    @property
    def margin_lo(self) -> float: return margin_for(self.sd_lo)

    @property
    def margin_hi(self) -> float: return margin_for(self.sd_hi)

    def regime(self, m: float) -> str:
        """§3ae's regimes, applied to any margin value (point estimate or band end)."""
        if m <= 0.030: return "A"
        if m <= 0.050: return "B"
        return "C"

    def sentence(self) -> str:
        """The §3ae mandatory reporting condition, in the form it registers."""
        return (
            f"margin {self.margin:.4f} from seed_sd {self.seed_sd:.4f}, "
            f"{self.n_seeds} seeds; 95% interval on the true sd is "
            f"[{self.sd_lo:.4f}, {self.sd_hi:.4f}], so the margin spans "
            f"[{self.margin_lo:.4f}, {self.margin_hi:.4f}] — a regime-"
            f"{self.regime(self.margin)} point estimate may have an interval reaching "
            f"regime {self.regime(self.margin_hi)}.")


@dataclass(frozen=True)
class E2Result:
    """One E2 arm scored against E1. Nothing here renders without the band."""
    arm: str
    arm_macro_f1: float
    e1_macro_f1: float
    band: E2Band
    split: str = "test_3000"
    paired: bool = True

    @property
    def delta(self) -> float:
        return self.arm_macro_f1 - self.e1_macro_f1

    @property
    def accepted(self) -> bool:
        return self.delta >= self.band.margin

    def statement(self) -> str:
        """The ONLY textual rendering. The band is concatenated, not optional."""
        verdict = "ACCEPTED" if self.accepted else "NOT ACCEPTED"
        return (
            f"E2 [{self.arm}] on {self.split}"
            f"{' (paired)' if self.paired else ''}: {self.arm_macro_f1:.4f} vs E1 "
            f"{self.e1_macro_f1:.4f}, delta {self.delta:+.4f} against margin "
            f"{self.band.margin:.4f} — {verdict}. {self.band.sentence()}")

    def as_dict(self) -> dict:
        """Serialised form. `mandatory_band_statement` is always present."""
        return {
            "experiment": "E2", "arm": self.arm, "split": self.split,
            "paired": self.paired,
            "arm_macro_f1": self.arm_macro_f1, "e1_macro_f1": self.e1_macro_f1,
            "delta": self.delta, "margin": self.band.margin,
            "seed_sd": self.band.seed_sd, "n_seeds": self.band.n_seeds,
            "margin_band_95": [self.band.margin_lo, self.band.margin_hi],
            "sd_band_95": [self.band.sd_lo, self.band.sd_hi],
            "regime_point": self.band.regime(self.band.margin),
            "regime_band_upper": self.band.regime(self.band.margin_hi),
            "accepted": self.accepted,
            "mandatory_band_statement": self.band.sentence(),
        }

    def __str__(self) -> str:
        return self.statement()
