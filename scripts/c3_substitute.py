"""C3: measure across-seed macro-F1 std, substitute it into E2's preregistered formula.

    python scripts/c3_substitute.py --results-dir <dir> [--write]

E2's accept rule is registered as a FORMULA with one free parameter:

    best arm beats E1 by  >=  sqrt(2) * 1.96 * seed_sd

Substituting a measured value into a preregistered formula is not writing a new rule, so
hard rule 6 holds through this step — but ONLY if the substitution is mechanical. Hence
this script rather than a human with a calculator and three open JSON files.

WHAT THIS DELIBERATELY DOES NOT PRINT
-------------------------------------
Nothing about E1, E2 or E3. Not the per-seed macro-F1 values, not their mean, not any
test-split number. It prints `seed_sd`, the derived margin, and their uncertainty.

The sealing is STRUCTURAL, not disciplinary: C3's registered metric is
"macro-F1 std across >=3 training seeds of E1 on train_holdout_3000" — the SELECTION
split — so this script has no reason to open a test field and does not. A standard
deviation carries no information about the level it was computed around, so printing it
reveals nothing about whether E1 passed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import statistics
import sys
from pathlib import Path

SQRT2, Z = math.sqrt(2.0), 1.96
SEEDS = (1, 2, 3)
DATE = _dt.date.today().isoformat()
# chi-square(n-1) 95% interval, n=3. The sd of 3 numbers is a very noisy estimate of the
# sd it is estimating; these are the multipliers that say how noisy.
SIGMA_OVER_S_LO, SIGMA_OVER_S_HI = 0.5207, 6.2847

# The exact preregistered paragraph this substitution replaces. Matched in full rather
# than by a prefix, so a rule that has been edited by hand since registration fails to
# match instead of being silently overwritten.
REG_BLOCK = (
    "- **Accept rule: NOT YET SETTABLE. [C3-gated \u2014 blocking]** The margin is\n"
    "  **\u221a2 \u00d7 1.96 \u00d7 seed_sd**, where `seed_sd` is the across-seed macro-F1 std measured by\n"
    "  C3. It will be filled in once C3 reports, and before E2 runs."
)


def margin(seed_sd: float) -> float:
    return SQRT2 * Z * seed_sd


def read_selection_macro_f1(results_dir: Path, loss_arm: str = "ce") -> list[float]:
    """Read ONLY `selection.macro_f1` from each seed file. Nothing else is touched."""
    out = []
    for s in SEEDS:
        p = results_dir / f"tier0_{loss_arm}_seed{s}.json"
        if not p.exists():
            raise SystemExit(
                f"missing {p}. C3 needs all {len(SEEDS)} seeds — a std over fewer is not "
                f"the quantity E2's formula names, and hard rule 2 requires >=3 seeds."
            )
        d = json.loads(p.read_text())
        if d.get("loss_arm") != loss_arm:
            raise SystemExit(f"{p} has loss_arm={d.get('loss_arm')!r}, expected {loss_arm!r}")
        if d.get("selection_split") != "train_holdout_3000":
            raise SystemExit(
                f"{p} selection_split is {d.get('selection_split')!r}, not "
                f"'train_holdout_3000'. C3's registered metric is the std on the "
                f"SELECTION split; refusing to compute it from anything else."
            )
        out.append(float(d["selection"]["macro_f1"]))
    return out


def interpret(seed_sd: float, m: float) -> list[str]:
    """The reaction, preregistered in PREREGISTRATION 3ae BEFORE any number existed."""
    lines = []
    if m <= 0.030:
        lines.append("REGIME A — E2 is comfortably testable. The margin is at or below "
                     "the 0.03 gain E2 itself calls plausible at 137.7x imbalance, so a "
                     "real class-weighting effect of that size would be detected.")
    elif m <= 0.050:
        lines.append("REGIME B — E2 is testable only for a LARGE effect. The margin now "
                     "exceeds the 0.03 that E2 calls plausible, so the preregistered "
                     "expected effect would be declared not-accepted. Report E2's result "
                     "WITH this margin stated, and state that a plausible-sized effect "
                     "was outside detection.")
    else:
        lines.append("REGIME C — E2 is EFFECTIVELY UNTESTABLE at 3 seeds. The margin "
                     "exceeds the realistic headroom above E1 (LexGLUE DeBERTa macro-F1 "
                     "is 0.831), so essentially no attainable arm could clear it. "
                     "Running sqrt_inv_freq would consume 3-5 GPU-hours to produce a "
                     "foregone conclusion.")
    if seed_sd > 0.030:
        lines.append("C3's OWN falsification line is crossed (seed std > ~0.03): several "
                     "rules are unfalsifiable as written and must be loosened or moved "
                     "to paired comparisons. This was registered in C3 before the run.")
    lines.append(
        f"UNCERTAINTY ON THE MARGIN ITSELF: with 3 seeds the true sd lies in roughly "
        f"[{SIGMA_OVER_S_LO * seed_sd:.4f}, {SIGMA_OVER_S_HI * seed_sd:.4f}] at 95%, so "
        f"the margin implied by that interval spans "
        f"[{margin(SIGMA_OVER_S_LO * seed_sd):.4f}, {margin(SIGMA_OVER_S_HI * seed_sd):.4f}]. "
        f"A 12x band. The point estimate is what the formula registers, and it is used, "
        f"but the regime above is not robust to this interval.")
    return lines


def rewrite_preregistration(path: Path, seed_sd: float, m: float, n_seeds: int) -> None:
    """Replace E2's C3-gated paragraph with the substituted numeric rule.

    Matches the registered paragraph IN FULL. If E2's accept rule has been edited since
    registration the match fails and this raises, rather than overwriting an unknown rule
    with a generated one.
    """
    src = path.read_text()
    if src.count(REG_BLOCK) != 1:
        raise SystemExit(
            f"E2's registered C3-gated paragraph was not found exactly once in {path} "
            f"(found {src.count(REG_BLOCK)}). Either the substitution has already run, or "
            f"the rule was edited by hand. Refusing to guess where the number goes."
        )
    new = (
        f"- **Accept rule (SUBSTITUTED from C3, {n_seeds} seeds, {DATE}):** the arm beats "
        f"E1 by **\u2265 {m:.4f} macro-F1** on `test_3000`, paired on identical rows.\n"
        f"  - Derived mechanically by `scripts/c3_substitute.py` from the preregistered "
        f"formula `\u221a2 \u00d7 1.96 \u00d7 seed_sd`, with the C3-measured "
        f"`seed_sd = {seed_sd:.6f}` \u2014 the across-seed macro-F1 std on "
        f"`train_holdout_3000`, which is C3's registered metric. **No E1, E2 or E3 value "
        f"was read to produce this number**, and none appears in the script's output.\n"
        f"  - **Uncertainty the rule inherits:** at {n_seeds} seeds the true sd lies in "
        f"roughly [{SIGMA_OVER_S_LO * seed_sd:.4f}, {SIGMA_OVER_S_HI * seed_sd:.4f}] "
        f"(95%, chi-square), so the margin implied by that interval spans "
        f"[{margin(SIGMA_OVER_S_LO * seed_sd):.4f}, "
        f"{margin(SIGMA_OVER_S_HI * seed_sd):.4f}] \u2014 a 12\u00d7 band. The point "
        f"estimate is what the formula registers and is what is used; the band is stated "
        f"because the conclusion is not robust to it.\n"
        f"  - **Regime, per \u00a73ae, written before this number existed:**\n"
        + "".join(f"    - {ln}\n" for ln in interpret(seed_sd, m))
        + f"  - *Formerly: \"NOT YET SETTABLE. [C3-gated \u2014 blocking]\", registered as "
          f"the formula above.*"
    )
    path.write_text(src.replace(REG_BLOCK, new, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--loss-arm", default="ce")
    ap.add_argument("--prereg", type=Path,
                    default=Path(__file__).resolve().parents[1] / "PREREGISTRATION.md")
    ap.add_argument("--write", action="store_true",
                    help="write the substituted rule into PREREGISTRATION.md")
    a = ap.parse_args()

    vals = read_selection_macro_f1(a.results_dir, a.loss_arm)
    seed_sd = statistics.stdev(vals)          # ddof=1, the across-seed std C3 registers
    m = margin(seed_sd)

    print(f"C3 — {len(vals)} seeds, macro-F1 std on train_holdout_3000")
    print(f"  seed_sd = {seed_sd:.6f}")
    print(f"  E2 margin = sqrt(2) * 1.96 * seed_sd = {m:.4f} macro-F1")
    print()
    for ln in interpret(seed_sd, m):
        print(f"  {ln}")
    print()
    if a.write:
        rewrite_preregistration(a.prereg, seed_sd, m, len(vals))
        print(f"  WROTE the substituted rule into {a.prereg}")
        print("  Commit this BEFORE opening any E1/E2/E3 number.")
    else:
        print("  (dry run — pass --write to substitute into PREREGISTRATION.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
