"""C3 substitution: mechanical, sealed, and refusing anything it cannot verify."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("c3", ROOT / "scripts" / "c3_substitute.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["c3"] = mod
    spec.loader.exec_module(mod)
    return mod


c3 = _load()


def write_seeds(d: Path, sel_f1: list[float], *, loss_arm="ce",
                selection_split="train_holdout_3000") -> Path:
    for i, f1 in enumerate(sel_f1, start=1):
        (d / f"tier0_{loss_arm}_seed{i}.json").write_text(json.dumps({
            "loss_arm": loss_arm, "seed": i, "selection_split": selection_split,
            "selection": {"macro_f1": f1, "accuracy": 0.9},
            # Values the script must never read. Deliberately absurd: if any of them
            # reached the output or the margin, it would be unmistakable.
            "test_3000_fp32": {"macro_f1": 999.0, "accuracy": 999.0},
            "test_3000_int8": {"macro_f1": -999.0, "accuracy": -999.0},
            "e3_int8_minus_fp32_macro_f1": 123456.0,
        }))
    return d


def test_margin_is_the_registered_formula():
    assert c3.margin(0.01) == pytest.approx(2**0.5 * 1.96 * 0.01)
    assert c3.margin(0.0) == 0.0


def test_seed_sd_is_the_sample_std_of_the_selection_split(tmp_path):
    vals = [0.820, 0.826, 0.831]
    write_seeds(tmp_path, vals)
    got = c3.read_selection_macro_f1(tmp_path)
    assert got == vals
    assert statistics.stdev(got) == pytest.approx(statistics.stdev(vals))


def test_it_reads_only_the_selection_field(tmp_path, capsys):
    """The sealing property. Test numbers are planted as 999/-999/123456; none may
    appear in the output or influence the margin."""
    write_seeds(tmp_path, [0.820, 0.826, 0.831])
    sys.argv = ["c3", "--results-dir", str(tmp_path)]
    c3.main()
    out = capsys.readouterr().out
    for forbidden in ("999", "123456", "-999"):
        assert forbidden not in out, f"a sealed value {forbidden!r} leaked into the output"
    assert "0.005" in out or "seed_sd" in out


def test_output_does_not_reveal_the_level_only_the_spread(tmp_path, capsys):
    """Two seed sets with the SAME spread but very different means must produce
    identical output. That is what makes printing seed_sd safe."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    write_seeds(a, [0.700, 0.706, 0.711])
    write_seeds(b, [0.900, 0.906, 0.911])
    outs = []
    for d in (a, b):
        sys.argv = ["c3", "--results-dir", str(d)]
        c3.main()
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1], "the output differs with the mean — it is not sealed"


def test_missing_seed_refuses_rather_than_using_two(tmp_path):
    write_seeds(tmp_path, [0.82, 0.83])          # only seeds 1 and 2
    with pytest.raises(SystemExit, match="all 3 seeds|>=3 seeds|missing"):
        c3.read_selection_macro_f1(tmp_path)


def test_wrong_selection_split_refuses(tmp_path):
    write_seeds(tmp_path, [0.82, 0.83, 0.84], selection_split="test_3000")
    with pytest.raises(SystemExit, match="SELECTION split"):
        c3.read_selection_macro_f1(tmp_path)


def test_wrong_loss_arm_refuses(tmp_path):
    write_seeds(tmp_path, [0.82, 0.83, 0.84], loss_arm="sqrt_inv_freq")
    with pytest.raises(SystemExit, match="loss_arm"):
        c3.read_selection_macro_f1(tmp_path)   # asks for "ce", files say otherwise


@pytest.mark.parametrize("sd,regime", [
    (0.004, "REGIME A"),        # margin 0.0157
    (0.014, "REGIME B"),        # margin 0.0388
    (0.025, "REGIME C"),        # margin 0.0693
])
def test_regimes_are_reported_as_preregistered(sd, regime):
    lines = " ".join(c3.interpret(sd, c3.margin(sd)))
    assert regime in lines
    assert "12x" in lines or "12×" in lines or "UNCERTAINTY" in lines


def test_c3_own_falsification_line_is_flagged():
    lines = " ".join(c3.interpret(0.035, c3.margin(0.035)))
    assert "falsification line is crossed" in lines.lower() or "OWN falsification" in lines


def test_substitution_replaces_the_registered_paragraph_exactly(tmp_path):
    prereg = tmp_path / "P.md"
    prereg.write_text("before\n" + c3.REG_BLOCK + "\nafter\n")
    c3.rewrite_preregistration(prereg, 0.005, c3.margin(0.005), 3)
    got = prereg.read_text()
    assert c3.REG_BLOCK not in got
    assert "SUBSTITUTED from C3" in got and "0.0139" in got
    assert got.startswith("before\n") and got.rstrip().endswith("after")


def test_substitution_refuses_an_edited_rule(tmp_path):
    prereg = tmp_path / "P.md"
    prereg.write_text("- **Accept rule:** someone edited this by hand\n")
    with pytest.raises(SystemExit, match="Refusing to guess"):
        c3.rewrite_preregistration(prereg, 0.005, c3.margin(0.005), 3)


def test_substitution_is_not_idempotent_it_refuses_a_second_run(tmp_path):
    prereg = tmp_path / "P.md"
    prereg.write_text(c3.REG_BLOCK + "\n")
    c3.rewrite_preregistration(prereg, 0.005, c3.margin(0.005), 3)
    with pytest.raises(SystemExit, match="already run|Refusing"):
        c3.rewrite_preregistration(prereg, 0.005, c3.margin(0.005), 3)


SUBSTITUTED_MARKER = "**Accept rule (SUBSTITUTED from C3,"


def test_e2_rule_is_in_exactly_one_of_its_two_valid_states():
    """E2's accept rule has exactly two legitimate states, and this test guards BOTH.

    BEFORE substitution: the registered C3-gated paragraph is present verbatim, so
    `c3_substitute.py` will find it on the day it is needed. That was this test's
    original job.

    AFTER substitution: that paragraph is legitimately gone, replaced by the numeric
    rule. Asserting the old state here would then fail forever and the natural fix --
    deleting the test -- would remove drift detection entirely at the exact moment the
    rule became load-bearing. So the post-substitution state is checked instead: the
    rule carries its margin, the mandatory 3ae band, and the regime.

    Anything else -- neither state, or both -- means the rule was hand-edited.
    """
    src = (ROOT / "PREREGISTRATION.md").read_text()
    pre = src.count(c3.REG_BLOCK)
    post = src.count(SUBSTITUTED_MARKER)
    assert (pre, post) in {(1, 0), (0, 1)}, (
        f"E2's accept rule is in neither valid state (registered block x{pre}, "
        f"substituted block x{post}). It was edited by hand, or the substitution ran "
        f"twice. Fix before any E1/E2/E3 number is opened.")

    if post == 1:
        # substituted: the number and its band must both be there (3ae is mandatory)
        assert "√2 × 1.96 × seed_sd" in src, "the derivation was dropped"
        assert "seed_sd = " in src, "the measured seed_sd was dropped"
        assert "95%, chi-square" in src, "the 3ae uncertainty band was dropped"
        assert "REGIME" in src, "the preregistered regime interpretation was dropped"


def test_substitution_refuses_to_run_twice():
    """Idempotence guard: once substituted, the script must refuse rather than
    re-substitute or guess where the number goes."""
    src = (ROOT / "PREREGISTRATION.md").read_text()
    if src.count(c3.REG_BLOCK) == 0:
        import pytest
        with pytest.raises(SystemExit, match="already run|not found exactly once"):
            c3.rewrite_preregistration(ROOT / "PREREGISTRATION.md", 0.0032, 0.0089, 3)
