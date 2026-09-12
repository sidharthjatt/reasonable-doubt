"""The registered scorer must be the one that produced the numbers — PREREGISTRATION 3ay.

WHY THIS FILE EXISTS, AND WHY THE OBVIOUS TEST WAS REJECTED
-----------------------------------------------------------
The first proposed mitigation was "assert sklearn and src.eval.metrics.score agree on the
committed artefacts". That test is **green today only by luck**: every committed number was
scored on `test_3000`, whose gold covers all 100 classes, so the two calls happened to
average over the same label set and agreed to within 4 ULP. It would stay green until a
subset arm appeared — which is precisely when it would be needed. **A test that cannot fail
on the defect it guards is the proxy-check shape (3e).**

So the tests below are built to FAIL if the guard is removed:

  1. `test_subset_case_must_diverge` constructs the case where the two implementations MUST
     disagree, and asserts both the direction and that the divergence is large — not ULP
     scale. If someone "fixes" metrics.score by delegating to sklearn, this goes red.
  2. `test_scorer_refuses_absent_classes` asserts the guard raises rather than deflating.
  3. `test_committed_artefacts_record_classes_averaged` asserts the label set is RECORDED,
     because macro-F1 is meaningless without it — with an explicit grandfathered list so
     the outstanding debt is visible and shrinking rather than silently tolerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.metrics import score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Artefacts produced BEFORE 3ay, by the sklearn bypass path (kaggle_tier0.py:307/310,
# kaggle_tier0_dev.py:123, and the free-tier rung-0 scripts). They carry a macro_f1 with no
# classes_averaged. Listed EXPLICITLY so the debt is visible: this list may shrink, and a
# new artefact may never be added to it.
GRANDFATHERED = {
    "tier0_ce_seed1.json", "tier0_ce_seed2.json", "tier0_ce_seed3.json",
    "tier0_dev_ce_seed1.json", "tier0_dev_ce_seed2.json", "tier0_dev_ce_seed3.json",
    "rung0_groq_gptoss120b.json", "rung0_groq_qwen3.6-27b.json",
}
# e6_frontier.json was grandfathered on 2026-09-10 and CAME OFF the list the same day: it
# now records classes_averaged: 100 and its scorer. The list shrank, which is the only
# direction test_grandfathered_list_only_shrinks permits.


def _sklearn_macro(gold, pred, labels=None):
    from sklearn.metrics import f1_score
    p = [x if x is not None else "\x00__UNMATCHED__" for x in pred]
    kw = {"labels": labels} if labels is not None else {}
    return float(f1_score(gold, p, average="macro", zero_division=0, **kw))


def test_subset_case_must_diverge():
    """The two implementations MUST disagree when the model predicts a class absent from
    gold — the exact E8 situation (97 of 100 classes present).

    This is the test the naive version could not be: on `test_3000` gold covers all 100
    classes, nothing can be predicted outside it, and the two agree to 4 ULP. Here they
    diverge by a *reportable* amount, and the assertion is that they diverge — so the test
    goes red if metrics.score is ever reimplemented on top of sklearn's default behaviour.
    """
    # 4 gold classes; the model predicts a 5th that has no gold examples.
    gold = ["A", "A", "B", "B", "C", "C", "D", "D"]
    pred = ["A", "A", "B", "B", "C", "E", "D", "D"]   # "E" is absent from gold

    registered = score(gold, pred, labels=sorted(set(gold))).macro_f1
    sklearn_default = _sklearn_macro(gold, pred)      # averages over gold u pred = 5 classes

    assert registered > sklearn_default, (
        "sklearn's default averages over gold u pred, scoring the absent class 0.0 and "
        "DEFLATING macro-F1. If these are equal, the registered scorer no longer protects "
        "against the deflation it exists to prevent."
    )
    # Large, not ULP scale: 1/5 of the average is a zero that should not be there.
    assert registered - sklearn_default > 0.05, (
        f"divergence {registered - sklearn_default:.4f} is too small to be the deflation "
        "effect; this test is no longer exercising the defect it guards"
    )


def test_scorer_refuses_absent_classes():
    """metrics.score RAISES rather than silently deflating — hard rule 11."""
    gold = ["A", "A", "B", "B"]
    with pytest.raises(ValueError, match="NO gold examples"):
        score(gold, gold, labels=["A", "B", "C", "D"])
    # ...and averages them in only when the caller says so explicitly.
    r = score(gold, gold, labels=["A", "B", "C", "D"], allow_absent_classes=True)
    assert r.macro_f1 == pytest.approx(0.5)
    assert r.classes_averaged == 4
    assert sorted(r.absent_classes) == ["C", "D"]


def test_e8_subset_divergence_is_real_and_recorded():
    """E8's own numbers: the registered scorer and sklearn's default differ materially."""
    from tests.conftest import require_artifact

    p = require_artifact(RESULTS / "e8_fewshot.json",
                         "E8's recorded few-shot metrics; results/* is gitignored so "
                         "experiment outputs are absent from a clean checkout")
    d = json.loads(p.read_text())
    assert d["classes_averaged"] == 97, "E8 must record the label set it averaged over"
    for arm in ("fewshot", "zeroshot"):
        assert d["reports"][arm]["classes_averaged"] == 97
        assert d["reports"][arm]["absent_classes"] == []


def test_committed_artefacts_record_classes_averaged():
    """Any artefact carrying a macro_f1 must record the label set it averaged over.

    macro-F1 is defined relative to that set; without it the number cannot be compared to
    another. Grandfathered artefacts are listed by name so the debt is explicit.
    """
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            for i, v in enumerate(o[:2]):
                yield from walk(v, f"{path}[{i}]")
        else:
            yield path, o

    missing = []
    for f in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        keys = [k for k, _ in walk(d)]
        if any(k.endswith("macro_f1") for k in keys) and not any(
                "classes_averaged" in k for k in keys):
            if f.name not in GRANDFATHERED:
                missing.append(f.name)

    assert not missing, (
        f"{len(missing)} artefact(s) report a macro_f1 without recording "
        f"classes_averaged: {missing}. macro-F1 is defined relative to the label set "
        "averaged over; add it, or add the file to GRANDFATHERED with a reason."
    )


def _extract_notebook_port(name: str):
    """Pull the ported scorer out of a standalone Kaggle notebook and make it callable.

    The notebooks cannot import src/ — they run on Kaggle where the repo is absent — so the
    scorer is PORTED into them. A port that is never compared to its original is two
    implementations again (3ay), so this executes the real shipped block.
    """
    import numpy as np
    from sklearn.metrics import f1_score

    src = (ROOT / "notebooks" / name).read_text()
    start = src.index("# --- BEGIN PORT")
    end = src.index("# --- END PORT")
    ns: dict = {"np": np, "f1_score": f1_score}
    exec(src[start:end], ns)          # noqa: S102 — executing our own committed source
    return ns["_macro_report"]


@pytest.mark.parametrize("notebook", ["kaggle_tier0.py", "kaggle_tier0_dev.py"])
def test_notebook_port_matches_registered_scorer(notebook):
    """The ported scorer must agree with src.eval.metrics.score, INCLUDING on the case
    where sklearn's default does not — otherwise closing the bypass changed nothing."""
    port = _extract_notebook_port(notebook)

    cases = [
        (["A", "A", "B", "B", "C", "C", "D", "D"], ["A", "A", "B", "B", "C", "E", "D", "D"]),
        (["A", "B", "C"], ["A", "B", "C"]),
        ([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0]),
        (["A"] * 9 + ["B"], ["A"] * 8 + ["B", "B"]),          # imbalanced tail
    ]
    for gold, pred in cases:
        want = score(gold, pred, labels=sorted(set(gold)))
        got = port(gold, pred)
        assert got["macro_f1"] == pytest.approx(want.macro_f1, abs=1e-12), (
            f"{notebook}: port disagrees with the registered scorer on {gold}/{pred}")
        assert got["accuracy"] == pytest.approx(want.accuracy, abs=1e-12)
        assert got["classes_averaged"] == want.classes_averaged
        assert got["classes_in_gold"] == want.classes_in_gold

    # And it must inherit the guard, not just the arithmetic.
    with pytest.raises(ValueError, match="NO gold examples"):
        port(["A", "A", "B", "B"], ["A", "A", "B", "B"], labels=["A", "B", "C"])

    # The divergence case: the port must NOT reproduce sklearn's deflation.
    gold = ["A", "A", "B", "B", "C", "C", "D", "D"]
    pred = ["A", "A", "B", "B", "C", "E", "D", "D"]
    assert port(gold, pred)["macro_f1"] - _sklearn_macro(gold, pred) > 0.05, (
        f"{notebook}: the port reproduces sklearn's absent-class deflation — the bypass "
        "is not actually closed")


# kaggle_tier1.py is EXCLUDED, deliberately and with a reason recorded rather than left as
# a silent gap. It still calls sklearn's f1_score directly (line ~816), which is 3ay's
# bypass. It is not fixed because NO TIER 1 RUN REMAINS: E4-A reports Tier 1 as not accepted
# at n=1, and E4b-A is superseded by E6's existing measurement, so nothing will execute this
# notebook again. Editing it would be a change no run verifies -- and an unverified edit to a
# scoring path is the defect class this file exists to guard, not a fix for it. If a Tier 1
# run is ever revived, this exclusion must be removed BEFORE it is launched.
SKLEARN_BYPASS_EXCLUDED = {
    "kaggle_tier1.py": "no Tier 1 run remains (E4-A, E4b-A); an unrunnable edit is "
                       "change without verification. Remove this entry before any "
                       "future Tier 1 run.",
}


def test_sklearn_bypass_exclusions_are_named_and_still_exist():
    """An exclusion must point at a real file and carry a reason."""
    for name, reason in SKLEARN_BYPASS_EXCLUDED.items():
        assert (ROOT / "notebooks" / name).exists(), (
            f"{name} is excluded from the sklearn-bypass check but no longer exists; "
            "remove the exclusion so it measures real debt")
        assert len(reason) > 40, f"{name}'s exclusion needs a substantive reason"


def test_notebooks_no_longer_call_sklearn_directly_for_reported_metrics():
    """The reported-metric helpers must route through the port, not f1_score.

    kaggle_tier1.py is excluded — see SKLEARN_BYPASS_EXCLUDED above for the reason.
    """
    for name in ("kaggle_tier0.py", "kaggle_tier0_dev.py"):
        assert name not in SKLEARN_BYPASS_EXCLUDED
        src = (ROOT / "notebooks" / name).read_text()
        assert 'def macro(y, pred): return f1_score(' not in src, (
            f"{name} still defines macro() directly on sklearn — 3ay's bypass")
        assert "_macro_report" in src, f"{name} is missing the ported scorer"


def _git_tracked_results() -> set[str]:
    """Basenames of the files under results/ that git tracks. Empty if git is unavailable."""
    import subprocess

    try:
        out = subprocess.run(["git", "ls-files", "results/"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    if out.returncode != 0:
        return set()
    return {Path(line).name for line in out.stdout.split() if line}


def test_grandfathered_list_only_shrinks():
    """Every grandfathered name must still exist — a stale entry hides a real gap.

    DECIDABILITY IS PER NAME, not per checkout, and that is a correction. This test used
    to skip only when NONE of the names were present, on the assumption that results/ is
    either fully checked out (locally) or fully absent (CI). That assumption broke the
    moment some result summaries were committed for REPORT.md to cite: a clean checkout
    then has *some* of them, which flipped the test out of its skip branch and made it
    report the still-gitignored names as stale. They are not stale. They are untracked.

    So: a name git TRACKS must be on disk, and its absence is real staleness, checked
    everywhere. A name git does NOT track cannot be judged from a clean checkout at all.
    The tracked half of the check now runs in CI, where the whole test used to skip.
    """
    tracked = _git_tracked_results()

    missing_tracked = sorted(n for n in GRANDFATHERED
                             if n in tracked and not (RESULTS / n).exists())
    assert not missing_tracked, (
        f"GRANDFATHERED names that git TRACKS but are not on disk: {missing_tracked}. "
        f"That is real staleness: remove them from the list, so it measures outstanding "
        f"debt rather than history."
    )

    undecidable = sorted(n for n in GRANDFATHERED
                         if n not in tracked and not (RESULTS / n).exists())
    if undecidable:
        pytest.skip(
            f"PARTIALLY VERIFIED, then SKIPPED. The tracked grandfathered names were "
            f"checked and are all present. {len(undecidable)} of {len(GRANDFATHERED)} "
            f"are gitignored (`results/*`) and absent, so a clean checkout cannot tell a "
            f"deliberately removed entry from one never fetched: {undecidable}. "
            f"Run locally with the artefacts on disk to decide these."
        )
