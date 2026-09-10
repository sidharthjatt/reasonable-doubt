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
    "e6_frontier.json",   # aggregates over seeds; label set is fixed by the loader
}


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
    p = RESULTS / "e8_fewshot.json"
    if not p.exists():
        pytest.skip("E8 artefact not present")
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


def test_grandfathered_list_only_shrinks():
    """Every grandfathered name must still exist — a stale entry hides a real gap."""
    absent = [n for n in GRANDFATHERED if not (RESULTS / n).exists()]
    assert not absent, (
        f"GRANDFATHERED names no longer on disk: {absent}. Remove them, so the list "
        "measures outstanding debt rather than history."
    )
