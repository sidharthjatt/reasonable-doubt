"""The Kaggle notebooks must COMPILE, or none of their guards ever run.

kaggle_tier1.py once shipped with `fp16` passed twice to the same `dict()` call.
That is a compile-time error, so nothing in the cell executed: not the version gate,
not PREFLIGHT, not the precision/device gate — every check built to catch the previous
three failures sat downstream of a file that never parsed. The checks are only ever as
good as the file reaching them, and nothing in this repo compiled the notebooks.

Why `compile(..., "exec")` and not `ast.parse`: they do not agree. Duplicate keyword
arguments are rejected during symbol-table construction, not during parsing, so
`ast.parse("dict(a=1, a=2)")` SUCCEEDS while `compile(...)` raises. The original
check used `ast.parse` and reported the broken file as passing — the same
'verified a proxy for the property you care about' error recorded in PREREGISTRATION
3o. `test_ast_parse_would_not_have_caught_it` below pins that distinction so nobody
'simplifies' this back to ast.parse.

CELL 1 of each notebook is IPython (`!pip install ...`) and is not valid Python. The
notebooks carry an explicit `CELL 2 of 2` banner, which is the same split the editing
tooling uses, so it is reused here rather than re-invented.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

CELL2_BANNER = "# ============================ CELL 2 of 2"

# Files split into an IPython CELL 1 and a Python CELL 2; only CELL 2 is compiled.
TWO_CELL = ["kaggle_tier0.py", "kaggle_tier1.py"]
# Probes are plain Python end to end and must compile whole.
WHOLE_FILE = ["kaggle_probe_qlora.py", "kaggle_probe_tier0.py",
              "kaggle_probe_persist.py"]


def _cell2(path: Path) -> str:
    src = path.read_text()
    assert CELL2_BANNER in src, (
        f"{path.name} has no {CELL2_BANNER!r} banner. The two-cell protocol is what the "
        f"run instructions in notebooks/RUNNING.md depend on; if the banner moved or was "
        f"renamed, this test is compiling the wrong region and RUNNING.md is now wrong too."
    )
    assert src.count(CELL2_BANNER) == 1, f"{path.name} has multiple CELL 2 banners"
    return src[src.index(CELL2_BANNER):]


@pytest.mark.parametrize("name", TWO_CELL)
def test_two_cell_notebook_cell2_compiles(name: str) -> None:
    path = NOTEBOOKS / name
    assert path.exists(), f"{name} is missing from notebooks/"
    try:
        compile(_cell2(path), f"{name}::CELL2", "exec")
    except SyntaxError as exc:  # pragma: no cover - the message is the point
        pytest.fail(
            f"{name} CELL 2 does not compile: {exc.msg} (line {exc.lineno} of the cell)\n"
            f"    {(exc.text or '').rstrip()}\n"
            f"Nothing in the cell runs until this is fixed — not the version gate, not "
            f"PREFLIGHT, not the precision gate."
        )


@pytest.mark.parametrize("name", WHOLE_FILE)
def test_probe_compiles_whole(name: str) -> None:
    path = NOTEBOOKS / name
    assert path.exists(), f"{name} is missing from notebooks/"
    try:
        compile(path.read_text(), name, "exec")
    except SyntaxError as exc:  # pragma: no cover
        pytest.fail(
            f"{name} does not compile: {exc.msg} (line {exc.lineno})\n"
            f"    {(exc.text or '').rstrip()}\n"
            f"A probe that cannot compile cannot gate the run it exists to gate."
        )


@pytest.mark.parametrize("name", TWO_CELL)
def test_cell1_is_ipython_and_is_deliberately_excluded(name: str) -> None:
    """CELL 1 is excluded because it is IPython, not because compiling is optional.

    If a notebook's CELL 1 ever becomes valid Python, the exclusion above is no longer
    justified and CELL 2 should stop being carved out. This test states the reason, so
    the carve-out cannot quietly outlive it.
    """
    src = (NOTEBOOKS / name).read_text()
    cell1 = src[: src.index(CELL2_BANNER)]
    assert "!pip" in cell1, (
        f"{name} CELL 1 no longer contains a `!pip` magic. The only reason CELL 1 is not "
        f"compiled is that it is IPython. If that is no longer true, compile the whole file."
    )
    with pytest.raises(SyntaxError):
        compile(cell1, f"{name}::CELL1", "exec")


def test_ast_parse_would_not_have_caught_it() -> None:
    """Pin the reason this suite uses compile() rather than ast.parse().

    This is not a test of the standard library for its own sake: it is the evidence that
    the weaker instrument returns a false pass on the exact defect that shipped. If a
    future Python makes ast.parse strict here, this test fails and the docstring above
    can be simplified — a deliberate decision, not an accident.
    """
    duplicate_kwarg = "f = dict(a=1, a=2)"
    ast.parse(duplicate_kwarg)  # succeeds — this is the false pass
    with pytest.raises(SyntaxError, match="keyword argument repeated"):
        compile(duplicate_kwarg, "<duplicate>", "exec")
