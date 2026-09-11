"""onnxruntime telemetry must be disabled before any session exists.

ORT's telemetry client crashes on macOS during process teardown —
`recursive_mutex lock failed` inside `Microsoft::Applications::Events::HttpClientManager`,
reached from `PosixTelemetry::Shutdown`. It fires after the work is done, so a script that
succeeded exits non-zero with a crash dialog. To anything reading exit codes — CI steps,
shell `&&` chains, a container's liveness — that is indistinguishable from failure.

The disable must precede the FIRST session: the telemetry provider is initialised with it,
and disabling afterwards leaves the registered shutdown hook in place.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.serve.config import ServiceConfig  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "src/ort_runtime.py",
    # STANDALONE BY REQUIREMENT: Kaggle runs this file without the repo on the path, so
    # it cannot import from src.* at all (test_training_script_is_standalone pins that).
    # It disables telemetry inline instead, which this test verifies below.
    "src/train/tier0_encoder.py",
}


def _py_files():
    for base in ("src", "scripts"):
        for p in (ROOT / base).rglob("*.py"):
            if "__pycache__" not in p.parts:
                yield p


def test_nothing_imports_onnxruntime_directly():
    """One import path, so "before any session" is true by construction rather than by
    everyone remembering. AST, not grep: a substring check would hit docstrings."""
    offenders = []
    for p in _py_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                if any(a.name == "onnxruntime" or a.name.startswith("onnxruntime.")
                       for a in node.names):
                    offenders.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module == "onnxruntime":
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        f"import onnxruntime directly at {offenders}. Use "
        f"src.ort_runtime.import_onnxruntime() so telemetry is disabled first.")


def test_import_onnxruntime_disables_telemetry_and_is_idempotent(monkeypatch):
    import src.ort_runtime as mod

    calls = []
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "disable_telemetry_events",
                        lambda *a, **k: calls.append(1))
    monkeypatch.setattr(mod, "_disabled", False)

    assert mod.telemetry_disabled() is False
    mod.import_onnxruntime()
    assert calls == [1] and mod.telemetry_disabled() is True
    mod.import_onnxruntime()
    assert calls == [1], "the disable must run exactly once"


@pytest.fixture
def config():
    from src.serve.config import ServiceConfig

    return ServiceConfig.load()


def test_the_disable_precedes_the_first_session(monkeypatch, config):
    """Ordering is the whole property: disabling after a session leaves the shutdown
    hook registered, which is the thing that crashes."""
    from tests.conftest import require_artifact

    require_artifact(config.tier0_model_dir,
                     "the served encoder; models/ is gitignored so CI has no artefact "
                     "to build a session from")
    import src.ort_runtime as mod

    ort = pytest.importorskip("onnxruntime")
    order = []
    orig_sess = ort.InferenceSession
    monkeypatch.setattr(ort, "disable_telemetry_events",
                        lambda *a, **k: order.append("disable"))

    class Spy(orig_sess):
        def __init__(self, *a, **k):
            order.append("session")
            super().__init__(*a, **k)

    monkeypatch.setattr(ort, "InferenceSession", Spy)
    monkeypatch.setattr(mod, "_disabled", False)

    from src.data.labels import load_labels
    from src.serve.tier0 import Tier0Encoder

    Tier0Encoder(model_dir=config.tier0_model_dir, labels=load_labels())
    assert order[0] == "disable", order
    assert "session" in order


def test_the_standalone_trainer_disables_telemetry_inline():
    """src/train/tier0_encoder.py is exempt from the single-import rule because Kaggle
    runs it without this repo. Exempt from the rule, NOT from the behaviour."""
    src = (ROOT / "src" / "train" / "tier0_encoder.py").read_text()
    assert "disable_telemetry_events()" in src
    assert "from src." not in src, "it must remain standalone"

    tree = ast.parse(src)
    disables = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "disable_telemetry_events"]
    # ONLY the ORT session builders. Matching bare "from_pretrained" also catches the
    # transformers calls that load the model for TRAINING, hundreds of lines earlier and
    # nothing to do with onnxruntime — that made this assertion fail on correct code.
    ort_builders = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id.startswith("ORT")
    ]
    assert disables, "no disable_telemetry_events() call"
    assert ort_builders, "no ORT* session builder found; has the export moved?"
    assert min(disables) < min(ort_builders), (
        f"telemetry disabled at line {min(disables)} but optimum builds a session at "
        f"{min(ort_builders)} — the disable must come first")
