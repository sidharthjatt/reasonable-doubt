"""The ONE way this project imports onnxruntime.

WHY THIS MODULE EXISTS. onnxruntime ships a telemetry client that runs at process exit,
and on macOS its shutdown path crashes:

    libc++abi: terminating due to uncaught exception of type std::system_error:
    recursive_mutex lock failed: Invalid argument

The crashed thread is inside `Microsoft::Applications::Events::HttpClientManager`, reached
from `PosixTelemetry::Shutdown`. It fires AFTER the work is finished — `scripts/` writes
its artefacts, prints its result, and then the process dies while tearing down telemetry,
raising a "Python quit unexpectedly" dialog and a non-zero exit.

THAT IS NOT COSMETIC. A non-zero exit from a script that succeeded is
indistinguishable, to any caller that checks the exit code, from a script that failed. CI
steps, shell `&&` chains and the container's own liveness all read exit codes. A crash
that happens to come after the useful work is still a crash.

`disable_telemetry_events()` MUST be called BEFORE the first InferenceSession is
constructed — the telemetry provider is initialised with the first session, and disabling
it afterwards leaves the already-registered shutdown hook in place. Routing every import
through `import_onnxruntime()` is what makes "before any session" true by construction
rather than by everyone remembering.

SIDE BENEFIT, and it is not the reason: the service stops sending usage telemetry.
"""

from __future__ import annotations

__all__ = ["import_onnxruntime", "telemetry_disabled"]

_disabled = False


def import_onnxruntime():
    """Return the onnxruntime module with telemetry disabled.

    Idempotent and safe to call from anywhere; the disable runs exactly once, on the
    first call, which is necessarily before that caller creates a session.
    """
    global _disabled
    import onnxruntime as ort

    if not _disabled:
        # Not wrapped in a bare except: if this call starts failing we want to know,
        # because the failure mode it prevents is a crash at exit that looks like a
        # failed run. A missing attribute on some future build is worth surfacing.
        ort.disable_telemetry_events()
        _disabled = True
    return ort


def telemetry_disabled() -> bool:
    """Whether this process has disabled ORT telemetry. Reported in /health."""
    return _disabled
