"""Secret redaction, enforced at the point of WRITE.

Redacting when something is displayed is too late: by then the secret is already on
disk in a cache entry, a ledger line, or a results file, and disk is what gets
committed. So the guard runs where bytes are persisted — :meth:`ResponseCache.put`
and :meth:`SpendLedger._append` both call :func:`assert_no_secrets` on the exact
payload they are about to serialize, and raise rather than write.

Two independent detectors, because each catches what the other misses:

* **Value matching** — the live values of secret-looking environment variables. Exact,
  and catches a key that has no recognisable prefix.
* **Pattern matching** — known provider key shapes (``sk-ant-``, ``gsk_``, ``hf_``).
  Catches a secret that is not in this process's environment, e.g. one pasted into a
  prompt or echoed back by a provider.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

__all__ = [
    "KEY_PATTERNS",
    "SECRET_ENV_SUFFIXES",
    "SecretLeakError",
    "assert_no_secrets",
    "redact",
    "secret_values",
]

# Provider key shapes. Kept deliberately loose on the tail so a rotated key format
# still trips the guard.
KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"hf_[A-Za-z0-9]{8,}"),
    re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),  # Google/Gemini
)

SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS")

REDACTED = "<REDACTED>"

# Below this length a value is too short to be a real key and matching it would
# redact ordinary text (an empty or placeholder env var, typically).
MIN_SECRET_LEN = 12


class SecretLeakError(RuntimeError):
    """A payload about to be persisted contained secret material. Nothing was written."""


def secret_values(env: dict[str, str] | None = None) -> list[str]:
    """Live values of secret-looking environment variables, longest first.

    Longest-first matters for redaction: a key that is a prefix of another must not
    partially mask it.
    """
    env = os.environ if env is None else env
    values = {
        value
        for name, value in env.items()
        if any(name.upper().endswith(s) for s in SECRET_ENV_SUFFIXES)
        and isinstance(value, str)
        and len(value) >= MIN_SECRET_LEN
    }
    return sorted(values, key=len, reverse=True)


def redact(text: str, env: dict[str, str] | None = None) -> str:
    """Replace any secret material in ``text`` with ``<REDACTED>``.

    Use on anything headed for a log line or an error message. It is a safety net, not
    a licence to pass secrets around: the primary control is never putting them in the
    payload at all.
    """
    if not text:
        return text
    for value in secret_values(env):
        text = text.replace(value, REDACTED)
    for pattern in KEY_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _iter_strings(item)


def assert_no_secrets(payload: Any, *, context: str, env: dict[str, str] | None = None) -> None:
    """Raise :class:`SecretLeakError` if ``payload`` contains secret material.

    Called immediately before serializing to disk. The error deliberately does NOT
    include the offending value — reporting a leak must not itself leak.

    Args:
        payload: The object about to be written. Scanned recursively; non-JSON types
            are stringified so nothing escapes by being an unusual type.
        context: Where the write was happening, for the error message.
    """
    values = secret_values(env)
    strings = list(_iter_strings(payload))
    if not isinstance(payload, (str, dict, list, tuple, set)):
        strings.append(str(payload))
    else:
        # Catch secrets hiding inside non-string leaves (custom objects, bytes reprs).
        try:
            strings.append(json.dumps(payload, default=str))
        except (TypeError, ValueError):
            strings.append(str(payload))

    for text in strings:
        for value in values:
            if value in text:
                raise SecretLeakError(
                    f"REFUSING TO WRITE — {context} contained the value of a secret "
                    "environment variable. Nothing was written. The offending value is "
                    "not reproduced here."
                )
        for pattern in KEY_PATTERNS:
            if pattern.search(text):
                raise SecretLeakError(
                    f"REFUSING TO WRITE — {context} contained something shaped like an "
                    f"API key (prefix {pattern.pattern.split('[')[0]}…). Nothing was "
                    "written. The offending value is not reproduced here."
                )
