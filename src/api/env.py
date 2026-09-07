"""Load `.env` from the project root.

`.env` is gitignored and holds live credentials. Nothing in this module reads, logs or
returns a key value — it only populates ``os.environ`` so provider SDKs can find them.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PROJECT_ROOT", "load_env"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | str | None = None, *, override: bool = False) -> list[str]:
    """Load the project's `.env`. Returns the NAMES of variables set (never values)."""
    from dotenv import dotenv_values

    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return []
    loaded = []
    for name, value in dotenv_values(env_path).items():
        if value and (override or not os.environ.get(name)):
            os.environ[name] = value
            loaded.append(name)
    return loaded
