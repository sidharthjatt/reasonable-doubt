"""Build the frozen startup-canary row set (TRAIN only, no eval role touched).

    python scripts/build_canary.py [--force]

Writes configs/canary_200.json. Committed, like every other frozen row set, so the
service checks the same rows on every host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loading import load_ledgar  # noqa: E402
from src.serve.canary import CANARY_PATH, build_canary, verify_canary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing canary file")
    args = ap.parse_args()

    if CANARY_PATH.exists() and not args.force:
        print(f"{CANARY_PATH} exists; refusing to overwrite a frozen row set "
              f"without --force")
        return 1

    ds = load_ledgar()
    canary = build_canary(ds)
    verify_canary(canary, ds)
    CANARY_PATH.write_text(json.dumps(canary.as_dict(), indent=2) + "\n")

    print(f"wrote {CANARY_PATH}")
    print(f"  split={canary.split} n={canary.n} seed={canary.seed}")
    print(f"  excluded: {canary.excluded_manifests}")
    print(f"  text_sha256={canary.text_sha256[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
