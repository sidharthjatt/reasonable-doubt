"""Build and commit the frozen evaluation manifests.

    python scripts/build_manifests.py

Rebuilding must be idempotent: with the same seeds and the same dataset, the written
files are byte-identical. This script refuses to overwrite a manifest whose contents
would change unless ``--force`` is given, because an unnoticed change to a committed
manifest invalidates every result already recorded against it.

No network beyond the HuggingFace dataset download. No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import DEFAULT_LABELS_PATH, LabelNormalizer, save_labels  # noqa: E402
from src.data.loading import label_names, load_ledgar  # noqa: E402
from src.data.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_DIR,
    MANIFEST_SPECS,
    build_manifest,
    verify_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    ap.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a manifest whose contents would change.",
    )
    args = ap.parse_args()

    print("Loading LEDGAR (coastalcph/lex_glue:ledgar)…")
    ds = load_ledgar()
    names = label_names(ds)

    # Fails loudly if two labels fold to the same normalization key.
    LabelNormalizer(names)
    path = save_labels(names, args.labels_path)
    print(f"  wrote {len(names)} canonical labels -> {path}")

    changed = False
    for name in MANIFEST_SPECS:
        manifest = build_manifest(name, ds)
        verify_manifest(manifest, ds)  # self-check before writing

        target = manifest.path(args.manifest_dir)
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing == json.loads(json.dumps(asdict(manifest))):
                print(f"  {name}: unchanged ({manifest.text_sha256[:12]}…)")
                continue
            if not args.force:
                print(
                    f"\nREFUSING to overwrite {target}: regenerating produced different\n"
                    "contents. A committed manifest that changes invalidates every\n"
                    "result recorded against it. Re-run with --force only if you intend\n"
                    "to retire those results.",
                    file=sys.stderr,
                )
                return 2
            changed = True

        manifest.save(args.manifest_dir)
        absent = manifest.num_labels - manifest.classes_represented
        print(
            f"  {name}: n={manifest.n} split={manifest.split} seed={manifest.seed} "
            f"sha256={manifest.text_sha256[:12]}…\n"
            f"      classes represented {manifest.classes_represented}/"
            f"{manifest.num_labels} ({absent} absent), "
            f"per-class min={manifest.min_class_count} max={manifest.max_class_count}"
        )

    print("\nDone." + (" (some manifests were overwritten)" if changed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
