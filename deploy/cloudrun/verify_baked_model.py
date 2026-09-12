"""Verify the BAKED model bytes at image build time. Exits non-zero on any mismatch.

The Space image downloads the artefact from the Hub at build time and hashes it. This
image copies it from the build context instead, so there is no Hub dependency at build OR
at run: a cold start loads the file that is already in the layer. The hash check is the
same one and is kept for the same reason — the startup canary asks "does this behave?",
which is a different question from "are these the bytes we published?".
"""
import hashlib
import sys
from pathlib import Path

EXPECTED = {
    "model.onnx": "e70fb095ee8ddd82c9449d6da0006b4fa1c7a92bb8547d5538f869c445010200",
    "config.json": "0b84b3be80278cd6e622ebeb0302fe186997225a605b98b6709b48cd4cb4f64e",
    "tokenizer.json": "133b1071a455bb96ec0d7aa1242e616cc42ec29652fded524fb6883832afa903",
    "tokenizer_config.json": "f3fe8bea48e9f8d6884a74d7aecafb3ae9ed869bd5f6a8561d63cc5f45d15a73",
    "special_tokens_map.json": "b2f1b2f15f29a6b6d9d6ea4eca1675d2c231a71477f151d48f79cc83a625ba21",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "added_tokens.json": "dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/models/onnx_ce10ep_1_fp32")
    bad, missing = [], []
    for name, want in EXPECTED.items():
        p = d / name
        if not p.exists():
            missing.append(name)
            continue
        got = sha256(p)
        print(f"  {name:26s} {got[:16]}… {'ok' if got == want else 'MISMATCH'}", flush=True)
        if got != want:
            bad.append((name, want, got))
    if missing or bad:
        print("\nREFUSING TO BUILD — the baked artefact is not the published one:",
              file=sys.stderr)
        for name in missing:
            print(f"  MISSING: {name}", file=sys.stderr)
        for name, want, got in bad:
            print(f"  {name}\n    expected {want}\n    got      {got}", file=sys.stderr)
        return 1
    print(f"all {len(EXPECTED)} baked files match the published sha256 -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
