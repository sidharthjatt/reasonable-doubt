"""Pull the served artefact from the public model repo and VERIFY THE BYTES.

Runs before uvicorn. No token: the model repo is public and this Space holds no secret.

WHY A HASH CHECK AND NOT JUST A DOWNLOAD. The startup canary already refuses a model that
cannot classify, but it runs on 200 rows and answers "does this behave?" — not "is this
the artefact we published?". A silently different revision that still scores well would
pass the canary and serve something nobody recorded. These two checks are complementary
and neither replaces the other.

Mismatch EXITS NON-ZERO. The Space fails to build rather than serving unverified weights.
"""
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "sidharthjatt/reasonable-doubt-deberta-ledgar"

# Recorded at upload time from the local artefact, and re-verified against an anonymous
# download before this Space existed.
EXPECTED = {
    "model.onnx": "e70fb095ee8ddd82c9449d6da0006b4fa1c7a92bb8547d5538f869c445010200",
    "config.json": "0b84b3be80278cd6e622ebeb0302fe186997225a605b98b6709b48cd4cb4f64e",
    "tokenizer.json": "133b1071a455bb96ec0d7aa1242e616cc42ec29652fded524fb6883832afa903",
    "tokenizer_config.json": "f3fe8bea48e9f8d6884a74d7aecafb3ae9ed869bd5f6a8561d63cc5f45d15a73",
    "special_tokens_map.json": "b2f1b2f15f29a6b6d9d6ea4eca1675d2c231a71477f151d48f79cc83a625ba21",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "added_tokens.json": "dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f",
}

# The BASENAME matters: ServiceConfig cross-checks it against the artefact the router
# threshold was calibrated on and refuses to start when they differ.
DEST = Path(os.environ.get("TIER0_MODEL_DIR", "/app/models/onnx_ce10ep_1_fp32"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    bad = []
    for name, want in EXPECTED.items():
        print(f"  fetching {name} …", flush=True)
        p = Path(hf_hub_download(REPO, name, local_dir=str(DEST), token=False))
        got = sha256(p)
        status = "ok" if got == want else "MISMATCH"
        print(f"    sha256 {got[:16]}… {status}", flush=True)
        if got != want:
            bad.append((name, want, got))
    if bad:
        print("\nREFUSING TO START — downloaded bytes are not the published artefact:",
              file=sys.stderr)
        for name, want, got in bad:
            print(f"  {name}\n    expected {want}\n    got      {got}", file=sys.stderr)
        return 1
    print(f"all {len(EXPECTED)} files verified against the recorded sha256 -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
