"""Create and populate the public model repo. Token comes from .env and is never logged."""
import os, sys, hashlib
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path("/Users/sidharthchoudhary/Downloads/track/projects/reasonable-doubt")
load_dotenv(ROOT / ".env")
TOKEN = os.environ["HF_TOKEN"]
REPO = "sidharthjatt/reasonable-doubt-deberta-ledgar"
SRC = ROOT / "models" / "onnx_ce10ep_1_fp32"

from huggingface_hub import HfApi
api = HfApi(token=TOKEN)

api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
print(f"repo ready: {REPO} (public)")

staging = ROOT / "deploy" / "_model_upload"
staging.mkdir(parents=True, exist_ok=True)
for f in sorted(SRC.iterdir()):
    if f.is_file():
        (staging / f.name).write_bytes(f.read_bytes())
(staging / "README.md").write_text((ROOT / "deploy" / "model_card" / "README.md").read_text())
(staging / "router_threshold_fp32.json").write_text(
    (ROOT / "configs" / "router_threshold_fp32.json").read_text())
(staging / "labels.json").write_text((ROOT / "configs" / "labels.json").read_text())

print("uploading", sum(f.stat().st_size for f in staging.iterdir()) / 1e6, "MB")
api.upload_folder(repo_id=REPO, repo_type="model", folder_path=str(staging),
                  commit_message="Serve E1b seed 1 FP32 ONNX + tokenizer + router threshold")
print("UPLOAD DONE")
