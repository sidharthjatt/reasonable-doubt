"""Create and populate the public Docker Space. No secrets are set on it."""
import os, shutil
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path("/Users/sidharthchoudhary/Downloads/track/projects/reasonable-doubt")
load_dotenv(ROOT / ".env")
from huggingface_hub import HfApi

REPO = "sidharthjatt/reasonable-doubt"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="space", space_sdk="docker", private=False,
                exist_ok=True)
print(f"space ready: {REPO} (public, docker)")

stage = ROOT / "deploy" / "_space_upload"
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)
for f in ("Dockerfile", "README.md", "download_model.py"):
    shutil.copy(ROOT / "deploy" / "space" / f, stage / f)
for d in ("src", "configs"):
    shutil.copytree(ROOT / d, stage / d,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
print("files:", sorted(p.name for p in stage.iterdir()))

api.upload_folder(repo_id=REPO, repo_type="space", folder_path=str(stage),
                  commit_message="Tier 0 FP32 clause classifier: /classify, /health, demo page")
print("SPACE UPLOAD DONE")
print("secrets on this space:", api.list_space_secrets(REPO)
      if hasattr(api, "list_space_secrets") else "(not queried)")
