import os, hashlib
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path("/Users/sidharthchoudhary/Downloads/track/projects/reasonable-doubt")
load_dotenv(ROOT / ".env")
from huggingface_hub import HfApi, hf_hub_download
REPO = "sidharthjatt/reasonable-doubt-deberta-ledgar"
api = HfApi(token=os.environ["HF_TOKEN"])
print("files in repo:", sorted(api.list_repo_files(REPO)))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()
# Download ANONYMOUSLY — the Space pulls with no secret, so verify the public path.
out = Path("/private/tmp/claude-501/-Users-sidharthchoudhary-Downloads-track-projects-reasonable-doubt/58aec8e7-aebb-4f22-8685-8b64354ca243/scratchpad/dl")
out.mkdir(parents=True, exist_ok=True)
ok = True
for name in ("model.onnx", "config.json", "tokenizer.json", "spm.model"):
    p = hf_hub_download(REPO, name, local_dir=str(out), token=False)
    a, b = sha(p), sha(ROOT / "models" / "onnx_ce10ep_1_fp32" / name)
    same = a == b
    ok &= same
    print(f"  {name:22s} local={b[:16]}… remote={a[:16]}…  {'MATCH' if same else 'DIFFER'}")
print("ANONYMOUS DOWNLOAD REPRODUCES LOCAL BYTES:", ok)
