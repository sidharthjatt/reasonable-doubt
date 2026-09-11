# Reasonable Doubt — Tier 0 INT8 -> Claude Sonnet 5 (E4b-A). CPU ONLY.
#
# There is no GPU path and no CUDA: Tier 0 is an ONNX INT8 encoder that runs on CPU, and
# Tier 2 is an API call. torch is NOT installed — the service loads ONNX through
# onnxruntime and never touches the training stack.
#
# THE MODEL IS NOT IN THIS IMAGE — it is PULLED AT BUILD TIME from the public model repo
# sidharthjatt/reasonable-doubt-deberta-ledgar and verified against recorded sha256s
# (download_model.py). No token, no secret: the repo is public, so this Space holds none.
#
# Locally the same image takes a mount instead:
#   docker run --rm -p 8000:8000 \
#     -v "$PWD/models:/models:ro" -e TIER0_MODEL_DIR=/models/onnx_ce10ep_1_fp32 \
#     reasonable-doubt:local
#
# PRECISION IS FP32 (3bg). INT8 is qualified on no Linux target: it diverges between
# macOS-arm64 and Linux-aarch64 and collapses to chance on non-VNNI x86, so this image
# serves the FP32 ONNX export. TIER0_PRECISION selects which precision's threshold and
# canary numbers apply, and the config refuses a threshold from the other precision.
#
# THE STARTUP CANARY RUNS INSIDE THE CONTAINER, and that is the point of running it at
# all: Kaggle's non-VNNI x86 scored these exact weights at macro-F1 0.0037 — chance —
# with no error raised, and a container is precisely where you cannot see that happening.
# create_app() classifies 200 frozen rows before serving and refuses to start below the
# floor, so a mis-built or wrong-ISA image fails at boot instead of serving chance-level
# labels with plausible confidences.

FROM python:3.11-slim

# onnxruntime 1.29.0 is PINNED to the version that quantised the host baseline and every
# artefact since (3bd/3be). INT8 artefacts produced or run under a different quantiser
# toolchain are not comparable, and this image is where the deployed one runs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TIER0_MODEL_DIR=/app/models/onnx_ce10ep_1_fp32 \
    TIER0_PRECISION=fp32 \
    HF_HOME=/tmp/hf

WORKDIR /app

# Runtime dependencies only. Deliberately NOT from pyproject.toml: that file carries the
# local research stack (torch-free but including datasets, matplotlib, mlx extras) and an
# image that installs it would be several hundred MB of things the service never imports.
RUN pip install --no-cache-dir \
      "onnxruntime==1.29.0" \
      "transformers>=4.44,<4.58" \
      "sentencepiece>=0.2" \
      "protobuf>=4.25" \
      "numpy>=1.26" \
      "pyyaml>=6.0" \
      "pydantic>=2.7" \
      "anthropic>=0.40" \
      "python-dotenv>=1.0" \
      "fastapi>=0.111" \
      "uvicorn>=0.30" \
      "datasets>=2.19" \
      "huggingface-hub>=0.34,<1.0"

# Source and the committed configs. configs/ carries serve.yaml, costs.yaml (every price,
# hard rule 5), router_threshold.json (dev_2000-calibrated, hard rule 1) and
# canary_200.json (the frozen startup row set) — all tracked, all required at boot.
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY download_model.py /app/download_model.py

# Non-root. The service writes only to the response cache and the ledger, both of which
# are mounted read-write when wanted and absent otherwise.
RUN useradd --create-home --uid 10001 serve \
 && mkdir -p /app/cache /app/results /app/models /tmp/hf \
 && chown -R serve:serve /app /tmp/hf
USER serve

# PULL AND VERIFY THE WEIGHTS AT BUILD TIME, as the `serve` user so the files are owned
# by the account that reads them. A hash mismatch fails the build — the Space never comes
# up serving an artefact nobody recorded.
RUN python /app/download_model.py

# The LEDGAR rows the startup canary scores. Cached into the image so the canary is not
# at the mercy of a dataset fetch on every cold start.
RUN python -c "from src.data.loading import load_ledgar; load_ledgar()"

# RUNTIME-WRITABLE PATHS, set as root at the very end. Hugging Face may run the container
# under a different uid than the one built here; everything the service reads is
# world-readable by default, but the dataset cache and response cache are WRITTEN to, and
# a uid mismatch there would fail at startup for a reason that looks nothing like its cause.
USER root
RUN chmod -R a+rwX /tmp/hf /app/cache /app/results \
 && chmod -R a+rX /app/models /app/src /app/configs
USER serve

EXPOSE 8000

# The canary runs during create_app(), i.e. before uvicorn binds. A failure exits
# non-zero and the container does not come up.
CMD ["uvicorn", "src.serve.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
