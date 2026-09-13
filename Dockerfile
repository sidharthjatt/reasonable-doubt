# Reasonable Doubt — Tier 0 INT8 -> Claude Sonnet 5 (E4b-A). CPU ONLY.
#
# There is no GPU path and no CUDA: Tier 0 is an ONNX INT8 encoder that runs on CPU, and
# Tier 2 is an API call. torch is NOT installed — the service loads ONNX through
# onnxruntime and never touches the training stack.
#
# THE MODEL IS NOT IN THIS IMAGE. models/ is gitignored, 715 MB for the served FP32
# artefact, and the served one HAS changed once already (E1b seed 1 replaced E1's, §3bc).
# Baking it would make the image the source of truth for something configs/serve.yaml
# decides. Mount it and point TIER0_MODEL_DIR at the mount:
#
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
#
# SINCE §3bp IT RUNS IN THE BACKGROUND while the port binds immediately. The property is
# unchanged: no prediction is returned until the canary passes, and a FAILED canary
# refuses every request for the life of the process. What changed is that a cold client
# gets an instant 503 saying "warming up" instead of a two-minute hang, and a failed
# canary leaves a live /health to inspect instead of a crash-looping container.

FROM python:3.11-slim

# onnxruntime 1.29.0 is PINNED to the version that quantised the host baseline and every
# artefact since (3bd/3be). INT8 artefacts produced or run under a different quantiser
# toolchain are not comparable, and this image is where the deployed one runs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TIER0_MODEL_DIR=/models/onnx_ce10ep_1_fp32 \
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
      "datasets>=2.19"

# Source and the committed configs. configs/ carries serve.yaml, costs.yaml (every price,
# hard rule 5), router_threshold.json (dev_2000-calibrated, hard rule 1) and
# canary_200.json (the frozen startup row set) — all tracked, all required at boot.
COPY src/ /app/src/
COPY configs/ /app/configs/
# The figures the demo page shows. Already generated for REPORT.md, served as-is.
COPY docs/figures/ /app/docs/figures/

# Non-root. The service writes only to the response cache and the ledger, both of which
# are mounted read-write when wanted and absent otherwise.
RUN useradd --create-home --uid 10001 serve \
 && mkdir -p /app/cache /app/results /tmp/hf \
 && chown -R serve:serve /app /tmp/hf
USER serve

EXPOSE 8000

# The canary runs during create_app(), i.e. before uvicorn binds. A failure exits
# non-zero and the container does not come up.
CMD ["uvicorn", "src.serve.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
