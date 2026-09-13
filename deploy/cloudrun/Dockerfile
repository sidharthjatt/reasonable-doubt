# Reasonable Doubt on Cloud Run — Tier 0 FP32 encoder, no escalation, no secrets.
#
# DIFFERS FROM THE REPO Dockerfile IN ONE WAY THAT MATTERS: the model is BAKED IN, not
# mounted and not downloaded at startup. Cloud Run has no volume to mount from, and
# fetching 715 MB from the Hub on every cold start would add both latency and a failure
# mode (the Hub being slow or down would take the service down with it). The bytes are
# verified against the published sha256 AT BUILD TIME instead, so a wrong artefact fails
# the build rather than reaching production.
#
# NO API KEY, NO SECRET, NO ESCALATION. configs/serve.yaml has tier2.enabled: false
# (§3bm), and ANTHROPIC_API_KEY is not set here, not passed at deploy, and not read from
# any secret store. The service cannot spend money.
#
# THE STARTUP CANARY IS THE POINT OF THIS DEPLOY. Every x86 number in this project before
# §3bn came from an EMULATED amd64 container on Apple Silicon.
#
# It runs on a background thread (§3bp) so the port binds in seconds rather than ~150.
# /classify returns 503 until the canary passes and FOREVER if it fails; the qualification
# check is enforced in Cascade.classify, not in the endpoint, so no caller can route
# around it.

FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TIER0_MODEL_DIR=/app/models/onnx_ce10ep_1_fp32 \
    TIER0_PRECISION=fp32 \
    HF_HOME=/tmp/hf \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# onnxruntime PINNED to the version every artefact was produced under (§3bd/§3be).
# No huggingface-hub download path is needed at runtime; datasets pulls what it needs.
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

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY verify_baked_model.py /app/verify_baked_model.py

# The served artefact, baked. 715 MB, and it is the whole reason this image is large.
COPY models/onnx_ce10ep_1_fp32/ /app/models/onnx_ce10ep_1_fp32/

# VERIFY BEFORE ANYTHING ELSE DEPENDS ON IT. A mismatch fails the build.
RUN python /app/verify_baked_model.py /app/models/onnx_ce10ep_1_fp32

# The LEDGAR rows the canary scores, baked so a cold start does no network I/O at all.
RUN python -c "from src.data.loading import load_ledgar; load_ledgar()"

RUN useradd --create-home --uid 10001 serve \
 && mkdir -p /app/cache /app/results \
 && chown -R serve:serve /app /tmp/hf \
 && chmod -R a+rwX /tmp/hf /app/cache /app/results \
 && chmod -R a+rX /app/models /app/src /app/configs
USER serve

# CLOUD RUN INJECTS $PORT and routes to it; a hardcoded port fails the startup probe with
# a message that says nothing about the cause. Shell form so the variable expands.
CMD exec uvicorn src.serve.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}
