#!/usr/bin/env bash
# Stage a build context (the repo .dockerignore excludes models/ by design, correctly, for
# the mount-based local image), build linux/amd64, push to Artifact Registry.
set -euo pipefail

PROJECT="${PROJECT:-reasonable-doubt-sid}"
REGION="${REGION:-asia-south1}"
REPO="${REPO:-services}"
IMAGE="${IMAGE:-reasonable-doubt}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${IMAGE}:${TAG}"

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp "${ROOT}/deploy/cloudrun/Dockerfile" "${ROOT}/deploy/cloudrun/verify_baked_model.py" "$CTX/"
cp -R "${ROOT}/src" "${ROOT}/configs" "$CTX/"
mkdir -p "$CTX/docs"
cp -R "${ROOT}/docs/figures" "$CTX/docs/"
mkdir -p "$CTX/models"
cp -R "${ROOT}/models/onnx_ce10ep_1_fp32" "$CTX/models/"
find "$CTX" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "building ${URI}"
docker build --platform linux/amd64 -t "$URI" "$CTX"
docker push "$URI"
echo "$URI"
