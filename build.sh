#!/usr/bin/env bash
# ==============================================================================
# build.sh — Build the Docker image for unitree_IL_lerobot (π0.5 / pi05)
#
# Usage:
#   ./build.sh               # normal build (uses Docker layer cache)
#   ./build.sh --no-cache    # full rebuild from scratch
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="unitree-lerobot-pi05"
IMAGE_TAG="latest"

NO_CACHE_FLAG=""
if [[ "${1:-}" == "--no-cache" ]]; then
    NO_CACHE_FLAG="--no-cache"
    echo ">>> Building with --no-cache"
fi

echo "=============================================="
echo "  Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Target policy: π0.5 (pi05)"
echo "  Context: ${SCRIPT_DIR}"
echo "=============================================="

DOCKER_BUILDKIT=1 docker build \
    ${NO_CACHE_FLAG} \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo ""
echo "=============================================="
echo "  ✓ Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Run with:  ./docker_run.sh"
echo "=============================================="
