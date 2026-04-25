#!/usr/bin/env bash
# ==============================================================================
# docker_run.sh — Run the unitree-lerobot π0.5 container with persistent volumes
#
# Persistent volume mounts:
#   Source code            → /workspace              (live editing)
#   LeRobot datasets       → /workspace/lerobot_data
#   Training outputs       → .../outputs_pi05_single_arm_dual_cam  (checkpoints)
#   Open-loop eval results → /workspace/open_loop_eval_results
#   Datasets (3-camera)    → /workspace/datasets
#   Experiment logs        → /workspace/experiment_logs
#   HuggingFace cache      → ~/.cache/huggingface   (model downloads persist)
#   Wandb config           → ~/.config/wandb        (API key persists)
#
# Usage:
#   ./docker_run.sh                                 # interactive bash
#   ./docker_run.sh python -s src/lerobot/scripts/lerobot_train.py ...
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="unitree-lerobot-pi05"
IMAGE_TAG="latest"
CONTAINER_NAME="unitree-lerobot-pi05-dev"

# ---- Host paths (edit these if your layout differs) ----
PROJECT_DIR="${SCRIPT_DIR}"
LEROBOT_DATA_DIR="${SCRIPT_DIR}/lerobot_data"
LEROBOT_DIR="${SCRIPT_DIR}/unitree_lerobot/lerobot"
WANDB_OUTPUTS_DIR="${SCRIPT_DIR}/unitree_lerobot/lerobot/wandb"
EVAL_RESULTS_DIR="${SCRIPT_DIR}/open_loop_eval_results"
DATASETS_DIR="${SCRIPT_DIR}/datasets"
EXPERIMENT_LOGS_DIR="$(dirname "${SCRIPT_DIR}")/experiment_logs"
RUN_SCRIPT_PATH="${SCRIPT_DIR}/run.sh"

HF_CACHE_DIR="${HOME}/drive2/.cache/huggingface"
WANDB_DIR="${HOME}/.config/wandb"
WANDB_CACHE="${HOME}/drive2/deepansh/.cache/wandb"

# ---- Ensure host directories exist ----
mkdir -p "${LEROBOT_DATA_DIR}"
# mkdir -p "${LEROBOT_DIR}"
mkdir -p "${WANDB_OUTPUTS_DIR}"
mkdir -p "${EVAL_RESULTS_DIR}"
mkdir -p "${DATASETS_DIR}"
mkdir -p "${HF_CACHE_DIR}"
# mkdir -p "${WANDB_DIR}"
# mkdir -p "${WANDB_CACHE}"
# [[ -f "${RUN_SCRIPT_PATH}" ]] || touch "${RUN_SCRIPT_PATH}"

# ---- Stop any previous container with the same name ----
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ">>> Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

echo "=============================================="
echo "  Running: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Container name: ${CONTAINER_NAME}"
echo "=============================================="

# docker run -it \
#     --name "${CONTAINER_NAME}" \
#     --gpus all \
#     --runtime=nvidia \
#     --shm-size=16g \
#     --net=host \
#     --privileged \
#     \
#     `# ---- Project source (live-mount for development) ----` \
#     -v "${PROJECT_DIR}:/workspace" \
#     -v "${RUN_SCRIPT_PATH}:/workspace/run.sh" \
#     \
#     `# ---- Persistent data volumes ----` \
#     -v "${LEROBOT_DATA_DIR}:/workspace/lerobot_data" \
#     -v "${LEROBOT_DIR}:/workspace/unitree_lerobot/lerobot" \
#     -v "${EVAL_RESULTS_DIR}:/workspace/open_loop_eval_results" \
#     -v "${DATASETS_DIR}:/workspace/datasets" \
#     -v "${EXPERIMENT_LOGS_DIR}:/workspace/experiment_logs" \
#     \
#     `# ---- Cache volumes (avoid re-downloading models/data) ----` \
#     -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
#     -v "${WANDB_DIR}:/root/.config/wandb" \
#     -v "${WANDB_CACHE}:/root/.local/share/wandb" \
#     \
#     `# ---- Environment variables ----` \
#     -e HF_HUB_ENABLE_HF_TRANSFER=1 \
#     -e HF_HOME=/root/.cache/huggingface \
#     -e HF_DATASETS_CACHE="/root/.cache/huggingface/datasets" \
#     -e HF_LEROBOT_HOME=/workspace/lerobot_data \
#     -e WANDB_CACHE_DIR=/root/.local/share/wandb \
#     ${WANDB_API_KEY:+-e WANDB_API_KEY="${WANDB_API_KEY}"} \
#     \
#     "${IMAGE_NAME}:${IMAGE_TAG}" \
#     "$@"

docker run -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --runtime=nvidia \
    --shm-size=16g \
    --net=host \
    --privileged \
    \
    `# ---- Project source (live-mount for development) ----` \
    -v "${PROJECT_DIR}:/workspace" \
    -v "${RUN_SCRIPT_PATH}:/workspace/run.sh" \
    \
    `# ---- Persistent data volumes ----` \
    -v "${LEROBOT_DATA_DIR}:/workspace/lerobot_data" \
    -v "${LEROBOT_DIR}:/workspace/unitree_lerobot/lerobot" \
    -v "${EVAL_RESULTS_DIR}:/workspace/open_loop_eval_results" \
    -v "${DATASETS_DIR}:/workspace/datasets" \
    -v "${EXPERIMENT_LOGS_DIR}:/workspace/experiment_logs" \
    \
    `# ---- Cache volumes (avoid re-downloading models/data) ----` \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${WANDB_DIR}:/root/.config/wandb" \
    -v "${WANDB_CACHE}:/root/.local/share/wandb" \
    \
    `# ---- Environment variables ----` \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_DATASETS_CACHE="/root/.cache/huggingface/datasets" \
    -e HF_LEROBOT_HOME=/workspace/lerobot_data \
    -e WANDB_CACHE_DIR=/root/.local/share/wandb \
    \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$@"
