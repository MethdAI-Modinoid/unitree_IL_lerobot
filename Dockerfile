# ==============================================================================
# Dockerfile for unitree_IL_lerobot_synced — π0.5 (pi05) policy
# Base: NVIDIA CUDA 12.8 + Ubuntu 22.04 (host driver ≥ 570.x)
# Python 3.10 (required by unitree_lerobot pyproject.toml: >=3.10,<3.11)
#
# All package versions pinned to match requirements_freezed.txt
# ==============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# ---- Prevent interactive prompts during apt install ----
ENV DEBIAN_FRONTEND=noninteractive

# ---- Fix /tmp permissions (known issue with some nvidia/cuda base images) ----
RUN chmod 1777 /tmp

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        software-properties-common \
        # Video / image codec libs (ffmpeg, av, opencv, imageio)
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        # OpenCV / GUI libs
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libxrender1 \
        libx11-6 \
        libxkbcommon-x11-0 \
        libegl1 \
        # Python build deps
        libffi-dev \
        libssl-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        liblzma-dev \
        zlib1g-dev \
        libncursesw5-dev \
        libxml2-dev \
        libxmlsec1-dev \
        tk-dev \
        # pynput / evdev (input devices for robot control)
        libevdev-dev \
        xdotool \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python 3.10 (deadsnakes PPA) ----
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# ---- Set python3.10 as default ----
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ---- Install pip & uv ----
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

ENV PIP_DEFAULT_TIMEOUT=300

# ---- Set working directory ----
WORKDIR /workspace

# ---- Copy only dependency manifests first (cache-friendly) ----
COPY pyproject.toml                         /workspace/pyproject.toml
COPY unitree_lerobot/lerobot/pyproject.toml /workspace/unitree_lerobot/lerobot/pyproject.toml
COPY unitree_lerobot/lerobot/setup.py       /workspace/unitree_lerobot/lerobot/setup.py
COPY unitree_sdk2_python/setup.py           /workspace/unitree_sdk2_python/setup.py
COPY unitree_sdk2_python/pyproject.toml     /workspace/unitree_sdk2_python/pyproject.toml

# ==============================================================================
# Layer 1: PyTorch stack (CUDA 12.8) — pinned to requirements_freezed.txt
# NOTE: --extra-index-url for PyTorch cu128 wheels; default PyPI for NVIDIA deps
#       UV_HTTP_TIMEOUT=300 to handle slow pypi.nvidia.com downloads
# ==============================================================================
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
        torch==2.9.1 \
        torchvision==0.24.1 \
        torchcodec==0.9.1 \
        --extra-index-url https://download.pytorch.org/whl/cu128

# ==============================================================================
# Layer 2: π0.5 core dependencies — exact versions from requirements_freezed.txt
#   transformers  — PaliGemma / Gemma backbone used by pi05 policy
#   scipy         — required by the lerobot[pi] extra for pi0/pi05
#   accelerate    — mixed-precision & multi-GPU training
#   diffusers     — action-head diffusion components
#   safetensors   — checkpoint format
#   einops        — tensor reshaping in pi05 model
# ==============================================================================
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
        "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" \
        "scipy==1.14.1" \
        "accelerate==1.12.0" \
        "diffusers==0.35.2" \
        "safetensors==0.7.0" \
        "einops==0.8.1"

# ==============================================================================
# Layer 3: HuggingFace ecosystem — dataset loading, model hub, tokenizers
# ==============================================================================
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
        "datasets==4.1.1" \
        "huggingface-hub[hf-transfer,cli]==0.35.3" \
        "tokenizers==0.21.4" \
        "hf-transfer==0.1.9"

# ==============================================================================
# Layer 4: LeRobot core deps — versions from requirements_freezed.txt
# ==============================================================================
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
        "draccus==0.10.0" \
        "gymnasium==1.2.3" \
        "av==15.1.0" \
        "jsonlines==4.0.0" \
        "opencv-python-headless==4.12.0.88" \
        "imageio[ffmpeg]==2.37.2" \
        "deepdiff==8.6.1" \
        "pynput==1.8.1" \
        "pyserial==3.5" \
        "rerun-sdk==0.26.2" \
        "termcolor==3.3.0" \
        "wandb==0.24.0"

# ==============================================================================
# Layer 5: Unitree / robot control dependencies
# ==============================================================================
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
        "cyclonedds==0.10.2" \
        "casadi==3.7.0" \
        "meshcat==0.3.2" \
        "pyzmq==27.1.0" \
        "pyngrok==7.5.0" \
        "evdev==1.9.2" \
        "logging-mp==0.1.6"

# ==============================================================================
# Layer 6: Remaining pinned packages from requirements_freezed.txt
# ==============================================================================
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install \
        "numpy==2.2.6" \
        "pillow==12.1.0" \
        "pandas==2.3.3" \
        "matplotlib==3.10.8" \
        "pyyaml==6.0.3" \
        "pyyaml-include==1.4.1" \
        "protobuf==6.33.2" \
        "pydantic==2.12.5" \
        "rich==14.2.0" \
        "rich-click==1.9.5" \
        "tyro==1.0.3" \
        "cloudpickle==3.1.2" \
        "psutil==7.2.1" \
        "toml==0.10.2" \
        "gitpython==3.1.46" \
        "tqdm==4.67.1" \
        "regex==2025.11.3" \
        "requests==2.32.5" \
        "packaging==25.0" \
        "setuptools==80.9.0" \
        "cmake==4.1.3"

# ---- Now copy the full project source ----
COPY . /workspace/

# ==============================================================================
# Layer 7: Install local editable packages
#   - unitree_sdk2_python (CycloneDDS robot communication)
#   - lerobot             (core ML framework — skip deps, already installed above)
#   - unitree_lerobot     (top-level project glue)
# ==============================================================================
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --no-deps -e /workspace/unitree_sdk2_python \
    && pip install --no-deps -e /workspace/unitree_lerobot/lerobot \
    && pip install --no-deps -e /workspace/

# ---- Verify pi05 critical imports work ----
# NOTE: GPU/CUDA is NOT available during docker build; only check at runtime.
RUN python -c "\
import torch; \
print(f'  torch={torch.__version__} cuda={torch.version.cuda} (runtime GPU check deferred)'); \
import transformers; \
import scipy; \
import accelerate; \
import diffusers; \
import einops; \
import safetensors; \
import lerobot; \
from lerobot.policies.pi05.modeling_pi05 import PI05Policy; \
print('=== pi05 Docker build verification PASSED ==='); \
print(f'  transformers={transformers.__version__}'); \
print(f'  lerobot={lerobot.__version__}'); \
"

# ---- Environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_LEROBOT_HOME=/workspace/lerobot_data

# ---- Default entrypoint ----
ENTRYPOINT ["/bin/bash"]
