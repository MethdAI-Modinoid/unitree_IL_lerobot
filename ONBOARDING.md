# Methdai VLA Research — Unitree G1 + π0.5 Onboarding Guide

## A Note Before You Start

This document exists to show you where we are, not to tell you where to go.

Everything in here — the model choices, the training configs, the dataset structure, the evaluation setup — reflects our current best understanding, which is incomplete and almost certainly improvable. Some decisions were made deliberately, some were made under time pressure, and some we're genuinely unsure about. If something looks wrong, over-engineered, under-engineered, or just doesn't make sense to you, **that instinct is valuable — please say so**. Ask why. Push back. The reason you're here is not to execute a fixed plan but to think critically about what we're doing and make it better.

There are no dumb questions. If this doc doesn't explain something clearly, that's a gap we want to fix. If you read a section and think "wait, why would you do it that way?", bring it up — you might be right.

---

This document is the primary onboarding reference for anyone joining the Methdai VLA team. It covers everything from conceptual background to running training and inference end-to-end.

This repo is Methdai's internal fork of [unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot), customized for π0.5 training, our dataset pipelines, open-loop evaluation infrastructure, and ablation management. For generic setup instructions (conda install, submodule init, etc.) see [README.md](README.md). This doc covers the Methdai-specific layer and all the conceptual background a new hire needs.

---

## Table of Contents

1. [Background Concepts](#1-background-concepts)
2. [Robot & Dataset Configuration](#2-robot--dataset-configuration)
3. [Environment Setup](#3-environment-setup)
4. [Data Pipeline](#4-data-pipeline)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Our Dataset](#7-our-dataset)
8. [Experiments Conducted](#8-experiments-conducted)
9. [Planned Ablations & Future Work](#9-planned-ablations--future-work)
10. [Repo Structure Quick Reference](#10-repo-structure-quick-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Background Concepts

### 1.1 Imitation Learning (IL)

Imitation Learning is how we train robot policies here. Instead of designing reward functions or running trial-and-error in simulation, a human operator teleoperates the robot to collect demonstrations, and the policy learns to copy those demonstrations via supervised learning on (observation → action) pairs. No reward engineering, no sim-to-real gap — the robot learns directly from real human behavior.

### 1.2 What is a VLA?

A **Vision-Language-Action** model is a policy that takes camera images and an optional text task description as input, and outputs robot joint commands. The "language" part enables task conditioning: the same trained model can be told "pick up the apple" or "pick up the cube" without retraining — the VLM backbone interprets the instruction and conditions the action output accordingly. VLAs are typically large pretrained models (billions of parameters) fine-tuned on robot demonstration data.

### 1.3 What is π0.5?

π0.5 (pi-zero-point-five) is Physical Intelligence's second-generation VLA policy, described in [Black et al. 2025](https://www.physicalintelligence.company/download/pi05.pdf). Key points for our use:

- **Architecture**: PaliGemma 3B VLM (handles vision + language) with a separate **action expert** head that uses **flow matching** to generate smooth action trajectories.
- **Flow matching**: A generative modeling technique. During training, the model learns to map random noise → clean actions. At inference, it iteratively denoises a random action sequence into a precise, executable trajectory. This produces smoother, more precise actions than simple regression.
- **Pretrained base**: We start from `lerobot/pi05_base` on Hugging Face Hub — a checkpoint co-trained on a massive heterogeneous dataset (97.6% of training data is web data, other robot types, and semantic prediction tasks). This gives the model strong visual and linguistic priors before we ever show it a Unitree G1 demo.
- **Fine-tuning**: We fine-tune `lerobot/pi05_base` on our Unitree G1 pick-and-place demonstrations using LeRobot's training script.
- **Action chunking**: The policy predicts 50 steps at once (~1.67 seconds at 30 Hz) rather than one step at a time. This reduces compounding errors and improves temporal consistency.

### 1.4 What is LeRobot?

[LeRobot](https://github.com/huggingface/lerobot) is HuggingFace's open-source framework for robot imitation learning. It provides:

- **Dataset format** (v3.0): Episodes stored as Parquet files (state/action) + AV1-encoded MP4 videos (images). Efficient, compressed, HF Hub compatible.
- **Training script**: `lerobot_train.py` — a single entry point with [tyro](https://github.com/brentyi/tyro) CLI config that handles data loading, model training, checkpointing, and W&B logging.
- **Policy implementations**: Act, Diffusion, Pi0, Pi05, Groot, and more — all under `unitree_lerobot/lerobot/src/lerobot/policies/`.
- **Our fork**: `unitree_lerobot/lerobot/` (git submodule, branch `pi05`) — adds Unitree-specific patches and the π0.5 policy.

The env var `HF_LEROBOT_HOME` controls where datasets are cached on disk (default is `~/.cache/huggingface/lerobot/`; we override this to `./lerobot_data/`).

### 1.5 What is the Unitree G1?

The Unitree G1 is a full-size humanoid robot (~1.3 m tall, ~35 kg). It has two 7-DOF arms, two dexterous hands, and an onboard perception suite. We communicate with it via **Unitree SDK2** — a DDS/CycloneDDS pub-sub middleware — using the `unitree_sdk2_python` submodule at the repo root.

Our current experimental setup uses:
- **Single right arm** (7 joint DOF)
- **Dex3 right hand** (7-DOF dexterous hand, compressed to a single gripper scalar)
- **Dual cameras**: head camera (`cam_left_high`) + right wrist camera (`cam_right_wrist`)
- **30 Hz** control frequency

---

## 2. Robot & Dataset Configuration

This section explains how CLI flags and dataset names map to hardware. Understanding this is essential for running scripts correctly.

### 2.1 Arm configurations

The `--arm` flag in eval scripts selects the arm controller class:

| Flag | Robot variant | Notes |
|---|---|---|
| `G1_29` | 29-DOF G1 | **Our hardware** — use this |
| `G1_23` | 23-DOF G1 | Older variant; only use if data was collected on it |

Both controllers target the same 7 right-arm joints: `kRightShoulderPitch`, `kRightShoulderRoll`, `kRightShoulderYaw`, `kRightElbow`, `kRightWristRoll`, `kRightWristPitch`, `kRightWristYaw`.

### 2.2 End-effector (hand) configurations

The `--ee` flag selects the hand controller:

| `--ee` | Hand | DOF | Notes |
|---|---|---|---|
| `dex3` | Unitree Dex3 | 7 | **Our primary hand** |
| `dex1` | Unitree Dex1 | 6 | Simpler hand |
| `inspire1` | Inspire Robotics | — | Third-party alternative |
| `brainco` | Brainco | — | Third-party alternative |

**Gripper representation**: Raw Dex3 has 7 motor positions per hand. For dual-arm setups this is 28D total (14 arm + 14 hand). We compress the hand state to a single 0→1 scalar (0 = open, 1 = closed) using the gripper converter at [unitree_lerobot/eval_robot/utils/gripper_converter.py](unitree_lerobot/eval_robot/utils/gripper_converter.py). This shrinks the action space and makes it easier for the policy to learn grasp/release timing.

### 2.3 Camera configurations

| Key in dataset | Physical camera | Resolution |
|---|---|---|
| `observation.images.cam_left_high` | Head camera (high vantage, left-of-center) | 480×640 |
| `observation.images.cam_right_wrist` | Right wrist-mounted camera | 480×640 |
| `observation.images.cam_left_wrist` | Left wrist camera (dual-arm setups only) | 480×640 |

- **Dual cam** = head + one wrist camera
- **Single cam** = head camera only

### 2.4 DOF configuration naming convention

Dataset names encode the configuration so you can tell at a glance what's inside:

| Dataset name | Arm | Hand repr | Cameras | State/action dim |
|---|---|---|---|---|
| `single_arm_dual_cam` | Right arm | 1D gripper | head + right wrist | **8D** |
| `single_camera_single_gripper` | Right arm | 1D gripper | head only | 8D |
| `three_camera` | Dual arm | Full 28D hand | head + both wrists | 28D+ |

**DOF conversion modes** (via `convert_hands_to_gripper.py`):

| `--dof-mode` | Input → Output | Description |
|---|---|---|
| `single_gripper` | 28D → 8D | Right arm (7 DOF) + right gripper (1D) — **our primary config** |
| `dual_gripper` | 28D → 16D | Both arms (7 DOF × 2) + both grippers (1D × 2) |
| `single_full` | 28D → 14D | Right arm (7 DOF) + full right hand (7 DOF) |
| `dual_full` | 28D → 28D | Both arms + full both hands (no compression) |

### 2.5 Observation and Action State

This is the most important thing to understand conceptually. Every timestep in the dataset has these keys:

---

**`observation.state`** — current robot proprioception (what the robot senses about its own joints)
- Shape: `[8]` for our primary config
- Contents: `[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw, gripper_openness]`
- Units: radians for arm joints; normalized 0→1 for gripper

**`action`** — target joint positions that the policy must predict
- Shape: `[8]` (identical structure to `observation.state`)
- This is what the human operator commanded during teleoperation
- The policy learns to predict this from observations

**`observation.images.cam_left_high`** — head camera RGB image, shape `[480, 640, 3]`

**`observation.images.cam_right_wrist`** — right wrist camera RGB image, shape `[480, 640, 3]`

**Task string** — natural language description (e.g., `"pick up the cube and place it in the brown box"`). During training this may be empty or auto-generated. At inference, pass it via `--custom_task`.

---

**Full policy I/O:**
```
Input:  observation.state [8]
        observation.images.cam_left_high  [480, 640, 3]
        observation.images.cam_right_wrist [480, 640, 3]
        task_string (text)

Output: action chunk [50 × 8]   ← 50 future timesteps predicted at once
```

The policy runs at 30 Hz but only re-computes every `n_action_steps` steps (the action chunk is executed open-loop between re-computations).

---

## 3. Environment Setup

### 3.1 Docker (Recommended — especially for training)

The `Dockerfile` bundles all dependencies: CUDA 12.8, PyTorch, Transformers, LeRobot, and Unitree SDK. This is the reproducible path.

**Build the image** (first time, or after `Dockerfile` changes):
```bash
docker build -t unitree_lerobot .
```

**Run an interactive shell:**
```bash
./docker_run.sh
```

**Run a script directly:**
```bash
./docker_run.sh python -s src/lerobot/scripts/lerobot_train.py --dataset.repo_id=... <args>
```

`docker_run.sh` automatically mounts the source tree, `lerobot_data/`, checkpoint directories, and the HuggingFace cache, and sets all required env vars (`HF_LEROBOT_HOME`, `HF_HOME`, `WANDB_CACHE_DIR`, etc.).

### 3.2 Conda (Alternative — for local eval without Docker)

See [README.md §1](README.md#1--environment-setup) for the full conda setup. Key requirements:
- Python 3.10
- `pinocchio` (robot kinematics) via conda-forge
- `ffmpeg 7.1.1` via conda-forge (for AV1 video encoding/decoding)
- LeRobot submodule installed in editable mode: `cd unitree_lerobot/lerobot && pip install -e .`
- This package editable: `pip install -e .` at repo root

---

## 4. Data Pipeline

The full pipeline from teleoperation to a trainable dataset:

```
AVP teleoperation → raw JSON episodes → sort/rename → LeRobot v3.0 → DOF conversion → HF Hub
```

### 4.1 Data Collection

Raw data is collected with [avp_teleoperate](https://github.com/unitreerobotics/avp_teleoperate) — an Apple Vision Pro XR teleoperation system for Unitree G1. Each episode is saved as a folder:

```
datasets/task_name/
  episode_0001/
    colors/    ← camera frames as PNG
    depths/    ← depth frames
    data.json  ← joint states, actions, timestamps
  episode_0002/
  ...
```

### 4.2 Conversion: JSON → LeRobot v3.0

```bash
# Step 1: normalize episode numbering (ensures sequential episode_0000, episode_0001, ...)
python unitree_lerobot/utils/sort_and_rename_folders.py --data_dir datasets/task_name/

# Step 2: convert to LeRobot Parquet + AV1 MP4
HF_LEROBOT_HOME=./lerobot_data python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir datasets/task_name \
    --repo-id deepansh-methdai/your_dataset_name \
    --robot_type Unitree_G1_Dex3 \
    --push_to_hub        # omit if not uploading to HF Hub
```

The `--robot_type` flag controls which joints and cameras are extracted from the raw JSON. The full registry of available robot types is in [unitree_lerobot/utils/constants.py](unitree_lerobot/utils/constants.py).

### 4.3 DOF Conversion

Raw G1+Dex3 data is 28D. We convert to 8D single-gripper for our primary experiments:

```bash
# Always dry-run first to verify feature shapes
python -s unitree_lerobot/utils/convert_hands_to_gripper.py \
    --repo-id deepansh-methdai/task_name \
    --output-repo-id deepansh-methdai/task_name_test \
    --dof-mode single_gripper \
    --root ./lerobot_data/deepansh-methdai/task_name \
    --dry-run

# Then run for real
python -s unitree_lerobot/utils/convert_hands_to_gripper.py \
    --repo-id deepansh-methdai/task_name \
    --output-repo-id deepansh-methdai/task_name_single_gripper \
    --dof-mode single_gripper \
    --root ./lerobot_data/deepansh-methdai/task_name \
    --push-to-hub
```

You can also remove camera features from a dataset if you want a single-camera variant:
```bash
lerobot-edit-dataset \
    --repo_id deepansh-methdai/task_name \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.cam_left_wrist']"
```

---

## 5. Training

All training runs through LeRobot's training script inside the `unitree_lerobot/lerobot/` submodule (or via Docker, which sets up paths automatically).

### 5.1 Primary training command (π0.5)

```bash
cd unitree_lerobot/lerobot

HF_LEROBOT_HOME=../../lerobot_data \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=true \
    --policy.dtype=bfloat16 \
    --steps=20000 \
    --batch_size=32 \
    --save_freq=2000 \
    --log_freq=50 \
    --wandb.enable=true \
    --output_dir=../../outputs_pi/my_run \
    --job_name=my_run
```

### 5.2 Key training flags explained

| Flag | What it does | Notes |
|---|---|---|
| `--policy.pretrained_path=lerobot/pi05_base` | Load π0.5 pretrained base from HF Hub | Always use this — don't train from scratch |
| `--policy.freeze_vision_encoder=true` | Freeze the PaliGemma image encoder | Faster; prevents forgetting pretrained visual features. **Our primary config.** |
| `--policy.freeze_vision_encoder=false` | Let the vision encoder fine-tune | May help if our camera setup differs significantly from pretraining data |
| `--policy.train_expert_only=true` | Only update the action expert head | Fewest trainable params; fastest; but limits the model's ability to use VLM reasoning |
| `--policy.train_expert_only=false` | Update VLM + action expert | **Our primary config.** More expressive but slower to train |
| `--policy.gradient_checkpointing=true` | Recompute activations during backward pass | Trades compute for GPU memory — **required** to fit a 3B model on one GPU |
| `--policy.compile_model=true` | `torch.compile` the model | Faster training throughput after a one-time compilation cost |
| `--policy.dtype=bfloat16` | Mixed precision training | **Required** to fit the 3B model in memory |
| `--batch_size=32` | Batch size per GPU | Reduce to 16 or 8 if OOM |
| `--steps` | Total gradient update steps | We typically use 20k–30k |
| `--save_freq` | Checkpoint save interval (in steps) | We use 2000–5000 |
| `--wandb.enable=true` | Log to Weights & Biases | Make sure `WANDB_API_KEY` is set |

### 5.3 Resuming a training run

```bash
python -s src/lerobot/scripts/lerobot_train.py \
    --resume=true \
    --config_path=../../outputs_pi/my_run/020000/pretrained_model/train_config.json \
    --steps=30000
```

The `train_config.json` in any checkpoint directory stores the full original training config. Pass it with `--config_path` to resume from that checkpoint with the same settings (override `--steps` to extend the run).

### 5.4 Checkpoint structure

```
outputs_pi/my_run/
  020000/pretrained_model/      ← checkpoint at step 20000
    config.json                 ← policy architecture + hyperparams
    train_config.json           ← full training config (used for resume)
    model.safetensors           ← weights
    policy_preprocessor.json    ← input normalization stats
    policy_postprocessor.json   ← action clipping/scaling
```

---

## 6. Evaluation

### 6.1 Open-loop dataset evaluation (no robot needed — start here)

Runs the trained policy on held-out dataset episodes. The policy receives observations from the recorded dataset (not live cameras), predicts actions, and we compare predicted vs ground-truth actions.

**Metrics computed**: MAE, MSE, RMSE — overall and per joint. Results saved to `metrics.json` alongside a per-joint RMSE plot.

```bash
python -s unitree_lerobot/eval_robot/eval_g1_dataset.py \
    --policy.path=./outputs_pi/my_run/020000/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --episodes=10 \
    --frequency=30 \
    --arm=G1_29 \
    --ee=dex3 \
    --send_real_robot=false \
    --visualization=false
```

**Important caveat**: Open-loop metrics measure prediction accuracy on recorded data, not real-world task success. A model can have low RMSE and still fail on the robot, or succeed on the robot despite mediocre open-loop numbers. Always validate closed-loop before drawing conclusions.

### 6.2 Synthetic (visual robustness) evaluation

Same as open-loop, but image backgrounds are replaced with a solid color or random noise before being fed to the policy. This tests whether the model uses visual context to drive its actions, or is mostly relying on proprioception and timing.

```bash
python -s unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py \
    --policy.path=./outputs_pi/my_run/020000/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --frame_mode=white \        # or: black, random
    --episodes=10 \
    --frequency=30 \
    --arm=G1_29 --ee=dex3 \
    --send_real_robot=false --visualization=false
```

If open-loop RMSE degrades significantly with a white background, the model is relying heavily on visual scene content — which is good (it's using vision), but also means it may be fragile to visual distribution shift.

### 6.3 Real-world closed-loop inference

This is the gold standard. The policy runs live: reads camera frames in real time, predicts actions every N steps, and sends commands to the robot arm and hand at 30 Hz.

**Prerequisites before running:**
1. Image server running on the robot PC (see [avp_teleoperate image server docs](https://github.com/unitreerobotics/avp_teleoperate?tab=readme-ov-file#31-%EF%B8%8F-image-server))
2. DDS network configured (robot and control PC on same subnet)
3. Robot arm moved to a safe starting pose (scripts will ask for confirmation)

```bash
python -s unitree_lerobot/eval_robot/eval_g1.py \
    --policy.path=./outputs_pi/my_run/020000/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --frequency=30 \
    --arm=G1_29 \
    --ee=dex3 \
    --send_real_robot=true \
    --custom_task="pick up the cube and place it in the brown box"
```

Omit `--send_real_robot=true` (or set it to `false`) for a dry-run that loads the policy and simulates the loop without actually commanding the robot.

### 6.4 Dataset replay on robot (debugging tool)

Plays back ground-truth actions from the dataset directly to the robot — no policy involved. Use this to:
- Verify robot connectivity and DDS communication
- Check that joint ranges are safe before running the policy
- Confirm the dataset's actions are physically feasible on your hardware

```bash
python -s unitree_lerobot/eval_robot/replay_robot.py \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --episodes=0 \
    --frequency=30 \
    --arm=G1_29 \
    --ee=dex3 \
    --send_real_robot=true
```

Always run this before running inference on a new dataset or after hardware changes.

### 6.5 Automated ablation suite

[research/eval_suite.yaml](research/eval_suite.yaml) is a registry of checkpoints organized into ablation groups. [unitree_lerobot/eval_robot/run_open_loop_suite.py](unitree_lerobot/eval_robot/run_open_loop_suite.py) iterates over all groups, runs open-loop eval for each checkpoint, saves `metrics.json` and a per-joint RMSE plot, then generates a side-by-side `comparison.png` for each ablation group.

```bash
python -s unitree_lerobot/eval_robot/run_open_loop_suite.py \
    --config=research/eval_suite.yaml
```

Results land in `results/open_loop/<ablation_group>/<label>/`. To add a new checkpoint to the suite, just add an entry to `eval_suite.yaml`.

---

## 7. Our Dataset

**Primary dataset**: `deepansh-methdai/single_arm_dual_cam` (stored locally in `lerobot_data/`)

| Property | Value |
|---|---|
| Task | Pick-and-place manipulation |
| Robot configuration | Unitree G1, right arm only, Dex3 hand |
| State / action dimension | **8D** (7 arm joints + 1 gripper scalar) |
| Cameras | `cam_left_high` (head) + `cam_right_wrist` (wrist) |
| Total episodes | 798 |
| Total frames | 387,633 |
| Control frequency | 30 Hz |
| Average episode length | ~485 frames (~16 s) |
| Video codec | AV1 (MP4) |
| Dataset format | LeRobot v3.0 (Parquet + MP4) |

This dataset was derived from a raw 28D capture (dual-arm, full Dex3 hand) collected via `avp_teleoperate`, then converted to 8D using `convert_hands_to_gripper.py --dof-mode single_gripper` (right arm only, right gripper scalar).

The dataset schema (`lerobot_data/deepansh-methdai/single_arm_dual_cam/meta/info.json`):
```json
{
  "observation.state":               {"shape": [8]},
  "action":                          {"shape": [8]},
  "observation.images.cam_left_high":  {"shape": [480, 640, 3]},
  "observation.images.cam_right_wrist": {"shape": [480, 640, 3]}
}
```

---

## 8. Experiments Conducted

All experiments use the `deepansh-methdai/single_arm_dual_cam` dataset. Open-loop evaluation is run on 10 held-out episodes (6,619 total timesteps). Numeric results are in `results/open_loop/` (JSON + comparison bar charts) and will be formally analyzed in a separate research report.

### Primary run

- **Config**: 30k steps, frozen vision encoder (`freeze_vision_encoder=true`), full VLM fine-tune (`train_expert_only=false`)
- **Checkpoint**: `outputs_pi/pi05_primary_30k/`
- **Best open-loop checkpoint**: 20k steps (further training shows diminishing or marginal returns — see Ablation B)
- This is our SOTA reference for all comparisons

### Ablation A — Vision Encoder Freezing

Do we let the PaliGemma image encoder adapt to our robot's visual domain, or keep it frozen?

| Config | Training flag | Checkpoint |
|---|---|---|
| Frozen (primary) | `--policy.freeze_vision_encoder=true` | `outputs_pi/pi05_primary_30k/020000/` |
| Unfrozen | `--policy.freeze_vision_encoder=false` | `outputs_pi/pi05_unfrozen_vision_20k/020000/` |

Motivation: Freezing prevents catastrophic forgetting of pretrained visual features and is faster (fewer parameters to update). Unfreezing may improve performance if our camera setup differs significantly from the base model's pretraining distribution, but risks overfitting to training scenes.

### Ablation B — Training Steps

How do open-loop metrics evolve as training progresses?

| Checkpoint | Steps | Source |
|---|---|---|
| 10k | Early | `outputs_pi/pi05_primary_30k/010000/` |
| 20k | Mid — best | `outputs_pi/pi05_primary_30k/020000/` |
| 30k | Full | `outputs_pi/pi05_primary_30k/030000/` |

Motivation: More steps is not always better. The open-loop metric helps us identify when the model has converged vs when it starts overfitting to training episodes.

### Ablation C — Expert-Only Training

Does fine-tuning the full VLM backbone provide meaningful benefit over only updating the action expert head?

| Config | Training flag | Checkpoint |
|---|---|---|
| Expert only | `--policy.train_expert_only=true` | `outputs_pi/pi05_expert_only_20k/020000/` |
| Full fine-tune (primary) | `--policy.train_expert_only=false` | `outputs_pi/pi05_primary_30k/020000/` |

Motivation: `train_expert_only=true` freezes the ~3B VLM and only trains the smaller action expert head — much faster and uses fewer GPU resources. The question is whether the VLM's frozen representations are sufficient, or if the full fine-tune is necessary to adapt to our task.

---

All metrics JSON files are at `results/open_loop/<ablation>/<label>/metrics.json`. Bar chart comparisons per ablation group are at `results/open_loop/<ablation>/comparison.png`. Detailed quantitative analysis and discussion of findings will be in the **formal research report**.

---

## 9. Planned Ablations & Future Work

See [research/research_plan.md](research/research_plan.md) for the complete research plan, timeline, and experiment checklist. High-level upcoming work:

| Work item | Description |
|---|---|
| **Closed-loop success rate** | 10 trials × 4 objects (2 seen, 2 unseen) on real robot → formal success rate table |
| **Visual perturbation (closed-loop)** | Background cloth changes; compare frozen vs unfrozen vision encoder in real-world conditions |
| **Failure mode analysis** | Categorize failures: grasp miss, grasp slip, wrong object, placement miss, collision, timeout |
| **RTC (Real-Time Correction)** | Allow the human operator to provide corrective signals during inference — not yet implemented |
| **More training data** | Collect additional episodes to study the data scaling curve |

---

## 10. Repo Structure Quick Reference

```
unitree_IL_lerobot/
├── unitree_lerobot/
│   ├── lerobot/                      ← LeRobot submodule (fork, branch: pi05)
│   │   └── src/lerobot/
│   │       ├── scripts/
│   │       │   └── lerobot_train.py  ← main training entry point
│   │       └── policies/
│   │           └── pi05/             ← π0.5 policy implementation
│   ├── eval_robot/
│   │   ├── eval_g1.py                ← real robot closed-loop inference
│   │   ├── eval_g1_dataset.py        ← open-loop dataset evaluation
│   │   ├── eval_g1_dataset_synthetic.py  ← visual robustness evaluation
│   │   ├── eval_g1_sim.py            ← Isaac Lab simulation inference
│   │   ├── run_open_loop_suite.py    ← automated ablation runner
│   │   ├── replay_robot.py           ← ground-truth action replay on robot
│   │   ├── make_robot.py             ← hardware setup helpers
│   │   ├── robot_control/            ← arm + hand controller classes
│   │   │   ├── robot_arm.py          ← G1_29/G1_23 arm DDS control
│   │   │   ├── robot_arm_ik.py       ← inverse kinematics (CasADi)
│   │   │   └── robot_hand_unitree.py ← Dex3/Dex1 hand control
│   │   ├── image_server/             ← camera streaming (server on robot PC)
│   │   └── utils/
│   │       ├── utils.py              ← MAE/MSE/RMSE metrics
│   │       └── gripper_converter.py  ← 7D hand ↔ 1D gripper conversion
│   └── utils/
│       ├── constants.py              ← robot config registry (RobotConfig)
│       ├── convert_unitree_json_to_lerobot.py
│       └── convert_hands_to_gripper.py
│
├── lerobot_data/                     ← dataset cache (not committed)
│   └── deepansh-methdai/
│       └── single_arm_dual_cam/      ← primary dataset
├── outputs_pi/                       ← trained checkpoints (not committed)
│   ├── pi05_primary_30k/
│   ├── pi05_expert_only_20k/
│   └── pi05_unfrozen_vision_20k/
├── results/open_loop/                ← eval results (JSON + plots)
│   ├── primary/
│   ├── training_steps/
│   ├── vision_encoder/
│   └── expert_only/
├── research/
│   ├── eval_suite.yaml               ← ablation checkpoint registry
│   └── research_plan.md             ← full research methodology & timeline
│
├── Dockerfile                        ← reproducible training environment
├── docker_run.sh                     ← run container with correct mounts
├── run.sh                            ← training command reference (history)
├── devnotes.md                       ← ad-hoc dev commands & snippets
└── README.md                         ← upstream Unitree generic setup guide
```

---

## 11. Troubleshooting

| Problem | Solution |
|---|---|
| `401 Unauthorized` on HuggingFace Hub | Run `huggingface-cli login` |
| `Unknown encoder 'libsvtav1'` | `conda install -c conda-forge ffmpeg` (need ffmpeg with SVT-AV1 support) |
| `FileNotFoundError: No such file or directory: 'ffmpeg'` | Same — install via conda-forge |
| `Access to google/paligemma-3b-pt-224 is restricted` | Request access on HF Hub model page, then re-login |
| CUDA OOM during training | Lower `--batch_size`; confirm `--policy.gradient_checkpointing=true` and `--policy.dtype=bfloat16` are set |
| Open-loop eval — mismatched action dims | `--ee` and `--arm` must match the dataset's robot type. Check `lerobot_data/.../meta/info.json` for the `robot_type` field |
| Image server not connecting during real-robot eval | Verify the IP address in `image_client.py` matches the robot PC; confirm the image server process is running on the robot PC |
| `RuntimeError: Could not load libtorchcodec` | FFmpeg is not properly installed — reinstall via conda-forge |

For additional setup troubleshooting see [README.md §6](README.md#6--troubleshooting).

---

*For research methodology, experimental design, and the formal results analysis see [research/research_plan.md](research/research_plan.md). For ad-hoc commands and quick-run snippets see [devnotes.md](devnotes.md).*
