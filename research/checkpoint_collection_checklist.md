# Checkpoint Collection & Training Checklist

This document provides a comprehensive checklist for training and collecting all required checkpoints for the Pi0.5 evaluation on Unitree G1.

---

## Overview

**Dataset**: `deepansh-methdai/single_arm_dual_cam`  
**Total Checkpoints Needed**: 7-8 checkpoints  
**GPU Required**: A100 40GB or H100  
**Training Time per Checkpoint**: ~12-16 hours for 20k steps  
**WandB Project**: `pi05_g1_evaluation`

---

## Checkpoint Training Matrix

| Checkpoint ID | Steps | Freeze Vision | Train Expert Only | Purpose |
|---------------|-------|---------------|-------------------|---------|
| `primary_20k` | 20,000 | **true** | false | **Primary evaluation (frozen vision config)** |
| `vision_frozen_20k` | 20,000 | true | false | Vision encoder ablation (same as primary) |
| `vision_unfrozen_20k` | 20,000 | false | false | Vision encoder ablation comparison |
| `expert_only_20k` | 20,000 | **true** | true | Expert-only ablation |
| `full_finetune_20k` | 20,000 | **true** | false | Expert-only ablation (same as primary) |
| `checkpoint_10k` | 10,000 | **true** | false | Training steps ablation |
| `checkpoint_15k` | 15,000 | **true** | false | Training steps ablation |
| `checkpoint_20k` | 20,000 | **true** | false | Training steps ablation (same as primary) |

**Note**: Some checkpoints serve multiple purposes. You may need only **3-4 unique training runs**:
1. **Primary (frozen vision, full finetune, 20k)** - also serves vision_frozen, full_finetune, checkpoint_20k
2. Unfrozen vision (20k) - for vision encoder ablation comparison
3. Expert-only with frozen vision (20k)
4. Early/mid checkpoints (10k, 15k) - optional if you save intermediate checkpoints from primary run

---

## Pre-Training Checklist

### Environment Setup
- [ ] Verify GPU availability and VRAM (need 40GB+)
```bash
nvidia-smi
```

- [ ] Set up working directory paths (UPDATE THESE!)
```bash
# Set these variables for your environment
export LEROBOT_REPO_DIR=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/unitree_lerobot/lerobot
export HF_LEROBOT_HOME=/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/unitree_IL_lerobot_synced/lerobot_data
export WANDB_PROJECT=pi05_g1_evaluation
export HF_TOKEN=<your_huggingface_token>  # If dataset is private
```

- [ ] Activate conda environment
```bash
conda deactivate
conda activate unitree_lerobot_synced
```

- [ ] Verify dataset is accessible
```bash
huggingface-cli download deepansh-methdai/single_arm_dual_cam --repo-type dataset
```

- [ ] Check Python environment and dependencies
```bash
python -c "from lerobot.common.policies.pi05.modeling_pi05 import Pi05Policy; print('✅ LeRobot installed')"
python -c "import wandb; print('✅ WandB installed')"
```

- [ ] Login to WandB
```bash
wandb login
```

---

## Training Commands

All commands assume you're in the root of the `unitree_lerobot` repository.

### Common Parameters (All Runs)
```bash
# Base parameters used in all training runs
--dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
--policy.type=pi05 \
--policy.pretrained_path=lerobot/pi05_base \
--policy.compile_model=true \
--policy.compile_mode=reduce-overhead \
--policy.gradient_checkpointing=true \
--policy.optimizer_lr=2.5e-5 \
--policy.dtype=bfloat16 \
--policy.device=cuda \
--batch_size=32 \
--save_freq=5000 \
--log_freq=50 \
--wandb.enable=true \
--wandb.project=pi05_g1_evaluation
```

---

### Checkpoint 1: Primary (Frozen Vision, Full Finetune, 20k)

**Purpose**: Primary evaluation + vision ablation baseline + expert ablation baseline + 20k steps baseline

**Configuration**:
- Freeze vision encoder: `true` ⭐ **PRIMARY CONFIG**
- Train expert only: `false`
- Training steps: `20,000`

**Command**:
```bash
cd ${LEROBOT_REPO_DIR} && \
rm -rf ./outputs_pi05_primary_20k && \
HF_LEROBOT_HOME=${HF_LEROBOT_HOME} \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=pi05 \
    --output_dir=./outputs_pi05_primary_20k/ \
    --job_name=pi05_primary_frozen \
    --policy.repo_id=deepansh-methdai/pi05_primary_vision_frozen_20k \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.compile_mode=reduce-overhead \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=2.5e-5 \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=20000 \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=pi05_g1_evaluation \
    --wandb.run_id=primary_vision_frozen_20k \
    --wandb.notes="Primary checkpoint: frozen vision encoder, full finetune, 20k steps"
```

**Checklist**:
- [x] Start training
- [x] Monitor WandB dashboard for first 30 minutes
- [ ] Verify checkpoints saved at 5k, 10k, 15k, 20k steps
- [ ] Training completed successfully
- [ ] Final checkpoint saved to: `./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model`
- [ ] Rename/copy 10k and 15k checkpoints for steps ablation

---

### Checkpoint 2: Unfrozen Vision Encoder (20k)

**Purpose**: Vision encoder ablation comparison (to test if unfreezing improves performance)

**Configuration**:
- Freeze vision encoder: `false` ⚠️
- Train expert only: `false`
- Training steps: `20,000`

**Command**:
```bash
cd ${LEROBOT_REPO_DIR} && \
rm -rf ./outputs_pi05_unfrozen_vision_20k && \
HF_LEROBOT_HOME=${HF_LEROBOT_HOME} \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=pi05 \
    --output_dir=./outputs_pi05_unfrozen_vision_20k/ \
    --job_name=pi05_unfrozen_vision \
    --policy.repo_id=deepansh-methdai/pi05_unfrozen_vision_20k \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.compile_mode=reduce-overhead \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=2.5e-5 \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=20000 \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=pi05_g1_evaluation \
    --wandb.run_name=vision_unfrozen_20k \
    --wandb.notes="Vision encoder ablation: unfrozen vision encoder, full finetune, 20k steps"
```

**Checklist**:
- [x] Start training
- [x] Monitor WandB dashboard
- [x] Verify unfrozen vision encoder in logs (more trainable params vs primary)
- [x] Training completed successfully
- [x] Final checkpoint saved to: `./outputs_pi05_unfrozen_vision_20k/checkpoints/020000/pretrained_model`

---

### Checkpoint 3: Expert-Only Training (20k)

**Purpose**: Expert-only ablation (freezes VLM, trains only expert head)

**Configuration**:
- Freeze vision encoder: `true` (consistent with primary)
- Train expert only: `true` ⚠️
- Training steps: `20,000`

**Command**:
```bash
cd ${LEROBOT_REPO_DIR} && \
rm -rf ./outputs_pi05_expert_only_20k && \
HF_LEROBOT_HOME=${HF_LEROBOT_HOME} \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=pi05 \
    --output_dir=./outputs_pi05_expert_only_20k/ \
    --job_name=pi05_expert_only \
    --policy.repo_id=deepansh-methdai/pi05_expert_only_20k \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.compile_mode=reduce-overhead \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.optimizer_lr=2.5e-5 \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=20000 \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=pi05_g1_evaluation \
    --wandb.run_name=expert_only_frozen_20k \
    --wandb.notes="Expert-only ablation: frozen vision encoder, expert-only training, 20k steps"
```

**Checklist**:
- [ ] Start training
- [ ] Monitor WandB dashboard
- [ ] Verify expert-only mode in logs (only expert head trainable)
- [ ] Training completed successfully
- [ ] Final checkpoint saved to: `./outputs_pi05_expert_only_20k/checkpoints/020000/pretrained_model`

---

### Checkpoint 4: Early Checkpoint (10k) - OPTIONAL

**Note**: If you saved intermediate checkpoints during Checkpoint 1 training, you already have this at `./outputs_pi05_primary_20k/checkpoints/010000/`. Otherwise, train separately:

**Purpose**: Training steps ablation (early stopping point)

**Configuration**:
- Freeze vision encoder: `true` (consistent with primary)
- Train expert only: `false`
- Training steps: `10,000` ⚠️

**Command**:
```bash
cd ${LEROBOT_REPO_DIR} && \
rm -rf ./outputs_pi05_steps_10k && \
HF_LEROBOT_HOME=${HF_LEROBOT_HOME} \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=pi05 \
    --output_dir=./outputs_pi05_steps_10k/ \
    --job_name=pi05_steps_10k \
    --policy.repo_id=deepansh-methdai/pi05_steps_10k \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.compile_mode=reduce-overhead \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=2.5e-5 \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=10000 \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=pi05_g1_evaluation \
    --wandb.run_name=steps_10k_frozen \
    --wandb.notes="Steps ablation: frozen vision encoder, 10k steps"
```

**Checklist**:
- [ ] Start training (or copy from primary run)
- [ ] Training completed successfully
- [ ] Checkpoint saved to: `./outputs_pi05_steps_10k/checkpoints/010000/pretrained_model` or `./outputs_pi05_primary_20k/checkpoints/010000/pretrained_model`

---

### Checkpoint 5: Mid Checkpoint (15k) - OPTIONAL

**Note**: If you saved intermediate checkpoints during Checkpoint 1 training, you already have this at `./outputs_pi05_primary_20k/checkpoints/015000/`. Otherwise, train separately:

**Purpose**: Training steps ablation (mid training point)

**Configuration**:
- Freeze vision encoder: `true` (consistent with primary)
- Train expert only: `false`
- Training steps: `15,000` ⚠️

**Command**:
```bash
cd ${LEROBOT_REPO_DIR} && \
rm -rf ./outputs_pi05_steps_15k && \
HF_LEROBOT_HOME=${HF_LEROBOT_HOME} \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=pi05 \
    --output_dir=./outputs_pi05_steps_15k/ \
    --job_name=pi05_steps_15k \
    --policy.repo_id=deepansh-methdai/pi05_steps_15k \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.compile_mode=reduce-overhead \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=2.5e-5 \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --steps=15000 \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=pi05_g1_evaluation \
    --wandb.run_name=steps_15k_frozen \
    --wandb.notes="Steps ablation: frozen vision encoder, 15k steps"
```

**Checklist**:
- [ ] Start training (or copy from primary run)
- [ ] Training completed successfully
- [ ] Checkpoint saved to: `./outputs_pi05_steps_15k/checkpoints/015000/pretrained_model` or `./outputs_pi05_primary_20k/checkpoints/015000/pretrained_model`

---

## Post-Training Checklist

### Verify All Checkpoints

After training, verify all checkpoints load correctly:

```bash
# Test loading each checkpoint (note: pretrained_model subdirectory)
cd ${LEROBOT_REPO_DIR}
for checkpoint_dir in ./outputs_pi05_*/checkpoints/*/pretrained_model; do
    echo "Testing: $checkpoint_dir"
    python -s -c "
from lerobot.common.policies.pi05.modeling_pi05 import Pi05Policy
policy = Pi05Policy.from_pretrained('$checkpoint_dir')
print('✅ Loaded successfully: $checkpoint_dir')
    " || echo "❌ Failed: $checkpoint_dir"
done
```

**Checklist**:
- [ ] Primary checkpoint (20k, unfrozen, full) loads successfully
- [ ] Frozen vision checkpoint (20k) loads successfully
- [ ] Expert-only checkpoint (20k) loads successfully
- [ ] 10k checkpoint loads successfully
- [ ] 15k checkpoint loads successfully

---

### Organize Checkpoints

Create a checkpoint inventory document:

```bash
# List all checkpoints with metadata
cat > checkpoint_inventory.txt << 'EOF'
Checkpoint Inventory for Pi0.5 Evaluation
==========================================

1. Primary (20k, FROZEN vision, full finetune) ⭐ PRIMARY
   Path: ./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model
   Config: freeze_vision_encoder=TRUE, train_expert_only=false
   Purpose: Primary evaluation, vision ablation baseline, expert ablation baseline

2. Unfrozen Vision (20k)
   Path: ./outputs_pi05_unfrozen_vision_20k/checkpoints/020000/pretrained_model
   Config: freeze_vision_encoder=false, train_expert_only=false
   Purpose: Vision encoder ablation comparison

3. Expert-Only (20k, frozen vision)
   Path: ./outputs_pi05_expert_only_20k/checkpoints/020000/pretrained_model
   Config: freeze_vision_encoder=TRUE, train_expert_only=true
   Purpose: Expert-only training ablation

4. Steps Ablation - 10k
   Path: ./outputs_pi05_primary_20k/checkpoints/010000/pretrained_model (or ./outputs_pi05_steps_10k/checkpoints/010000/pretrained_model)
   Config: freeze_vision_encoder=TRUE, train_expert_only=false, steps=10000
   Purpose: Training steps ablation

5. Steps Ablation - 15k
   Path: ./outputs_pi05_primary_20k/checkpoints/015000/pretrained_model (or ./outputs_pi05_steps_15k/checkpoints/015000/pretrained_model)
   Config: freeze_vision_encoder=TRUE, train_expert_only=false, steps=15000
   Purpose: Training steps ablation

6. Steps Ablation - 20k
   Path: ./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model (same as #1)
   Config: freeze_vision_encoder=TRUE, train_expert_only=false, steps=20000
   Purpose: Training steps ablation
EOF

cat checkpoint_inventory.txt
```

**Checklist**:
- [ ] Create checkpoint inventory document
- [ ] Document exact paths for each checkpoint
- [ ] Verify disk space (each checkpoint ~50GB)
- [ ] Backup checkpoints to external storage if needed

---

### WandB Export

Export training curves and metrics from WandB:

```bash
# Install wandb if not already
pip install wandb

# Export training data
wandb export pi05_g1_evaluation --format csv --output ./wandb_exports/
```

**Checklist**:
- [ ] Export training loss curves for all runs
- [ ] Export validation loss curves for all runs
- [ ] Export learning rate schedules
- [ ] Save plots to: `./results/figures/training_curves/`

---

## Evaluation Commands Reference

Once checkpoints are ready, use these commands for evaluation.

### Open-Loop Evaluation Template

Run on held-out episodes from the dataset to measure action prediction accuracy.

```bash
cd ${LEROBOT_REPO_DIR} && \
python -s unitree_lerobot/eval_robot/eval_g1_dataset.py \
    --policy.path=<CHECKPOINT_PATH>/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --root="" \
    --episodes=10 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=false \
    --send_real_robot=false
```

**Example - Evaluate Primary Checkpoint**:
```bash
cd ${LEROBOT_REPO_DIR} && \
python -s unitree_lerobot/eval_robot/eval_g1_dataset.py \
    --policy.path=./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --root="" \
    --episodes=10 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=false \
    --send_real_robot=false \
    > results/open_loop/primary_20k_eval.log 2>&1
```

### Synthetic Open-Loop Evaluation (with frame modes)

For testing with different background conditions:

```bash
cd ${LEROBOT_REPO_DIR} && \
python -s unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py \
    --policy.path=<CHECKPOINT_PATH>/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --root="" \
    --episodes=10 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=false \
    --send_real_robot=false \
    --frame_mode=white  # Options: white, random
```

### Closed-Loop Evaluation Template

Run on real robot for task success evaluation.

```bash
cd ${LEROBOT_REPO_DIR} && \
python -s unitree_lerobot/eval_robot/eval_g1.py \
    --policy.path=<CHECKPOINT_PATH>/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --root="" \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --send_real_robot=true \
    --custom_task="pick up the <OBJECT> and place it in the <TARGET>"
```

**Example - Evaluate Primary Checkpoint on Seen Object**:
```bash
cd ${LEROBOT_REPO_DIR} && \
python -s unitree_lerobot/eval_robot/eval_g1.py \
    --policy.path=./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --root="" \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --send_real_robot=true \
    --custom_task="pick up the blue cube and place it in the box"
```

**Optional Parameters**:
- `--policy.n_action_steps=30` - Override number of action steps per inference

---

## Checkpoint Priority Order

If you have limited compute resources, train in this priority order:

### High Priority (Required)
1. ✅ **Primary (20k, frozen vision, full finetune)** - Serves as baseline for all comparisons
2. ✅ **Unfrozen Vision (20k)** - Critical for vision encoder ablation comparison
3. ✅ **10k steps** - Use intermediate checkpoint from primary run (no extra training needed)

### Medium Priority (Important)
4. **Expert-Only (20k, frozen vision)** - Useful for understanding training efficiency
5. **15k steps** - Use intermediate checkpoint from primary run

### Low Priority (Optional)
6. Separate training runs for 10k/15k if intermediate checkpoints not saved

---

## Training Time Estimates

Based on A100 40GB GPU:

| Checkpoint | Steps | Estimated Time | Priority |
|------------|-------|----------------|----------|
| Primary 20k (frozen vision) | 20,000 | ~10-14 hours | ⭐⭐⭐ High |
| Unfrozen Vision 20k | 20,000 | ~12-16 hours | ⭐⭐⭐ High |
| Expert-Only 20k (frozen vision) | 20,000 | ~8-12 hours | ⭐⭐ Medium |
| Steps 10k | 10,000 | ~5-7 hours (or use saved) | ⭐⭐⭐ High |
| Steps 15k | 15,000 | ~8-10 hours (or use saved) | ⭐⭐ Medium |

**Total Training Time**: ~30-45 hours if training all from scratch  
**Optimized**: ~20-30 hours if reusing intermediate checkpoints

---

## Troubleshooting

### Out of Memory (OOM)

If training fails with OOM:
1. Reduce batch size: `--batch_size=16` or `--batch_size=8`
2. Increase gradient accumulation: `--gradient_accumulation_steps=2`
3. Reduce chunk size: `--policy.chunk_size=25`

### Checkpoint Not Saving

Verify save frequency and output directory:
```bash
ls -lh ./outputs_pi05_*/checkpoints/
```

Expected structure:
```
outputs_pi05_primary_20k/
├── checkpoints/
│   ├── 005000/
│   ├── 010000/
│   ├── 015000/
│   └── 020000/
└── logs/
```

### WandB Connection Issues

```bash
# Check WandB status
wandb status

# Re-login if needed
wandb login --relogin
```

---

## Final Checklist Summary

### Training Phase
- [ ] All 3 primary training runs completed (primary, frozen vision, expert-only)
- [ ] All checkpoints verified to load successfully
- [ ] WandB logs accessible for all runs
- [ ] Checkpoint inventory document created
- [ ] Intermediate checkpoints (10k, 15k) identified/saved

### Pre-Evaluation Phase
- [ ] All open-loop evaluation scripts tested in dry-run mode
- [ ] Closed-loop evaluation scripts tested in dry-run mode
- [ ] Results directories created (`./results/open_loop/`, `./results/closed_loop/`)
- [ ] Checkpoint paths documented in evaluation scripts

### Ready for Evaluation
- [ ] Proceed to **experiment_checklist.md** for evaluation protocol
- [ ] Select seen/unseen objects for closed-loop evaluation
- [ ] Schedule robot access time

---

## Quick Reference: All Checkpoint Paths

After training, you should have:

```bash
# Primary evaluation checkpoint (FROZEN VISION)
./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model

# Vision encoder ablation
./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model          # Frozen baseline (PRIMARY)
./outputs_pi05_unfrozen_vision_20k/checkpoints/020000/pretrained_model  # Unfrozen comparison

# Expert-only ablation
./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model       # Full finetune baseline (PRIMARY)
./outputs_pi05_expert_only_20k/checkpoints/020000/pretrained_model   # Expert-only comparison

# Training steps ablation (all frozen vision)
./outputs_pi05_primary_20k/checkpoints/010000/pretrained_model  # 10k
./outputs_pi05_primary_20k/checkpoints/015000/pretrained_model  # 15k
./outputs_pi05_primary_20k/checkpoints/020000/pretrained_model  # 20k (same as primary)
```

---

**Document Created**: 2026-04-04  
**Author**: Copilot  
**Related Documents**: context.md, research_plan.md, experiment_checklist.md
