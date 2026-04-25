# Pi0.5 Evaluation Experiment Checklist

This checklist is ordered to **minimize environment/configuration switching** during robot experiments.

---

## Phase 1: Preparation (No Robot Required)

### 1.1 Checkpoint Inventory
- [ ] List all available checkpoints and their configurations
- [ ] Verify checkpoint paths are accessible
- [ ] Document training config for each checkpoint (frozen/unfrozen, steps, etc.)

### 1.2 Dataset Preparation
- [ ] Verify `single_arm_dual_cam` dataset is accessible
- [ ] Identify held-out episodes for open-loop evaluation
- [ ] Count total episodes in dataset

### 1.3 Object Selection
- [ ] Select 2 **seen objects** (objects present in training dataset)
  - Seen Object 1: _______________
  - Seen Object 2: _______________
- [ ] Select 2 **unseen objects** (novel objects NOT in training data)
  - Unseen Object 1: _______________
  - Unseen Object 2: _______________
- [ ] Document object properties (size, shape, color, graspability)

### 1.4 Evaluation Script Setup
- [ ] Test open-loop eval script runs without errors
- [ ] Test closed-loop eval script runs without errors (simulation or dry-run)
- [ ] Prepare logging/recording setup (CSV template, video recording if needed)

---

## Phase 2: Open-Loop Evaluation (No Robot Required)

Run all open-loop evaluations before touching the robot.

### 2.1 Primary Checkpoint Open-Loop
- [ ] Run open-loop eval on primary checkpoint (20k steps)
- [ ] Record MSE, MAE, per-joint errors
- [ ] Save results to: `results/open_loop/primary_checkpoint.csv`

### 2.2 Ablation Checkpoints Open-Loop

#### Training Steps Ablation
- [ ] Run open-loop eval on 10k step checkpoint
- [ ] Run open-loop eval on 15k step checkpoint (if available)
- [ ] Run open-loop eval on 20k step checkpoint
- [ ] Save results to: `results/open_loop/training_steps_ablation.csv`

#### Vision Encoder Ablation
- [ ] Run open-loop eval on frozen vision checkpoint
- [ ] Run open-loop eval on unfrozen vision checkpoint
- [ ] Save results to: `results/open_loop/vision_encoder_ablation.csv`

#### Train Expert Only Ablation
- [ ] Run open-loop eval on expert-only checkpoint (if available)
- [ ] Run open-loop eval on full fine-tune checkpoint
- [ ] Save results to: `results/open_loop/expert_only_ablation.csv`

### 2.3 WandB Data Extraction
- [ ] Export training loss curves
- [ ] Export validation loss curves
- [ ] Save plots to: `results/figures/training_curves/`

---

## Phase 3: Closed-Loop Evaluation (Robot Required)

**IMPORTANT**: Group experiments to minimize setup changes.

### 3.1 Environment Setup (Once)
- [ ] Position robot at evaluation station
- [ ] Calibrate camera positions
- [ ] Verify robot connectivity and control
- [ ] Set up consistent lighting
- [ ] Mark object placement positions for reproducibility
- [ ] Test inference script with dummy run

### 3.2 Primary Evaluation - SEEN OBJECTS

Load primary checkpoint once, run all seen object trials.

#### Seen Object 1: _______________
| Trial | Success | Failure Mode | Notes |
|-------|---------|--------------|-------|
| 1 | ☐ | | |
| 2 | ☐ | | |
| 3 | ☐ | | |
| 4 | ☐ | | |
| 5 | ☐ | | |
| 6 | ☐ | | |
| 7 | ☐ | | |
| 8 | ☐ | | |
| 9 | ☐ | | |
| 10 | ☐ | | |

**Success Rate**: ___/10 = ___%

#### Seen Object 2: _______________
| Trial | Success | Failure Mode | Notes |
|-------|---------|--------------|-------|
| 1 | ☐ | | |
| 2 | ☐ | | |
| 3 | ☐ | | |
| 4 | ☐ | | |
| 5 | ☐ | | |
| 6 | ☐ | | |
| 7 | ☐ | | |
| 8 | ☐ | | |
| 9 | ☐ | | |
| 10 | ☐ | | |

**Success Rate**: ___/10 = ___%

### 3.3 Primary Evaluation - UNSEEN OBJECTS

Same checkpoint, switch to unseen objects.

#### Unseen Object 1: _______________
| Trial | Success | Failure Mode | Notes |
|-------|---------|--------------|-------|
| 1 | ☐ | | |
| 2 | ☐ | | |
| 3 | ☐ | | |
| 4 | ☐ | | |
| 5 | ☐ | | |
| 6 | ☐ | | |
| 7 | ☐ | | |
| 8 | ☐ | | |
| 9 | ☐ | | |
| 10 | ☐ | | |

**Success Rate**: ___/10 = ___%

#### Unseen Object 2: _______________
| Trial | Success | Failure Mode | Notes |
|-------|---------|--------------|-------|
| 1 | ☐ | | |
| 2 | ☐ | | |
| 3 | ☐ | | |
| 4 | ☐ | | |
| 5 | ☐ | | |
| 6 | ☐ | | |
| 7 | ☐ | | |
| 8 | ☐ | | |
| 9 | ☐ | | |
| 10 | ☐ | | |

**Success Rate**: ___/10 = ___%

### 3.4 Vision Encoder Ablation (Closed-Loop)

Quick test with visual perturbation (e.g., colored cloth under object).

#### Frozen Vision Encoder Checkpoint
- [ ] Add visual perturbation (describe: _______________)
- [ ] Run 5 trials on Seen Object 1

| Trial | Success | Failure Mode | Notes |
|-------|---------|--------------|-------|
| 1 | ☐ | | |
| 2 | ☐ | | |
| 3 | ☐ | | |
| 4 | ☐ | | |
| 5 | ☐ | | |

**Success Rate**: ___/5 = ___%

#### Unfrozen Vision Encoder Checkpoint
- [ ] Same visual perturbation
- [ ] Run 5 trials on Seen Object 1

| Trial | Success | Failure Mode | Notes |
|-------|---------|--------------|-------|
| 1 | ☐ | | |
| 2 | ☐ | | |
| 3 | ☐ | | |
| 4 | ☐ | | |
| 5 | ☐ | | |

**Success Rate**: ___/5 = ___%

---

## Phase 4: Analysis & Report

### 4.1 Data Compilation
- [ ] Compile all trial results into summary tables
- [ ] Calculate overall success rates with confidence intervals
- [ ] Categorize failure modes and compute frequencies

### 4.2 Figure Generation
- [ ] Bar chart: Success rate (Seen vs Unseen objects)
- [ ] Bar chart: Success rate by failure mode
- [ ] Line plot: Training/validation loss curves
- [ ] Bar chart: Open-loop MSE across checkpoints
- [ ] Bar chart: Vision encoder ablation comparison

### 4.3 Report Writing
- [ ] Introduction & motivation
- [ ] Background (Pi0.5, Unitree G1, VLA models)
- [ ] Methodology (dataset, training, evaluation protocol)
- [ ] Results (primary evaluation + ablations)
- [ ] Failure mode analysis
- [ ] Discussion & conclusions
- [ ] References

### 4.4 Final Review
- [ ] Proofread report
- [ ] Verify all figures/tables are referenced
- [ ] Check numerical consistency
- [ ] Final submission/presentation

---

## Failure Mode Reference

Use these codes when logging failures:

| Code | Failure Mode | Description |
|------|--------------|-------------|
| GM | Grasp Miss | Robot attempts grasp but misses object |
| GS | Grasp Slip | Object grasped but slips during manipulation |
| WO | Wrong Object | Robot approaches/grasps wrong object |
| PM | Placement Miss | Object grasped but placed in wrong location |
| CO | Collision | Robot collides with environment |
| TO | Timeout | Task not completed within time limit |
| OT | Other | Unexpected failure (describe in notes) |

---

## Quick Commands Reference

### Open-Loop Evaluation
```bash
python -s unitree_lerobot/eval_robot/eval_g1_dataset.py \
    --policy.path=<CHECKPOINT_PATH> \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --root="" \
    --episodes=10 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --visualization=false \
    --send_real_robot=false
```

### Closed-Loop Inference (Real Robot)
```bash
python -s unitree_lerobot/eval_robot/eval_g1.py \
    --policy.path=<CHECKPOINT_PATH> \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --root="" \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3" \
    --send_real_robot=true \
    --single=true \
    --custom_task="pick up the <OBJECT> and place it in the <TARGET>"
```

---

## Results Summary (Fill After Completion)

### Primary Evaluation Results

| Object Type | Object | Success Rate | Top Failure Mode |
|-------------|--------|--------------|------------------|
| Seen | Object 1 | __/10 = __% | |
| Seen | Object 2 | __/10 = __% | |
| Unseen | Object 1 | __/10 = __% | |
| Unseen | Object 2 | __/10 = __% | |
| **Overall Seen** | | __/20 = __% | |
| **Overall Unseen** | | __/20 = __% | |

### Ablation Results

| Ablation | Condition | Open-Loop MSE | Closed-Loop (if done) |
|----------|-----------|---------------|----------------------|
| Vision Encoder | Frozen | | __/5 = __% |
| Vision Encoder | Unfrozen | | __/5 = __% |
| Training Steps | 10k | | N/A |
| Training Steps | 15k | | N/A |
| Training Steps | 20k | | N/A |
