# Pi0.5 Evaluation on Unitree G1: Research Plan

## 1. Problem Statement

Evaluate the Pi0.5 Vision-Language-Action (VLA) model for pick-and-place manipulation on the Unitree G1 humanoid robot. The research focuses on:

1. **Primary Evaluation**: Quantify pick-and-place performance on seen and unseen objects
2. **Targeted Ablations**: Analyze impact of training configurations (vision encoder freezing, training steps, train_expert_only)

## 2. Scope & Constraints

| Constraint | Value |
|------------|-------|
| **Timeline** | ~15 working days |
| **Robot Access** | ~7-8 days (50% of working days) |
| **Robot Configuration** | Unitree G1, single arm, dual camera, 8D gripper (7 arm DOF + 1 gripper) |
| **Dataset** | `deepansh-methdai/single_arm_dual_cam` |
| **Primary Checkpoint** | Best checkpoint from `single_arm_dual_cam` training |
| **Output** | Formal written report with tables, figures, analysis |

## 3. Research Questions

1. **RQ1**: What is the pick-and-place success rate of Pi0.5 on seen vs unseen objects?
2. **RQ2**: What are the common failure modes and how do they differ between seen/unseen objects?
3. **RQ3**: How does freezing the vision encoder affect generalization to visual perturbations?
4. **RQ4**: How do training steps correlate with open-loop action prediction accuracy?

## 4. Experimental Design

### 4.1 Primary Evaluation

#### Closed-Loop (Real Robot)
- **Objects**: 2 seen objects + 2 unseen objects
- **Trials**: 10 per object = 40 total trials
- **Metrics**:
  - Success rate (binary: object successfully placed in target location)
  - Failure mode categorization

#### Open-Loop (Dataset Evaluation)
- **Method**: Run policy on held-out episodes from dataset
- **Metrics**:
  - Mean Squared Error (MSE) between predicted and ground-truth actions
  - Mean Absolute Error (MAE)
  - Per-joint error breakdown

### 4.2 Targeted Ablations

#### Ablation A: Vision Encoder Freezing

| Checkpoint | Training Config |
|------------|-----------------|
| Frozen | `--policy.freeze_vision_encoder=true` |
| Unfrozen | `--policy.freeze_vision_encoder=false` |

**Evaluation**:
1. Open-loop MSE on same episodes (no extra setup)
2. 1 simple visual perturbation test: same object with colored cloth/backdrop change (closed-loop, ~5 trials)

#### Ablation B: Training Steps

| Checkpoint | Steps |
|------------|-------|
| Early | 10,000 steps |
| Mid | 20,000 steps |
| Final | 30,000 steps |

**Evaluation**:
1. Training/validation loss curves from WandB
2. Open-loop MSE at each checkpoint

#### Ablation C: Train Expert Only

| Checkpoint | Config |
|------------|--------|
| Expert Only | `--policy.train_expert_only=true` (freezes VLM, ~3B params trainable) |
| Full Fine-tune | `--policy.train_expert_only=false` |

**Evaluation**:
1. Open-loop MSE comparison

### 4.3 Failure Mode Categories

| Category | Description |
|----------|-------------|
| **Grasp Miss** | Robot attempts grasp but misses object |
| **Grasp Slip** | Object grasped but slips during manipulation |
| **Wrong Object** | Robot approaches/grasps wrong object |
| **Placement Miss** | Object grasped but placed in wrong location |
| **Collision** | Robot collides with environment |
| **Timeout** | Task not completed within time limit |
| **Other** | Unexpected failure mode |

## 5. Data & Checkpoints

### 5.1 Dataset
- **Repo ID**: `deepansh-methdai/single_arm_dual_cam`
- **Configuration**: 8D (7 arm + 1 gripper), dual camera
- **FPS**: 30Hz
- **Use**: Training data + held-out episodes for open-loop eval

### 5.2 Checkpoints to Evaluate

| Checkpoint | Description | Ablation Purpose |
|------------|-------------|------------------|
| `single_arm_dual_cam_20k` | Primary (best) | Primary evaluation |
| Frozen vision encoder variant | If available | Ablation A |
| 10k, 15k, 20k step checkpoints | From same training run | Ablation B |
| Expert-only variant | If available | Ablation C |

## 6. Commands Reference

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
    --custom_task="<TASK_DESCRIPTION>"
```

### Training (Reference)
```bash
HF_LEROBOT_HOME=<LEROBOT_DATA_PATH> \
python -s src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=deepansh-methdai/single_arm_dual_cam \
    --policy.type=pi05 \
    --output_dir=./outputs_pi05_single_arm_dual_cam/ \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=<true|false> \
    --policy.train_expert_only=<true|false> \
    --policy.optimizer_lr=2.5e-5 \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=<STEPS> \
    --batch_size=32 \
    --save_freq=5000
```

## 7. Timeline

### Phase 1: Setup & Open-Loop (Days 1-3)
- [ ] Inventory existing checkpoints and verify paths
- [ ] Run open-loop evaluation on all checkpoints
- [ ] Extract WandB training curves
- [ ] Select best checkpoint for primary evaluation
- [ ] Prepare seen/unseen object sets

### Phase 2: Primary Closed-Loop Evaluation (Days 4-8)
- [ ] Set up evaluation environment (table, camera positions)
- [ ] Run 10 trials on Seen Object 1
- [ ] Run 10 trials on Seen Object 2
- [ ] Run 10 trials on Unseen Object 1
- [ ] Run 10 trials on Unseen Object 2
- [ ] Log success/failure and failure modes for each trial

### Phase 3: Ablation Closed-Loop (Days 9-11)
- [ ] Visual perturbation test with frozen vision checkpoint (~5 trials)
- [ ] Visual perturbation test with unfrozen vision checkpoint (~5 trials)
- [ ] Any additional quick ablation tests

### Phase 4: Analysis & Report Writing (Days 12-15)
- [ ] Compile results into tables
- [ ] Generate figures (success rate bar charts, loss curves, MSE comparisons)
- [ ] Write failure mode analysis
- [ ] Draft introduction, methodology, results, discussion
- [ ] Final report polish

## 8. Deliverables

1. **Formal Report** containing:
   - Introduction & motivation
   - Background on Pi0.5 and Unitree G1
   - Methodology (dataset, training, evaluation protocols)
   - Results (tables, figures)
   - Failure mode analysis
   - Ablation study results
   - Discussion & conclusions

2. **Experiment Logs**:
   - CSV/JSON files with trial-by-trial results
   - Video recordings (optional)
   - WandB dashboards

3. **Code Documentation**:
   - Evaluation scripts used
   - Reproducibility instructions

## 9. Key Training Parameters (Pi0.5)

| Parameter | Default | Your Config |
|-----------|---------|-------------|
| `chunk_size` | 50 | 50 |
| `n_action_steps` | 50 | 50 (or custom) |
| `freeze_vision_encoder` | false | varies |
| `train_expert_only` | false | varies |
| `batch_size` | 32 | 32 |
| `optimizer_lr` | 2.5e-5 | 2.5e-5 |
| `dtype` | bfloat16 | bfloat16 |
| `control_freq` | 30Hz | 30Hz |

## 10. Key Insights from Literature

### From [Black2024] Pi0 Paper
- **Flow Matching Architecture**: Pi0 uses flow matching on top of pre-trained VLM for precise continuous action generation
- **Cross-Embodiment Pre-training**: Model pre-trained on 7 distinct robot configurations and 68 tasks
- **Fine-tuning Strategy**: Zero-shot capability after pre-training, then fine-tune for specific downstream tasks

### From [Black2025] Pi0.5 Paper
- **Co-training on Heterogeneous Data**: 97.6% of training data comes from sources OTHER than the target task (web data, other robots, semantic prediction)
- **Hierarchical Inference**: Model predicts semantic subtask first, then low-level actions
- **Generalization Levels**: Physical (how to grasp), Semantic (where to put things), Environmental (adapting to new scenes)

### From [Omaisan2025] LoRA Fine-Tuning Paper
- **Frozen vs Unfrozen Vision Encoder**: With sufficient data (200+ episodes), unfrozen vision achieves 76% success rate
- **Data Requirement**: ~200 episodes minimum for reliable deployment (>70% success)
- **LoRA Parameters**: ~8.4M trainable params (frozen) vs ~33M (unfrozen) out of 3.1B total

### From [Liu2026] Continual Learning Paper
- **Catastrophic Forgetting**: VLAs are surprisingly resistant to forgetting during continual learning
- **LIBERO Benchmark**: Standard benchmark for evaluating VLA manipulation policies

### From [Inayat-Hussain2025] Thesis
- Reference for humanoid-specific VLA evaluation methodology

## 11. Notes

- **RTC (Real-Time Correction)**: Not used in this evaluation per scope decision. Can be explored in future work.
- **Seen vs Unseen Objects**: "Seen" objects appear in training dataset. "Unseen" are novel objects the model has never seen during training.
- **Robot Time Optimization**: Group experiments to minimize environment switching (see sequential experiment list to be provided).
- **Dataset Size Consideration**: Your `single_arm_dual_cam` dataset should have sufficient episodes based on Omaisan findings (~200+ recommended).
