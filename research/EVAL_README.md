# Open-Loop Evaluation — Changes & Usage Guide

## What Changed

### `unitree_lerobot/eval_robot/run_open_loop_suite.py` *(updated)*

**GPU cleanup between checkpoints.**  
`_run_single_eval()` now wraps all eval work in a `try/finally` that calls
`_destroy_policy()` after each run — moves the model to CPU, deletes all references,
forces `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.synchronize()`.
This prevents CUDA OOM when loading successive checkpoints on a 16 GB GPU.

**Synthetic-image eval integrated.**  
After the real-image eval, the suite loops over `synthetic_frame_modes` from
`eval_suite.yaml` and calls `eval_g1_dataset_synthetic.eval_policy()` with the
**already-loaded policy** (no extra GPU load per mode). Saves
`metrics_synthetic_<mode>.json` and `figure_synthetic_<mode>.png` to the same
run directory.

**3-panel comparison.png.**  
When synthetic results are present the comparison plot gains a third panel:
"Vision Robustness — Overall RMSE by Frame Mode", showing a grouped bar chart of
overall RMSE across real / white / black / random frames for each run label.

**Note on re-downloads**: local-path checkpoints (starting with `./outputs_pi/...`)
are read **entirely from disk**. `PreTrainedConfig.from_pretrained()` detects a local
directory and never contacts HuggingFace Hub. No re-downloads occur between runs.

---

### `unitree_lerobot/eval_robot/utils/utils.py`

Three new helper functions and one new config field were added.

**`compute_metrics(ground_truth, predicted, action_names)`**  
Takes two `(T, D)` numpy arrays and a list of joint-name strings. Returns a dict with
`mae_per_joint`, `mse_per_joint`, `rmse_per_joint` (each a list of `D` floats) and
scalar `mae_overall`, `mse_overall`, `rmse_overall`.

**`print_metrics_table(metrics)`**  
Logs a formatted table (via `logger_mp`) showing MAE / MSE / RMSE for every joint plus
an OVERALL row.

**`save_metrics_json(metrics, cfg, n_episodes, n_timesteps, output_path="")`**  
Writes a JSON file containing the metrics dict plus run metadata (checkpoint path,
dataset repo ID, episode/timestep counts). If `output_path` is empty the file is
auto-named `eval_results_<checkpoint_name>.json` in the working directory.

**`EvalRealConfig.output_path: str = ""`**  
New CLI flag `--output_path=...` lets you specify exactly where the JSON is written.
Leave empty for the auto-named default.

---

### `unitree_lerobot/eval_robot/eval_g1_dataset.py`

**Episode loop fixed.**  
Previously hardcoded to episode 0.  
Now respects `--episodes=N` (`cfg.episodes`): evaluates the first N episodes, or all
episodes when N = 0.  Policy / preprocessor / postprocessor are reset between episodes.

**Metrics computed and reported.**  
After all episodes finish, `compute_metrics()` is called on the concatenated arrays,
`print_metrics_table()` logs the result to console, and `save_metrics_json()` writes
the JSON file.

**Per-joint RMSE in plot titles.**  
Each subplot now reads  
`<joint_name>  (RMSE=X.XXXX  MAE=X.XXXX)`  
so errors are visible at a glance without opening the JSON.

**`eval_policy()` now returns the metrics dict** so the suite launcher can use it
directly without re-reading the JSON.

---

### `unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py`

Same changes as `eval_g1_dataset.py`.  
Additionally, the auto-generated JSON filename includes the frame mode, e.g.  
`eval_results_<checkpoint>_synthetic_white.json`  
so real and synthetic runs never collide.

---

### `research/eval_suite.yaml` *(new)*

Template checkpoint registry.  Fill in the checkpoint paths and run the suite.
Checkpoints still set to the `/path/to/...` placeholder are silently skipped so
you can do partial runs as checkpoints become available.

---

### `unitree_lerobot/eval_robot/run_open_loop_suite.py` *(new)*

Master launcher that iterates over all ablation groups and runs defined in
`eval_suite.yaml`.  For each run it:

1. Loads the policy and dataset
2. Calls `eval_policy()` (the same function used by `eval_g1_dataset.py`)
3. Saves `metrics.json` + `figure.png` to `<output_dir>/<group>/<label>/`
4. After every run in a group completes (≥ 2 runs), generates a
   `comparison.png` grouped bar chart

---

## How to Use

### Step 1 — Fill in your checkpoint paths

Open `research/eval_suite.yaml` and replace the `/path/to/...` placeholders:

```yaml
ablations:
  training_steps:
    runs:
      - label: 10k
        checkpoint: /your/actual/path/checkpoint_10k
      - label: 20k
        checkpoint: /your/actual/path/checkpoint_20k
```

### Step 2 — Run the full suite

```bash
python -s unitree_lerobot/eval_robot/run_open_loop_suite.py \
    --config=research/eval_suite.yaml
```

### Step 3 — Or run a single ablation group

```bash
python -s unitree_lerobot/eval_robot/run_open_loop_suite.py \
    --config=research/eval_suite.yaml \
    --only=training_steps
```

### Standalone single-checkpoint eval (unchanged CLI)

```bash
python -s unitree_lerobot/eval_robot/eval_g1_dataset.py \
    --policy.path=<CHECKPOINT_PATH> \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --episodes=10 --frequency=30 --arm=G1_29 --ee=dex3 \
    --send_real_robot=false --visualization=false \
    --output_path=results/my_run/metrics.json
```

### Synthetic-image eval (ablation A visual robustness)

```bash
python -s unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py \
    --policy.path=<CHECKPOINT_PATH> \
    --repo_id=deepansh-methdai/single_arm_dual_cam \
    --episodes=10 --arm=G1_29 --ee=dex3 \
    --send_real_robot=false --frame_mode=white
```

---

## Output Folder Structure

With `synthetic_frame_modes: [white]` set in `eval_suite.yaml`:

```
results/open_loop/
├── primary/
│   └── 30k/
│       ├── metrics.json
│       ├── figure.png
│       ├── metrics_synthetic_white.json   ← synthetic
│       └── figure_synthetic_white.png     ← synthetic
│
├── training_steps/
│   ├── 10k/  {metrics.json, figure.png, metrics_synthetic_white.json, figure_synthetic_white.png}
│   ├── 20k/  {same}
│   ├── 30k/  {same}
│   └── comparison.png   ← 3-panel: overall MAE/RMSE, per-joint RMSE, vision robustness
│
├── vision_encoder/
│   ├── frozen/   {same 4 files}
│   ├── unfrozen/ {same 4 files}
│   └── comparison.png
│
└── expert_only/
    ├── expert_only/   {same 4 files}
    ├── full_finetune/ {same 4 files}
    └── comparison.png
```

To disable synthetic eval entirely, set `synthetic_frame_modes: []` in `eval_suite.yaml`.

---

## metrics.json Schema

```json
{
  "checkpoint": "/path/to/checkpoint",
  "dataset": "deepansh-methdai/single_arm_dual_cam",
  "n_episodes": 10,
  "n_timesteps": 1500,
  "metrics": {
    "action_names": ["joint_0", "joint_1", ..., "gripper"],
    "mae_per_joint":  [0.012, 0.008, ...],
    "mse_per_joint":  [0.0002, 0.0001, ...],
    "rmse_per_joint": [0.014, 0.010, ...],
    "mae_overall":  0.011,
    "mse_overall":  0.00015,
    "rmse_overall": 0.012
  }
}
```

---

## comparison.png Layout

Each `comparison.png` is a 2- or 3-panel figure:

- **Panel 1**: grouped bar chart — overall MAE and RMSE (real images), one group per run label
- **Panel 2**: per-joint RMSE (real images), one group of bars per joint, one bar per run label
- **Panel 3** *(only when `synthetic_frame_modes` is non-empty)*: overall RMSE grouped by
  frame mode (real / white / black / random) for each run label — the "vision robustness" view

Useful for dropping directly into the research report (Section 4 / Ablation results).
