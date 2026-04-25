#!/usr/bin/env python
"""
Batch open-loop evaluation across all checkpoints.

For each checkpoint, runs:
  1) Normal open-loop eval         (eval_g1_dataset.py)
  2) Synthetic white frames eval   (eval_g1_dataset_synthetic.py --frame_mode=white)
  3) Synthetic random frames eval  (eval_g1_dataset_synthetic.py --frame_mode=random)

Saves per-checkpoint figures + a combined comparison grid.

Usage:
    conda activate unitree_lerobot_synced
    python -s batch_open_loop_eval.py

All configuration lives in the CONFIG dict below — edit it to match your run.
"""

import os
import sys
import shutil
import subprocess
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend — no display needed
import matplotlib.pyplot as plt
from matplotlib.image import imread

# ============================================================
# CONFIGURATION — edit these to match your training run
# ============================================================
CONFIG = {
    # Path that contains checkpoints/ subdirectory
    "training_output_dir": (
        "/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/"
        "unitree_IL_lerobot_synced/unitree_lerobot/lerobot/"
        "outputs_pi05_single_arm_dual_cam"
    ),
    # Which checkpoints to evaluate (None = all numeric ones, sorted)
    "checkpoints": None,          # e.g. ["002000", "010000", "020000"]
    # Dataset & robot config (same for every run)
    "repo_id":    "deepansh-methdai/single_arm_dual_cam",
    "root":       "",
    "episodes":   10,
    "frequency":  30,
    "arm":        "G1_29",
    "ee":         "dex3",
    # Where to store all output figures
    "output_dir": (
        "/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/"
        "unitree_IL_lerobot_synced/open_loop_eval_results"
    ),
    # Working directory for running the eval scripts
    "cwd": (
        "/home/deepansh/drive2/humanoid_ws/src/lerobot_dir/"
        "unitree_IL_lerobot_synced"
    ),
}

# Eval script paths (relative to cwd)
EVAL_NORMAL    = "unitree_lerobot/eval_robot/eval_g1_dataset.py"
EVAL_SYNTHETIC = "unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py"

# Scenarios: (label, script, extra_args)
SCENARIOS = [
    ("normal",          EVAL_NORMAL,    []),
    ("synthetic_white", EVAL_SYNTHETIC, ["--frame_mode=white"]),
    ("synthetic_random", EVAL_SYNTHETIC, ["--frame_mode=random"]),
]


def discover_checkpoints(training_dir: str) -> list[str]:
    """Return sorted list of numeric checkpoint folder names."""
    ckpt_root = Path(training_dir) / "checkpoints"
    folders = []
    for p in sorted(ckpt_root.iterdir()):
        if p.is_dir() and re.fullmatch(r"\d+", p.name):
            folders.append(p.name)
    return folders


def run_eval(
    script: str,
    policy_path: str,
    extra_args: list[str],
    cwd: str,
) -> int:
    """Launch an eval script as a subprocess, auto-feeding 's' to stdin."""
    cmd = [
        sys.executable, "-s", script,
        f"--policy.path={policy_path}",
        f"--repo_id={CONFIG['repo_id']}",
        f"--root={CONFIG['root']}",
        f"--episodes={CONFIG['episodes']}",
        f"--frequency={CONFIG['frequency']}",
        f"--arm={CONFIG['arm']}",
        f"--ee={CONFIG['ee']}",
        "--visualization=false",
        "--send_real_robot=false",
        *extra_args,
    ]
    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    proc = subprocess.run(
        cmd,
        input="s\n",           # auto-press 's'
        text=True,
        cwd=cwd,
        env=os.environ.copy(),
    )
    return proc.returncode


def collect_figure(src_name: str, dst_path: str, cwd: str):
    """Move a figure from cwd to the destination, creating dirs as needed."""
    src = Path(cwd) / src_name
    if src.exists():
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst_path)
        print(f"  ✓ Saved {dst_path}")
    else:
        print(f"  ✗ Expected {src} but it does not exist!")


def build_combined_figure(output_dir: str, checkpoints: list[str]):
    """
    Build a grid figure:  rows = checkpoints, cols = scenarios.
    Each cell shows the per-checkpoint evaluation plot.
    """
    n_ckpts = len(checkpoints)
    n_scenarios = len(SCENARIOS)

    fig, axes = plt.subplots(
        n_ckpts, n_scenarios,
        figsize=(8 * n_scenarios, 6 * n_ckpts),
        squeeze=False,
    )

    for row, ckpt in enumerate(checkpoints):
        for col, (label, _, _) in enumerate(SCENARIOS):
            ax = axes[row][col]
            img_path = Path(output_dir) / ckpt / f"{label}.png"
            if img_path.exists():
                img = imread(str(img_path))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center",
                        fontsize=14, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label.replace("_", " ").title(), fontsize=14)
            if col == 0:
                ax.set_ylabel(f"ckpt {ckpt}", fontsize=12)

    fig.suptitle("Open-Loop Evaluation — All Checkpoints × Scenarios", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    combined_path = Path(output_dir) / "combined_comparison.png"
    fig.savefig(str(combined_path), dpi=120)
    plt.close(fig)
    print(f"\n✓ Combined comparison figure saved to {combined_path}")


def main():
    output_dir = CONFIG["output_dir"]
    cwd = CONFIG["cwd"]
    training_dir = CONFIG["training_output_dir"]

    checkpoints = CONFIG["checkpoints"] or discover_checkpoints(training_dir)
    if not checkpoints:
        print("No checkpoints found — exiting.")
        return

    print(f"Checkpoints to evaluate: {checkpoints}")
    print(f"Output directory: {output_dir}\n")

    # Figure filenames produced by the eval scripts (hardcoded in those scripts)
    FIGURE_MAP = {
        "normal":           "figure.png",
        "synthetic_white":  "figure_synthetic_white.png",
        "synthetic_random": "figure_synthetic_random.png",
    }

    for ckpt in checkpoints:
        policy_path = str(
            Path(training_dir) / "checkpoints" / ckpt / "pretrained_model"
        )
        if not Path(policy_path).exists():
            print(f"⚠ Skipping checkpoint {ckpt}: {policy_path} not found")
            continue

        for label, script, extra_args in SCENARIOS:
            rc = run_eval(script, policy_path, extra_args, cwd)
            if rc != 0:
                print(f"  ✗ {label} for checkpoint {ckpt} exited with code {rc}")

            # Move the figure into the organized output folder
            src_name = FIGURE_MAP[label]
            dst = str(Path(output_dir) / ckpt / f"{label}.png")
            collect_figure(src_name, dst, cwd)

    # Build the combined comparison grid
    build_combined_figure(output_dir, checkpoints)
    print("\nDone — all evaluations complete.")


if __name__ == "__main__":
    main()
