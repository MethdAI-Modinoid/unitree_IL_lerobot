"""
Master launcher for all open-loop evaluation runs defined in eval_suite.yaml.

For each ablation group × checkpoint run it:
  1. Loads the policy and dataset
  2. Calls eval_policy() from eval_g1_dataset (real images)
  3. Optionally runs eval_policy() from eval_g1_dataset_synthetic for each
     frame mode listed in suite_cfg["synthetic_frame_modes"] — reusing the
     already-loaded policy, so no extra GPU load per mode.
  4. Saves metrics.json + figure.png (+ per-mode synthetic equivalents) to
     <output_dir>/<group>/<label>/
  5. Explicitly destroys the policy and clears the CUDA cache before loading
     the next checkpoint, preventing OOM errors on successive runs.
  6. After every run in a group finishes, generates a combined comparison.png

Usage
-----
# Run the full suite
python -s unitree_lerobot/eval_robot/run_open_loop_suite.py \
    --config=research/eval_suite.yaml

# Run only one ablation group
python -s unitree_lerobot/eval_robot/run_open_loop_suite.py \
    --config=research/eval_suite.yaml \
    --only=training_steps
"""

import argparse
import gc
import json
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_PREFIX = "/path/to/"


def _is_placeholder(path: str) -> bool:
    return path.startswith(_PLACEHOLDER_PREFIX)


def _load_suite_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def _destroy_policy(policy, preprocessor, postprocessor):
    """Aggressively free GPU memory after a checkpoint run.

    Mirrors the _destroy_policy() pattern from
    unitree_lerobot/lerobot/examples/rtc/eval_dataset.py:386-420.
    """
    logger.info("  Releasing GPU memory...")
    for obj in (policy, preprocessor, postprocessor):
        try:
            if hasattr(obj, "cpu"):
                obj.cpu()
        except Exception:
            pass
    del policy, preprocessor, postprocessor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("  GPU memory released.")


def _run_single_eval(
    suite_cfg: dict, checkpoint: str, run_dir: Path
) -> tuple[dict, dict[str, dict]] | None:
    """Load policy + dataset, run real + synthetic evals, return (real_metrics, synthetic_metrics).

    Returns:
        (real_metrics, synthetic_metrics_by_mode) where synthetic_metrics_by_mode
        maps frame_mode -> metrics dict for each mode in suite_cfg["synthetic_frame_modes"].
        Returns None if eval_policy returns None (user aborted).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.utils.utils import get_safe_torch_device

    from unitree_lerobot.eval_robot.eval_g1_dataset import eval_policy
    from unitree_lerobot.eval_robot.eval_g1_dataset_synthetic import (
        eval_policy as eval_policy_synth,
    )
    from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(run_dir / "metrics.json")

    # Temporarily inject --policy.path into sys.argv so that
    # EvalRealConfig.__post_init__ (which calls parser.get_path_arg) finds the
    # pretrained path and doesn't warn about random weights.
    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0], f"--policy.path={checkpoint}"]
    try:
        eval_cfg = EvalRealConfig(
            repo_id=suite_cfg["repo_id"],
            episodes=suite_cfg.get("episodes", 10),
            frequency=suite_cfg.get("frequency", 30.0),
            arm=suite_cfg.get("arm", "G1_29"),
            ee=suite_cfg.get("ee", "dex3"),
            send_real_robot=False,
            visualization=False,
            output_path=metrics_path,
        )
    finally:
        sys.argv = old_argv

    policy_cfg = eval_cfg.policy  # loaded by __post_init__ from pretrained path

    device = get_safe_torch_device(policy_cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = LeRobotDataset(repo_id=suite_cfg["repo_id"])
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
    )

    # Monkey-patch plt.savefig so all figures (real + synthetic) land in run_dir
    _orig_savefig = plt.savefig

    def _patched_savefig(fname, *args, **kwargs):
        if not os.path.isabs(str(fname)):
            fname = str(run_dir / os.path.basename(str(fname)))
        return _orig_savefig(fname, *args, **kwargs)

    plt.savefig = _patched_savefig

    metrics = None
    synthetic_metrics_by_mode: dict[str, dict] = {}

    try:
        with torch.no_grad(), (
            torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext()
        ):
            # ── Real-image eval ──────────────────────────────────────────────
            metrics = eval_policy(eval_cfg, dataset, policy, preprocessor, postprocessor)

            if metrics is None:
                return None  # user aborted

            # ── Synthetic-image evals (reuse already-loaded policy) ──────────
            for mode in suite_cfg.get("synthetic_frame_modes", []) or []:
                logger.info(f"    Running synthetic eval  frame_mode={mode}")
                synth_cfg = replace(
                    eval_cfg,
                    output_path=str(run_dir / f"metrics_synthetic_{mode}.json"),
                )
                synth_metrics = eval_policy_synth(
                    synth_cfg, dataset, mode, policy, preprocessor, postprocessor
                )
                if synth_metrics is not None:
                    synthetic_metrics_by_mode[mode] = synth_metrics
                    logger.info(
                        f"    Synthetic [{mode}] RMSE={synth_metrics['rmse_overall']:.5f}"
                    )

    finally:
        plt.savefig = _orig_savefig
        plt.close("all")
        # ── Explicit GPU cleanup — prevents OOM on the next checkpoint ───────
        _destroy_policy(policy, preprocessor, postprocessor)

    return metrics, synthetic_metrics_by_mode


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def make_comparison_plot(
    group_name: str,
    run_labels: list,
    run_metrics: list,
    output_path: str,
    synthetic_run_metrics: dict[str, list] | None = None,
):
    """Grouped bar chart comparing runs within an ablation group.

    Panels:
      1 — Overall MAE and RMSE per run label (real images)
      2 — Per-joint RMSE per run label (real images)
      3 — Overall RMSE by frame mode (real + each synthetic mode), if available
    """
    n_runs = len(run_labels)
    action_names = run_metrics[0]["action_names"]
    n_joints = len(action_names)
    colors = plt.cm.tab10.colors

    has_synthetic = bool(synthetic_run_metrics)
    n_panels = 3 if has_synthetic else 2
    fig_height = 10 * n_panels // 2 + 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(10, n_runs * 3), fig_height))
    if n_panels == 1:
        axes = [axes]
    ax_top, ax_bot = axes[0], axes[1]
    ax_synth = axes[2] if has_synthetic else None

    fig.suptitle(f"Ablation: {group_name}", fontsize=14, fontweight="bold")

    # ── Panel 1: overall MAE + RMSE ─────────────────────────────────────────
    x = np.arange(n_runs)
    width = 0.35
    mae_vals = [m["mae_overall"] for m in run_metrics]
    rmse_vals = [m["rmse_overall"] for m in run_metrics]

    bars_mae = ax_top.bar(x - width / 2, mae_vals, width, label="MAE", color=colors[0], alpha=0.85)
    bars_rmse = ax_top.bar(x + width / 2, rmse_vals, width, label="RMSE", color=colors[1], alpha=0.85)

    for bar in list(bars_mae) + list(bars_rmse):
        ax_top.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8,
        )

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(run_labels)
    ax_top.set_ylabel("Error (rad / m)")
    ax_top.set_title("Overall MAE and RMSE (real images)")
    ax_top.legend()
    ax_top.set_ylim(bottom=0)

    # ── Panel 2: per-joint RMSE ──────────────────────────────────────────────
    x_joints = np.arange(n_joints)
    bar_width = 0.8 / n_runs

    for i, (label, m) in enumerate(zip(run_labels, run_metrics)):
        offsets = x_joints + (i - (n_runs - 1) / 2) * bar_width
        bars = ax_bot.bar(
            offsets, m["rmse_per_joint"], bar_width,
            label=label, color=colors[i % len(colors)], alpha=0.85,
        )
        for bar in bars:
            ax_bot.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax_bot.set_xticks(x_joints)
    ax_bot.set_xticklabels(action_names, rotation=30, ha="right", fontsize=8)
    ax_bot.set_ylabel("RMSE (rad / m)")
    ax_bot.set_title("Per-Joint RMSE (real images)")
    ax_bot.legend()
    ax_bot.set_ylim(bottom=0)

    # ── Panel 3: vision robustness (real vs synthetic per run) ───────────────
    if has_synthetic and ax_synth is not None:
        all_modes = ["real"] + list(synthetic_run_metrics.keys())
        n_modes = len(all_modes)
        bar_w = 0.8 / n_modes
        x_runs = np.arange(n_runs)

        for j, mode in enumerate(all_modes):
            if mode == "real":
                mode_rmse = [m["rmse_overall"] for m in run_metrics]
            else:
                mode_rmse = [
                    synthetic_run_metrics[mode][i]["rmse_overall"]
                    if i < len(synthetic_run_metrics[mode])
                    else 0.0
                    for i in range(n_runs)
                ]
            offsets = x_runs + (j - (n_modes - 1) / 2) * bar_w
            bars = ax_synth.bar(
                offsets, mode_rmse, bar_w,
                label=mode, color=colors[j % len(colors)], alpha=0.85,
            )
            for bar in bars:
                ax_synth.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8,
                )

        ax_synth.set_xticks(x_runs)
        ax_synth.set_xticklabels(run_labels)
        ax_synth.set_ylabel("Overall RMSE (rad / m)")
        ax_synth.set_title("Vision Robustness: Overall RMSE by Frame Mode")
        ax_synth.legend(title="frame mode")
        ax_synth.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Comparison plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run the full open-loop evaluation suite.")
    ap.add_argument("--config", required=True, help="Path to eval_suite.yaml")
    ap.add_argument(
        "--only",
        default=None,
        help="Run only this ablation group (e.g. training_steps). Default: run all.",
    )
    args = ap.parse_args()

    suite_cfg = _load_suite_config(args.config)
    output_dir = Path(suite_cfg.get("output_dir", "results/open_loop"))

    ablations = suite_cfg.get("ablations", {})
    if args.only:
        if args.only not in ablations:
            logger.error(f"--only={args.only} not found in config. Available: {list(ablations)}")
            sys.exit(1)
        ablations = {args.only: ablations[args.only]}

    for group_name, group_cfg in ablations.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Ablation group: {group_name}")
        logger.info(f"{'=' * 60}")

        runs = group_cfg.get("runs", [])
        completed_labels: list[str] = []
        completed_metrics: list[dict] = []
        # synthetic_by_mode[mode] is a list aligned with completed_labels
        synthetic_by_mode: dict[str, list[dict]] = {}

        for run in runs:
            label = run["label"]
            checkpoint = run["checkpoint"]

            if _is_placeholder(checkpoint):
                logger.warning(
                    f"  [{group_name}/{label}] SKIPPED — checkpoint path not set: {checkpoint}"
                )
                continue

            if not os.path.exists(checkpoint):
                logger.warning(
                    f"  [{group_name}/{label}] SKIPPED — checkpoint not found: {checkpoint}"
                )
                continue

            run_dir = output_dir / group_name / label
            logger.info(f"  Running [{group_name}/{label}]  checkpoint={checkpoint}")

            try:
                result = _run_single_eval(suite_cfg, checkpoint, run_dir)
            except Exception as exc:
                logger.error(f"  [{group_name}/{label}] FAILED: {exc}", exc_info=True)
                # Ensure GPU is cleared even on failure
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if result is None:
                logger.warning(f"  [{group_name}/{label}] Aborted by user.")
                continue

            metrics, synth_metrics_for_run = result
            completed_labels.append(label)
            completed_metrics.append(metrics)

            # Accumulate synthetic metrics aligned by mode
            for mode, sm in synth_metrics_for_run.items():
                synthetic_by_mode.setdefault(mode, []).append(sm)

            logger.info(
                f"  [{group_name}/{label}] Done — "
                f"overall RMSE={metrics['rmse_overall']:.5f}  MAE={metrics['mae_overall']:.5f}"
            )

        # Generate comparison plot if ≥2 runs completed
        if len(completed_labels) >= 2:
            comparison_path = str(output_dir / group_name / "comparison.png")
            make_comparison_plot(
                group_name,
                completed_labels,
                completed_metrics,
                comparison_path,
                synthetic_run_metrics=synthetic_by_mode or None,
            )
        elif len(completed_labels) == 1:
            logger.info(
                f"  [{group_name}] Only 1 run completed — skipping comparison plot "
                f"(need ≥ 2 runs)."
            )
        else:
            logger.warning(f"  [{group_name}] No runs completed.")

    logger.info("\nSuite finished.")


if __name__ == "__main__":
    main()
