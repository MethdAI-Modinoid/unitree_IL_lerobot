"""
Open-loop evaluation with synthetic (white / black / random) image frames.
Everything else (state, ground-truth actions, metrics, plotting) is identical
to eval_g1_dataset.py.

Usage examples:
    python -s unitree_lerobot/eval_robot/eval_g1_dataset_synthetic.py \
        --policy.path=<checkpoint_path> \
        --repo_id=deepansh-methdai/single_camera_single_gripper \
        --root="" --episodes=10 --frequency=30 \
        --arm="G1_29" --ee="dex3" \
        --visualization=false --send_real_robot=false \
        --frame_mode=white

    Supported --frame_mode values: white, black, random
"""

import torch
import tqdm
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pformat
from typing import Any
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)

from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
    compute_metrics,
    print_metrics_table,
    save_metrics_json,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.gripper_converter import GripperConverter, detect_dataset_mode

import os

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_image(original_tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a synthetic image tensor with the same shape/dtype as *original_tensor*.

    The dataset stores images as (C, H, W) float tensors in [0, 1].
    """
    if mode == "white":
        return torch.ones_like(original_tensor)
    elif mode == "black":
        return torch.zeros_like(original_tensor)
    elif mode == "random":
        return torch.rand_like(original_tensor)
    else:
        raise ValueError(f"Unknown frame_mode: {mode!r}. Choose from white, black, random.")


def extract_observation_synthetic(step: dict, frame_mode: str) -> dict:
    """Same as extract_observation but replaces every image with a synthetic frame."""
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            observation[key] = _make_synthetic_image(value, frame_mode)
        elif key == "observation.state":
            observation[key] = value

    return observation


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    frame_mode: str,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")
    logger_mp.info(f"Synthetic frame mode: {frame_mode}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # --- Setup Gripper Converter ---
    dataset_info = detect_dataset_mode(dataset.meta)
    gripper_converter = GripperConverter(ee_type=cfg.ee) if cfg.ee else None
    is_single_arm = dataset_info["arm_type"] == "single"
    logger_mp.info(f"Dataset mode: {dataset_info['mode']} ({dataset_info['state_dim']}D), arm_type: {dataset_info['arm_type']}")

    # Determine how many episodes to evaluate
    total_episodes = dataset.num_episodes
    n_episodes = cfg.episodes if cfg.episodes > 0 else total_episodes
    n_episodes = min(n_episodes, total_episodes)
    logger_mp.info(f"Evaluating {n_episodes} / {total_episodes} episodes  (frame_mode={frame_mode})")

    # Get joint names once
    first_step = dataset[dataset.meta.episodes["dataset_from_index"][0]]
    n_dims = first_step["action"].shape[0]
    action_names = (
        dataset.meta.features["action"]["names"][0]
        if "names" in dataset.meta.features["action"]
        else [f"Dim_{i}" for i in range(n_dims)]
    )

    if cfg.send_real_robot:
        from unitree_lerobot.eval_robot.make_robot import setup_robot_interface

        robot_interface = setup_robot_interface(cfg)
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )

        if is_single_arm:
            right_arm_init = first_step["observation.state"][:7].cpu().numpy()
            left_arm_init = np.zeros(7, dtype=np.float32)
            init_arm_pose = np.concatenate([left_arm_init, right_arm_init])
        else:
            init_arm_pose = first_step["observation.state"][:arm_dof].cpu().numpy()

    # ===============init robot=====================
    if cfg.send_real_robot:
        user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
        if user_input.lower() != "s":
            return None
        logger_mp.info("Initializing robot to starting pose...")
        tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
        robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
        time.sleep(1)

    all_episode_gt = []
    all_episode_pred = []

    for ep_idx in range(n_episodes):
        policy.reset()
        if preprocessor is not None:
            preprocessor.reset()
        if postprocessor is not None:
            postprocessor.reset()

        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]

        ground_truth_actions = []
        predicted_actions = []

        logger_mp.info(f"Episode {ep_idx + 1}/{n_episodes}  steps [{from_idx}, {to_idx})")

        for step_idx in tqdm.tqdm(range(from_idx, to_idx), desc=f"Episode {ep_idx + 1}"):
            loop_start_time = time.perf_counter()

            step = dataset[step_idx]
            observation = extract_observation_synthetic(step, frame_mode)

            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )
            action_np = action.cpu().numpy()

            ground_truth_actions.append(step["action"].numpy())
            predicted_actions.append(action_np)

            if cfg.send_real_robot:
                if is_single_arm:
                    right_arm_action = action_np[:7]
                    left_arm_action = np.zeros(7, dtype=np.float32)
                    arm_action = np.concatenate([left_arm_action, right_arm_action])
                else:
                    arm_action = action_np[:arm_dof]

                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee and gripper_converter:
                    converter_arm_dof = 7 if is_single_arm else arm_dof
                    ee_actions = gripper_converter.get_ee_actions(action_np, converter_arm_dof)

                    if is_single_arm:
                        ee_action = ee_actions[0]
                        if isinstance(ee_shared_mem["right"], SynchronizedArray):
                            ee_shared_mem["right"][:] = to_list(ee_action)
                        elif hasattr(ee_shared_mem["right"], "value"):
                            ee_shared_mem["right"].value = to_scalar(ee_action)
                    else:
                        left_ee_action, right_ee_action = ee_actions
                        if isinstance(ee_shared_mem["left"], SynchronizedArray):
                            ee_shared_mem["left"][:] = to_list(left_ee_action)
                            ee_shared_mem["right"][:] = to_list(right_ee_action)
                        elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                            ee_shared_mem["left"].value = to_scalar(left_ee_action)
                            ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if cfg.visualization:
                visualization_data(step_idx, observation, observation["observation.state"], action_np, rerun_logger)

            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

        all_episode_gt.append(np.array(ground_truth_actions))
        all_episode_pred.append(np.array(predicted_actions))

    # ── Aggregate across all episodes ─────────────────────────────────────────
    all_gt = np.concatenate(all_episode_gt, axis=0)
    all_pred = np.concatenate(all_episode_pred, axis=0)
    n_timesteps = all_gt.shape[0]

    metrics = compute_metrics(all_gt, all_pred, action_names)
    print_metrics_table(metrics)

    # Auto-name output JSON to distinguish from real-image runs
    out_path = cfg.output_path
    if not out_path:
        import os as _os
        ckpt = getattr(getattr(cfg, "policy", None), "pretrained_path", "unknown")
        ckpt_name = _os.path.basename(ckpt.rstrip("/")) if ckpt != "unknown" else "unknown"
        out_path = f"eval_results_{ckpt_name}_synthetic_{frame_mode}.json"

    save_metrics_json(metrics, cfg, n_episodes=n_episodes, n_timesteps=n_timesteps,
                      output_path=out_path)

    # ── Plot ───────────────────────────────────────────────────────────────────
    gt_plot = all_episode_gt[-1]
    pred_plot = all_episode_pred[-1]
    n_dims = gt_plot.shape[1]

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
    fig.suptitle(
        f"Ground Truth vs Predicted Actions  "
        f"(frame_mode={frame_mode}, ep {n_episodes}, overall RMSE={metrics['rmse_overall']:.4f})"
    )

    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes
        ax.plot(gt_plot[:, i], label="Ground Truth", color="blue")
        ax.plot(pred_plot[:, i], label="Predicted", color="red", linestyle="--")
        ax.set_title(
            f"{action_names[i]}  (RMSE={metrics['rmse_per_joint'][i]:.4f}  "
            f"MAE={metrics['mae_per_joint'][i]:.4f})",
            fontsize=9,
        )
        ax.legend(fontsize=7)

    if n_dims > 1:
        axes[-1].set_xlabel("Timestep")
    else:
        axes.set_xlabel("Timestep")

    plt.tight_layout()
    fig_path = f"figure_synthetic_{frame_mode}.png"
    plt.savefig(fig_path)
    logger_mp.info(f"Saved plot to {fig_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI helper: extract --frame_mode before draccus parser sees it
# ---------------------------------------------------------------------------

import sys


def _pop_frame_mode_arg() -> str:
    """Remove --frame_mode=<val> from sys.argv and return the value."""
    mode = "white"  # default
    new_argv = []
    for arg in sys.argv:
        if arg.startswith("--frame_mode="):
            mode = arg.split("=", 1)[1]
        elif arg == "--frame_mode":
            pass
        else:
            new_argv.append(arg)
    for i, arg in enumerate(sys.argv):
        if arg == "--frame_mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            new_argv = [a for a in new_argv if a != mode or sys.argv.index(a) != i + 1]
            break
    sys.argv = new_argv
    assert mode in ("white", "black", "random"), \
        f"--frame_mode must be one of white, black, random. Got: {mode!r}"
    return mode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))
    logging.info(f"Synthetic frame mode: {_FRAME_MODE}")

    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, _FRAME_MODE, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    _FRAME_MODE = _pop_frame_mode_arg()
    eval_main()
