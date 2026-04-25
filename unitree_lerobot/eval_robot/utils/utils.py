import json
import os
import numpy as np
import torch
from typing import Any
from contextlib import nullcontext
from copy import copy
import logging
from dataclasses import dataclass, field
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in [1, 3]:
                value = np.transpose(value, (2, 0, 1))
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if not use_dataset:
                # Skip non-tensor observations (like task strings)
                if not hasattr(observation[name], "unsqueeze"):
                    continue
                if "images" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()

            observation[name] = observation[name].unsqueeze(0).to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        action = postprocessor(action)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def reset_policy(policy: PreTrainedPolicy):
    policy.reset()


def cleanup_resources(image_info: dict[str, Any]):
    """Safely close and unlink shared memory resources."""
    logger_mp.info("Cleaning up shared memory resources.")
    for shm in image_info["shm_resources"]:
        if shm:
            shm.close()
            shm.unlink()


def to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_scalar(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().ravel()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


def compute_metrics(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    action_names: list,
) -> dict:
    """Compute per-joint and overall MAE / MSE / RMSE.

    Args:
        ground_truth: shape (T, D)
        predicted:    shape (T, D)
        action_names: list of D joint name strings

    Returns:
        dict with keys: action_names, mae_per_joint, mse_per_joint, rmse_per_joint,
                        mae_overall, mse_overall, rmse_overall
    """
    err = predicted - ground_truth
    mae_per_joint = np.mean(np.abs(err), axis=0).tolist()
    mse_per_joint = np.mean(err ** 2, axis=0).tolist()
    rmse_per_joint = np.sqrt(np.array(mse_per_joint)).tolist()
    return {
        "action_names": list(action_names),
        "mae_per_joint": mae_per_joint,
        "mse_per_joint": mse_per_joint,
        "rmse_per_joint": rmse_per_joint,
        "mae_overall": float(np.mean(mae_per_joint)),
        "mse_overall": float(np.mean(mse_per_joint)),
        "rmse_overall": float(np.sqrt(np.mean(mse_per_joint))),
    }


def print_metrics_table(metrics: dict) -> None:
    """Print a formatted per-joint + overall metrics table via logger."""
    names = metrics["action_names"]
    col_w = max(len(n) for n in names + ["OVERALL"]) + 2
    header = f"{'Joint':<{col_w}}{'MAE':>10}{'MSE':>10}{'RMSE':>10}"
    sep = "─" * len(header)
    lines = [sep, header, sep]
    for name, mae, mse, rmse in zip(
        names,
        metrics["mae_per_joint"],
        metrics["mse_per_joint"],
        metrics["rmse_per_joint"],
    ):
        lines.append(f"{name:<{col_w}}{mae:>10.5f}{mse:>10.5f}{rmse:>10.5f}")
    lines.append(sep)
    lines.append(
        f"{'OVERALL':<{col_w}}"
        f"{metrics['mae_overall']:>10.5f}"
        f"{metrics['mse_overall']:>10.5f}"
        f"{metrics['rmse_overall']:>10.5f}"
    )
    lines.append(sep)
    logger_mp.info("\n" + "\n".join(lines))


def save_metrics_json(metrics: dict, cfg, n_episodes: int, n_timesteps: int, output_path: str = "") -> str:
    """Save metrics + run metadata to a JSON file.

    Args:
        metrics:      output of compute_metrics()
        cfg:          EvalRealConfig instance
        n_episodes:   number of episodes evaluated
        n_timesteps:  total timesteps across all episodes
        output_path:  explicit path; if "", auto-generates from checkpoint name

    Returns:
        path of the written file
    """
    checkpoint = getattr(getattr(cfg, "policy", None), "pretrained_path", "unknown")
    if not output_path:
        ckpt_name = os.path.basename(checkpoint.rstrip("/")) if checkpoint != "unknown" else "unknown"
        output_path = f"eval_results_{ckpt_name}.json"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    payload = {
        "checkpoint": checkpoint,
        "dataset": cfg.repo_id,
        "n_episodes": n_episodes,
        "n_timesteps": n_timesteps,
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger_mp.info(f"Metrics saved to: {output_path}")
    return output_path


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None

    root: str = ""
    episodes: int = 0
    frequency: float = 30.0

    # Basic control parameters
    arm: str = "G1_29"  # G1_29, G1_23
    ee: str = "dex3"  # dex3, dex1, inspire1, brainco

    # Mode flags
    motion: bool = False
    headless: bool = False
    visualization: bool = False
    send_real_robot: bool = False
    use_dataset: bool = False
    single: bool = False

    rename_map: dict[str, str] = field(default_factory=dict)
    custom_task: str = ""
    output_path: str = ""  # where to save metrics JSON; auto-named if empty

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
