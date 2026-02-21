"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any
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
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.gripper_converter import GripperConverter, detect_dataset_mode

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


LEFT_ARM_Q = [0.0881, 0.0373, 0.5046, 1.1915, 0.0040, 0.2412, -0.0439]


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        # --- Setup Gripper Converter ---
        # Detect if dataset uses gripper (16D/8D) or full hand (28D/14D) representation
        dataset_info = detect_dataset_mode(dataset.meta)
        gripper_converter = GripperConverter(ee_type=cfg.ee) if cfg.ee else None
        is_single_arm = dataset_info["arm_type"] == "single"
        logger_mp.info(f"Dataset mode: {dataset_info['mode']} ({dataset_info['state_dim']}D), arm_type: {dataset_info['arm_type']}")

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        
        # For single arm, dataset has 7 arm joints (right arm only)
        # But IK expects 14D, so we need to pad with zeros for left arm
        if is_single_arm:
            right_arm_init = step["observation.state"][:7].cpu().numpy()
            left_arm_init = np.array(LEFT_ARM_Q, dtype=np.float32) if cfg.single else np.zeros(7, dtype=np.float32)
            init_arm_pose = np.concatenate([left_arm_init, right_arm_init])
        else:
            init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        task_command = cfg.custom_task if cfg.custom_task else step.get("task", "")
        logger_mp.info(f"Using task command: {task_command!r}")

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        prev_action_np = None  # For action smoothing
        smoothing_alpha = 0.4  # 0.0 = use previous action, 1.0 = use new action only
        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move
            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                
                # Build state tensor matching the dataset format
                if is_single_arm:
                    # Single arm: use right arm (7D) + right hand (7D or 1D gripper)
                    # current_arm_q is 14D [left(7), right(7)], extract right arm
                    right_arm_q = current_arm_q[7:14]
                    
                    if gripper_converter and dataset_info["mode"] == "gripper":
                        # Compact right hand to gripper value
                        from unitree_lerobot.eval_robot.utils.gripper_converter import estimate_hand_openness
                        gripper_value = estimate_hand_openness(right_ee_state, hand="right")
                        state_np = np.concatenate([right_arm_q, np.array([gripper_value], dtype=np.float32)])
                    else:
                        # Full hand mode
                        state_np = np.concatenate([right_arm_q, right_ee_state])
                else:
                    # Dual arm: use both arms (14D) + both hands
                    if gripper_converter and dataset_info["mode"] == "gripper":
                        # Compact both hands to gripper values
                        from unitree_lerobot.eval_robot.utils.gripper_converter import estimate_hand_openness
                        left_gripper = estimate_hand_openness(left_ee_state, hand="left")
                        right_gripper = estimate_hand_openness(right_ee_state, hand="right")
                        state_np = np.concatenate([
                            current_arm_q,
                            np.array([left_gripper, right_gripper], dtype=np.float32)
                        ])
                    else:
                        # Full hand mode
                        state_np = np.concatenate([current_arm_q, left_ee_state, right_ee_state])
                
                state_tensor = torch.from_numpy(state_np).float()
                observation["observation.state"] = state_tensor
                # 2. Get Action from Policy
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    task_command,
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                action_np = action.cpu().numpy()

                # # Apply exponential smoothing to reduce jerky motion
                # if prev_action_np is not None:
                #     action_np = smoothing_alpha * action_np + (1 - smoothing_alpha) * prev_action_np
                # prev_action_np = action_np.copy()

                # 3. Execute Action
                # For single arm, pad left arm with zeros to get 14D for IK
                if is_single_arm:
                    right_arm_action = action_np[:7]
                    left_arm_action = np.array(LEFT_ARM_Q, dtype=np.float32) if cfg.single else np.zeros(7, dtype=np.float32)
                    arm_action = np.concatenate([left_arm_action, right_arm_action])
                else:
                    arm_action = action_np[:arm_dof]
                    
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee and gripper_converter:
                    # Use gripper converter to handle gripper/full and single/dual arm
                    # For single arm, use arm_dof=7 so converter correctly detects 8D action
                    converter_arm_dof = 7 if is_single_arm else arm_dof
                    ee_actions = gripper_converter.get_ee_actions(action_np, converter_arm_dof)
                    
                    if is_single_arm:
                        # Single arm: only one end-effector (right arm/hand)
                        ee_action = ee_actions[0]
                        # logger_mp.info(f"EE Action: {ee_action}")
                        
                        # For single arm, the EE goes to the RIGHT hand
                        if isinstance(ee_shared_mem["right"], SynchronizedArray):
                            ee_shared_mem["right"][:] = to_list(ee_action)
                        elif hasattr(ee_shared_mem["right"], "value"):
                            ee_shared_mem["right"].value = to_scalar(ee_action)
                    else:
                        # Dual arm: two end-effectors
                        left_ee_action, right_ee_action = ee_actions
                        # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                        if isinstance(ee_shared_mem["left"], SynchronizedArray):
                            ee_shared_mem["left"][:] = to_list(left_ee_action)
                            ee_shared_mem["right"][:] = to_list(right_ee_action)
                        elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                            ee_shared_mem["left"].value = to_scalar(left_ee_action)
                            ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
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
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
