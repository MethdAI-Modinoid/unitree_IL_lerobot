#!/usr/bin/env python3
"""
=============================================================================
Convert Unitree G1 Dex3 Dataset DOF Configuration
=============================================================================

This script modifies a LeRobot dataset by converting the observation.state 
and action to different DOF configurations:

DOF Modes (--dof-mode):
  - dual_gripper:     28D -> 16D (14 arm + 2 gripper states) [default]
  - dual_full:        28D -> 28D (14 arm + 14 hand joints) - no conversion
  - single_gripper:   28D ->  8D (7 right arm + 1 right gripper)
  - single_full:      28D -> 14D (7 right arm + 7 right hand joints)

Note: Single arm modes always use the RIGHT arm.

The conversion process:
1. Read observation.state and action data (28 keys)
2. Extract arm states based on mode (left+right or right only)
3. Convert hand states to gripper if using gripper mode
4. Optionally remove camera features
5. Create new dataset with converted features

Usage:
    python -s convert_hands_to_gripper.py \\
        --repo-id <source_repo_id> \\
        --output-repo-id <output_repo_id> \\
        --dof-mode <mode> \\
        [--root <path>] \\
        [--remove-cameras cam1,cam2] \\
        [--push-to-hub]

Example (dual arm with gripper):
    python -s convert_hands_to_gripper.py \\
        --repo-id deepansh-methdai/three_camera \\
        --output-repo-id deepansh-methdai/three_camera_gripper \\
        --dof-mode dual_gripper \\
        --root ./lerobot_data/deepansh-methdai/three_camera

Example (single arm with gripper, remove left cameras):
    python -s convert_hands_to_gripper.py \\
        --repo-id deepansh-methdai/three_camera \\
        --output-repo-id deepansh-methdai/three_camera_single \\
        --dof-mode single_gripper \\
        --remove-cameras observation.images.cam_left_high,observation.images.cam_left_wrist \\
        --root ./lerobot_data/deepansh-methdai/three_camera

Author: Generated for Unitree G1 robot dataset conversion
=============================================================================
"""

import dataclasses
import logging
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from tqdm import tqdm

# Add lerobot to path
LEROBOT_PATH = Path(__file__).parent / "unitree_lerobot" / "lerobot" / "src"
sys.path.insert(0, str(LEROBOT_PATH))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


# =============================================================================
# SECTION 1: DOF Mode Definitions
# =============================================================================

class DofMode(Enum):
    """DOF configuration modes for dataset conversion."""
    DUAL_GRIPPER = "dual_gripper"      # 16D: 14 arm + 2 gripper
    DUAL_FULL = "dual_full"            # 28D: 14 arm + 14 hand (no conversion)
    SINGLE_GRIPPER = "single_gripper"  # 8D: 7 right arm + 1 gripper
    SINGLE_FULL = "single_full"        # 14D: 7 right arm + 7 right hand


DOF_MODE_INFO = {
    DofMode.DUAL_GRIPPER: {
        "input_dim": 28,
        "output_dim": 16,
        "description": "Dual arm with gripper states (14 arm + 2 gripper)",
    },
    DofMode.DUAL_FULL: {
        "input_dim": 28,
        "output_dim": 28,
        "description": "Dual arm with full hand joints (14 arm + 14 hand)",
    },
    DofMode.SINGLE_GRIPPER: {
        "input_dim": 28,
        "output_dim": 8,
        "description": "Single right arm with gripper (7 arm + 1 gripper)",
    },
    DofMode.SINGLE_FULL: {
        "input_dim": 28,
        "output_dim": 14,
        "description": "Single right arm with full hand (7 arm + 7 hand)",
    },
}


# =============================================================================
# SECTION 1: Hand Pose Constants (from sample_gripper_control.py)
# =============================================================================

DEX3_NUM_MOTORS = 7

# Predefined hand poses for estimation
HAND_POSES = {
    "left": {
        "open": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "close": np.array([
            0,      # Thumb0
            1.0,    # Thumb1
            1.74,   # Thumb2
            -1.57,  # Middle0
            -1.74,  # Middle1
            -1.57,  # Index0
            -1.74,  # Index1
        ], dtype=np.float32),
    },
    "right": {
        "open": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "close": np.array([
            0,      # Thumb0
            -1.0,   # Thumb1 (negative for right hand)
            -1.74,  # Thumb2 (negative for right hand)
            1.57,   # Index0 (positive for right hand)
            1.74,   # Index1
            1.57,   # Middle0
            1.74,   # Middle1
        ], dtype=np.float32),
    },
}


# =============================================================================
# SECTION 2: Gripper State Estimation Functions
# =============================================================================

def estimate_hand_openness(joint_positions: np.ndarray, hand: str = "left") -> float:
    """
    Estimate how open/closed a hand is based on current joint positions.
    
    Uses per-joint projection with averaging:
    - For each joint, compute t_i = (q_i - q_open_i) / (q_close_i - q_open_i)
    - Clip to [0, 1] and average all valid values
    
    Args:
        joint_positions: 7D array of current joint angles (radians)
        hand: "left" or "right"
        
    Returns:
        float: Estimated openness from 0.0 (fully open) to 1.0 (fully closed)
    """
    open_pose = HAND_POSES[hand]["open"]
    close_pose = HAND_POSES[hand]["close"]
    
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    
    t_values = []
    for i in range(DEX3_NUM_MOTORS):
        delta = close_pose[i] - open_pose[i]
        
        # Skip joints that don't move between open and closed
        if abs(delta) < 1e-6:
            continue
        
        # Compute interpolation parameter for this joint
        t_i = (joint_positions[i] - open_pose[i]) / delta
        
        # Clip to [0, 1] range
        t_i = np.clip(t_i, 0.0, 1.0)
        t_values.append(t_i)
    
    if len(t_values) == 0:
        return 0.0
    
    return float(np.mean(t_values))


def convert_state_to_gripper(state_28d: np.ndarray) -> np.ndarray:
    """
    Convert 28D state (14 arm + 14 hand joints) to 16D (14 arm + 2 gripper states).
    
    Input format (28 values):
        [0:7]   - Left arm joints (7 DOF)
        [7:14]  - Right arm joints (7 DOF)
        [14:21] - Left hand joints (7 DOF)
        [21:28] - Right hand joints (7 DOF)
    
    Output format (16 values):
        [0:7]   - Left arm joints (7 DOF)
        [7:14]  - Right arm joints (7 DOF)
        [14]    - Left gripper state (0.0=open, 1.0=closed)
        [15]    - Right gripper state (0.0=open, 1.0=closed)
    
    Args:
        state_28d: 28-dimensional state array
        
    Returns:
        np.ndarray: 16-dimensional state array
    """
    # Extract components
    left_arm = state_28d[0:7]
    right_arm = state_28d[7:14]
    left_hand = state_28d[14:21]
    right_hand = state_28d[21:28]
    
    # Estimate gripper states
    left_gripper = estimate_hand_openness(left_hand, hand="left")
    right_gripper = estimate_hand_openness(right_hand, hand="right")
    
    # Combine into 16D vector
    result = np.concatenate([
        left_arm,
        right_arm,
        np.array([left_gripper], dtype=np.float32),
        np.array([right_gripper], dtype=np.float32),
    ])
    
    return result.astype(np.float32)


# =============================================================================
# SECTION 3: Feature Generators for modify_features
# =============================================================================

def create_observation_state_generator(dataset: LeRobotDataset):
    """
    Create a generator function for the new observation.state feature.
    
    This returns a callable that takes (row_dict, episode_idx, frame_idx)
    and returns the converted 16D observation state.
    """
    def generator(row_dict: dict, episode_idx: int, frame_idx: int) -> np.ndarray:
        # Get the original 28D observation.state
        state_28d = np.array(row_dict["observation.state"], dtype=np.float32)
        return convert_state_to_gripper(state_28d)
    
    return generator


def create_action_generator(dataset: LeRobotDataset):
    """
    Create a generator function for the new action feature.
    
    This returns a callable that takes (row_dict, episode_idx, frame_idx)
    and returns the converted 16D action.
    """
    def generator(row_dict: dict, episode_idx: int, frame_idx: int) -> np.ndarray:
        # Get the original 28D action
        action_28d = np.array(row_dict["action"], dtype=np.float32)
        return convert_state_to_gripper(action_28d)
    
    return generator


# =============================================================================
# SECTION 4: New Feature Definitions
# =============================================================================

# Feature names for different DOF configurations
DUAL_ARM_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
]

SINGLE_ARM_NAMES = [
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
]

DUAL_HAND_NAMES = [
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]

SINGLE_HAND_NAMES = [
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]


def get_feature_names_for_mode(mode: DofMode) -> list[str]:
    """Get the feature names for a given DOF mode."""
    if mode == DofMode.DUAL_GRIPPER:
        return DUAL_ARM_NAMES + ["kLeftGripper", "kRightGripper"]
    elif mode == DofMode.DUAL_FULL:
        return DUAL_ARM_NAMES + DUAL_HAND_NAMES
    elif mode == DofMode.SINGLE_GRIPPER:
        return SINGLE_ARM_NAMES + ["kRightGripper"]
    elif mode == DofMode.SINGLE_FULL:
        return SINGLE_ARM_NAMES + SINGLE_HAND_NAMES
    else:
        raise ValueError(f"Unknown DOF mode: {mode}")


def get_feature_info_for_mode(mode: DofMode) -> dict:
    """Get feature info dict for a given DOF mode."""
    names = get_feature_names_for_mode(mode)
    dim = DOF_MODE_INFO[mode]["output_dim"]
    return {
        "dtype": "float32",
        "shape": [dim],
        "names": [names],
    }


def convert_state_for_mode(state_28d: np.ndarray, mode: DofMode) -> np.ndarray:
    """
    Convert 28D state to the target DOF mode.
    
    Input format (28 values):
        [0:7]   - Left arm joints (7 DOF)
        [7:14]  - Right arm joints (7 DOF)
        [14:21] - Left hand joints (7 DOF)
        [21:28] - Right hand joints (7 DOF)
    
    Args:
        state_28d: 28-dimensional state array
        mode: Target DOF mode
        
    Returns:
        np.ndarray: Converted state array
    """
    left_arm = state_28d[0:7]
    right_arm = state_28d[7:14]
    left_hand = state_28d[14:21]
    right_hand = state_28d[21:28]
    
    if mode == DofMode.DUAL_GRIPPER:
        # 16D: 14 arm + 2 gripper
        left_gripper = estimate_hand_openness(left_hand, hand="left")
        right_gripper = estimate_hand_openness(right_hand, hand="right")
        result = np.concatenate([
            left_arm, right_arm,
            np.array([left_gripper, right_gripper], dtype=np.float32),
        ])
    elif mode == DofMode.DUAL_FULL:
        # 28D: no conversion needed
        result = state_28d.copy()
    elif mode == DofMode.SINGLE_GRIPPER:
        # 8D: 7 right arm + 1 right gripper
        right_gripper = estimate_hand_openness(right_hand, hand="right")
        result = np.concatenate([
            right_arm,
            np.array([right_gripper], dtype=np.float32),
        ])
    elif mode == DofMode.SINGLE_FULL:
        # 14D: 7 right arm + 7 right hand
        result = np.concatenate([right_arm, right_hand])
    else:
        raise ValueError(f"Unknown DOF mode: {mode}")
    
    return result.astype(np.float32)


# Keep legacy function for backward compatibility
def convert_state_to_gripper(state_28d: np.ndarray) -> np.ndarray:
    """Legacy function: Convert 28D to 16D dual gripper mode."""
    return convert_state_for_mode(state_28d, DofMode.DUAL_GRIPPER)


# =============================================================================
# SECTION 5: Main Conversion Logic
# =============================================================================

@dataclasses.dataclass
class ConversionConfig:
    """Configuration for the dataset DOF conversion."""
    repo_id: str
    """Repository ID of the source dataset (e.g., 'deepansh-methdai/three_camera')"""
    
    output_repo_id: str
    """Repository ID for the output dataset (required)"""
    
    dof_mode: Literal["dual_gripper", "dual_full", "single_gripper", "single_full"] = "dual_gripper"
    """DOF configuration mode:
    - dual_gripper: 28D -> 16D (14 arm + 2 gripper) [default]
    - dual_full: 28D -> 28D (no conversion)
    - single_gripper: 28D -> 8D (7 right arm + 1 gripper)
    - single_full: 28D -> 14D (7 right arm + 7 right hand)
    """
    
    root: str | None = None
    """Root path to the dataset. If None, uses HF_LEROBOT_HOME"""
    
    output_dir: str | None = None
    """Output directory for the new dataset. If None, uses default location"""
    
    remove_cameras: str | None = None
    """Comma-separated list of camera keys to remove (e.g., 'observation.images.cam_left_high,observation.images.cam_left_wrist')"""
    
    push_to_hub: bool = False
    """Whether to push the modified dataset to Hugging Face Hub"""
    
    dry_run: bool = False
    """If True, only print what would be done without actually modifying"""


def convert_hands_to_gripper(cfg: ConversionConfig) -> LeRobotDataset | None:
    """
    Convert a Unitree G1 dataset from 28D hand joints to 16D gripper states.
    
    Args:
        cfg: Conversion configuration
        
    Returns:
        The modified LeRobotDataset, or None if dry_run is True
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Parse DOF mode
    dof_mode = DofMode(cfg.dof_mode)
    mode_info = DOF_MODE_INFO[dof_mode]
    
    # Parse cameras to remove
    cameras_to_remove: list[str] = []
    if cfg.remove_cameras:
        cameras_to_remove = [c.strip() for c in cfg.remove_cameras.split(",") if c.strip()]
    
    # Setup paths
    root = Path(cfg.root) if cfg.root else None
    output_repo_id = cfg.output_repo_id
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    
    logging.info(f"Loading dataset: {cfg.repo_id}")
    logging.info(f"Root path: {root}")
    logging.info(f"Output repo_id: {output_repo_id}")
    logging.info(f"DOF mode: {dof_mode.value} ({mode_info['description']})")
    logging.info(f"Conversion: [{mode_info['input_dim']}] -> [{mode_info['output_dim']}]")
    if cameras_to_remove:
        logging.info(f"Cameras to remove: {cameras_to_remove}")
    
    # Load the source dataset
    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=root,
    )
    
    # Verify dataset structure
    features = dataset.meta.features
    logging.info(f"Dataset has {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    logging.info(f"Available cameras: {dataset.meta.camera_keys}")
    
    # Validate cameras to remove
    for cam in cameras_to_remove:
        if cam not in features:
            raise ValueError(f"Camera '{cam}' not found in dataset. Available: {list(features.keys())}")
    
    # Check observation.state shape
    if "observation.state" not in features:
        raise ValueError("Dataset does not have 'observation.state' feature")
    
    obs_state_shape = features["observation.state"]["shape"]
    logging.info(f"observation.state shape: {obs_state_shape}")
    
    if obs_state_shape[0] != mode_info["input_dim"]:
        raise ValueError(f"Expected observation.state shape [{mode_info['input_dim']}], got {obs_state_shape}")
    
    # Check action shape
    if "action" not in features:
        raise ValueError("Dataset does not have 'action' feature")
    
    action_shape = features["action"]["shape"]
    logging.info(f"action shape: {action_shape}")
    
    if action_shape[0] != mode_info["input_dim"]:
        raise ValueError(f"Expected action shape [{mode_info['input_dim']}], got {action_shape}")
    
    if cfg.dry_run:
        logging.info("=== DRY RUN MODE ===")
        logging.info(f"Would convert with mode: {dof_mode.value}")
        logging.info(f"  - observation.state: [{mode_info['input_dim']}] -> [{mode_info['output_dim']}]")
        logging.info(f"  - action: [{mode_info['input_dim']}] -> [{mode_info['output_dim']}]")
        if cameras_to_remove:
            logging.info(f"  - Would remove cameras: {cameras_to_remove}")
        logging.info(f"Would save to: {output_repo_id}")
        if cfg.push_to_hub:
            logging.info("Would push to Hugging Face Hub")
        return None
    
    logging.info("Converting dataset features using direct parquet modification...")
    
    # Use a direct approach: copy and transform in a single pass
    final_dataset = _convert_dataset_direct(
        dataset=dataset,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
        dof_mode=dof_mode,
        cameras_to_remove=cameras_to_remove,
    )
    
    logging.info(f"Successfully created new dataset: {output_repo_id}")
    logging.info(f"New observation.state shape: [{mode_info['output_dim']}]")
    logging.info(f"New action shape: [{mode_info['output_dim']}]")
    
    # Verify the conversion
    logging.info("Verifying conversion...")
    sample_item = final_dataset[0]
    obs_state = sample_item["observation.state"]
    action = sample_item["action"]
    logging.info(f"Sample observation.state shape: {obs_state.shape}")
    logging.info(f"Sample action shape: {action.shape}")
    logging.info(f"Sample observation.state values: {obs_state.numpy()}")
    logging.info(f"Sample action values: {action.numpy()}")
    
    if cfg.push_to_hub:
        logging.info("Pushing dataset to Hugging Face Hub...")
        final_dataset.push_to_hub(upload_large_folder=True)
        logging.info("Successfully pushed to Hub!")
    
    return final_dataset


def _convert_dataset_direct(
    dataset: LeRobotDataset,
    output_repo_id: str,
    output_dir: Path | None = None,
    dof_mode: DofMode = DofMode.DUAL_GRIPPER,
    cameras_to_remove: list[str] | None = None,
) -> LeRobotDataset:
    """
    Convert dataset by directly modifying parquet files in a single pass.
    
    This is more efficient than using modify_features twice, as it avoids
    creating an intermediate dataset.
    
    Args:
        dataset: Source dataset
        output_repo_id: Output repository ID
        output_dir: Output directory (optional)
        dof_mode: DOF configuration mode
        cameras_to_remove: List of camera keys to exclude from output
    """
    import shutil
    import pandas as pd
    from tqdm import tqdm
    
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import (
        DATA_DIR,
        DEFAULT_DATA_PATH,
        write_info,
        write_stats,
        write_tasks,
    )
    from lerobot.utils.constants import HF_LEROBOT_HOME
    
    cameras_to_remove = cameras_to_remove or []
    
    # Setup output directory
    if output_dir is None:
        output_dir = HF_LEROBOT_HOME / output_repo_id
    else:
        output_dir = Path(output_dir)
    
    # Create new features dict with updated shapes, excluding cameras to remove
    new_features = {}
    for key, info in dataset.meta.features.items():
        if key in cameras_to_remove:
            logging.info(f"Excluding camera feature: {key}")
            continue
        if key == "observation.state":
            new_features[key] = get_feature_info_for_mode(dof_mode)
        elif key == "action":
            new_features[key] = get_feature_info_for_mode(dof_mode)
        else:
            new_features[key] = info.copy()
    
    # Determine remaining video keys
    remaining_video_keys = [k for k in dataset.meta.video_keys if k not in cameras_to_remove]
    
    # Create new metadata
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=dataset.meta.fps,
        features=new_features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(remaining_video_keys) > 0,
    )
    
    # Copy and transform data files
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    logging.info(f"Processing {len(parquet_files)} parquet files...")
    
    for src_path in tqdm(parquet_files, desc="Converting data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)
        
        # Remove camera columns from DataFrame
        for cam in cameras_to_remove:
            if cam in df.columns:
                df = df.drop(columns=[cam])
        
        # Get relative path info
        relative_path = src_path.relative_to(dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]
        
        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])
        
        # Transform observation.state and action columns using the specified mode
        new_obs_states = []
        new_actions = []
        
        for _, row in df.iterrows():
            obs_state_28d = np.array(row["observation.state"], dtype=np.float32)
            action_28d = np.array(row["action"], dtype=np.float32)
            
            new_obs_states.append(convert_state_for_mode(obs_state_28d, dof_mode))
            new_actions.append(convert_state_for_mode(action_28d, dof_mode))
        
        # Replace columns
        df["observation.state"] = new_obs_states
        df["action"] = new_actions
        
        # Write to destination
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        _write_parquet_direct(df, dst_path, new_meta)
    
    # Copy videos (excluding removed cameras)
    logging.info("Copying video files...")
    _copy_videos_direct(dataset, new_meta, exclude_keys=cameras_to_remove)
    
    # Copy episodes metadata and tasks (filtering out removed camera columns)
    logging.info("Copying metadata...")
    _copy_metadata_direct(dataset, new_meta, cameras_to_remove=cameras_to_remove)
    
    # Load and return the new dataset
    return LeRobotDataset(
        repo_id=output_repo_id,
        root=output_dir,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )


def _write_parquet_direct(df: pd.DataFrame, path: Path, meta) -> None:
    """Write DataFrame to parquet with proper schema."""
    import datasets
    import pyarrow.parquet as pq
    
    from lerobot.datasets.utils import embed_images, get_hf_features_from_features
    
    hf_features = get_hf_features_from_features(meta.features)
    ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")
    
    if len(meta.image_keys) > 0:
        ep_dataset = embed_images(ep_dataset)
    
    table = ep_dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _copy_videos_direct(src_dataset: LeRobotDataset, dst_meta, exclude_keys: list[str] | None = None) -> None:
    """Copy all video files from source to destination, excluding specified keys."""
    import shutil
    from tqdm import tqdm
    
    exclude_keys = exclude_keys or []
    
    for video_key in src_dataset.meta.video_keys:
        if video_key in exclude_keys:
            logging.info(f"Skipping excluded video key: {video_key}")
            continue
            
        video_files = set()
        for ep_idx in range(len(src_dataset.meta.episodes)):
            try:
                video_files.add(src_dataset.meta.get_video_file_path(ep_idx, video_key))
            except KeyError:
                continue
        
        for src_path in tqdm(sorted(video_files), desc=f"Copying {video_key} videos"):
            dst_path = dst_meta.root / src_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_dataset.root / src_path, dst_path)


def _copy_and_filter_episodes_metadata(src_dir: Path, dst_dir: Path, cameras_to_remove: list[str]) -> None:
    """Copy episodes metadata parquet files, filtering out columns for removed cameras."""
    import pandas as pd
    import pyarrow.parquet as pq
    from tqdm import tqdm
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_files = sorted(src_dir.glob("**/*.parquet"))
    
    for src_path in tqdm(parquet_files, desc="Filtering episodes metadata"):
        df = pd.read_parquet(src_path)
        
        # Build list of columns to drop (any column containing a removed camera key)
        columns_to_drop = []
        for col in df.columns:
            for cam in cameras_to_remove:
                if cam in col:
                    columns_to_drop.append(col)
                    break
        
        if columns_to_drop:
            logging.info(f"Dropping {len(columns_to_drop)} columns from episodes metadata: {columns_to_drop[:5]}...")
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Write to destination
        dst_path = dst_dir / src_path.relative_to(src_dir)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_path, index=False)


def _copy_metadata_direct(src_dataset: LeRobotDataset, dst_meta, cameras_to_remove: list[str] | None = None) -> None:
    """Copy episodes metadata, tasks, and calculate stats, filtering out removed cameras."""
    import shutil
    
    from lerobot.datasets.utils import write_info, write_stats, write_tasks
    
    cameras_to_remove = cameras_to_remove or []
    
    # Copy tasks
    if src_dataset.meta.tasks is not None:
        write_tasks(src_dataset.meta.tasks, dst_meta.root)
        dst_meta.tasks = src_dataset.meta.tasks.copy()
    
    # Copy episodes metadata, filtering out removed camera columns
    episodes_dir = src_dataset.root / "meta/episodes"
    dst_episodes_dir = dst_meta.root / "meta/episodes"
    if episodes_dir.exists():
        # Instead of simple copy, we need to filter parquet files
        _copy_and_filter_episodes_metadata(episodes_dir, dst_episodes_dir, cameras_to_remove)
    
    # Update info
    dst_meta.info.update({
        "total_episodes": src_dataset.meta.total_episodes,
        "total_frames": src_dataset.meta.total_frames,
        "total_tasks": src_dataset.meta.total_tasks,
        "splits": src_dataset.meta.info.get("splits", {"train": f"0:{src_dataset.meta.total_episodes}"}),
    })
    
    # Preserve video info
    if dst_meta.video_keys and src_dataset.meta.video_keys:
        for key in dst_meta.video_keys:
            if key in src_dataset.meta.features:
                dst_meta.info["features"][key]["info"] = src_dataset.meta.info["features"][key].get("info", {})
    
    write_info(dst_meta.info, dst_meta.root)
    
    # Calculate new stats for observation.state and action
    logging.info("Recalculating statistics for modified features...")
    _calculate_and_write_stats(src_dataset, dst_meta)


def _calculate_and_write_stats(src_dataset: LeRobotDataset, dst_meta) -> None:
    """Calculate stats for the new features including quantile statistics."""
    from lerobot.datasets.utils import write_stats
    
    # Copy stats from source for unchanged features
    new_stats = {}
    if src_dataset.meta.stats:
        for key in dst_meta.features:
            if key not in ["observation.state", "action"] and key in src_dataset.meta.stats:
                new_stats[key] = src_dataset.meta.stats[key]
    
    # For observation.state and action, we need to compute new stats
    # We'll compute them from the converted data
    logging.info("Computing statistics for observation.state and action...")
    
    import pandas as pd
    from pathlib import Path
    
    from lerobot.datasets.utils import DATA_DIR
    
    data_dir = dst_meta.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    
    all_obs_states = []
    all_actions = []
    
    for src_path in parquet_files:
        df = pd.read_parquet(src_path)
        all_obs_states.extend(df["observation.state"].tolist())
        all_actions.extend(df["action"].tolist())
    
    obs_states_array = np.array(all_obs_states, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    
    # Compute stats including quantiles
    # Quantile keys: q01, q10, q50, q90, q99
    quantiles = [0.01, 0.10, 0.50, 0.90, 0.99]
    quantile_keys = ["q01", "q10", "q50", "q90", "q99"]
    
    # Compute observation.state stats
    obs_quantiles = np.quantile(obs_states_array, quantiles, axis=0)
    new_stats["observation.state"] = {
        "min": obs_states_array.min(axis=0).tolist(),
        "max": obs_states_array.max(axis=0).tolist(),
        "mean": obs_states_array.mean(axis=0).tolist(),
        "std": obs_states_array.std(axis=0).tolist(),
        "count": [len(obs_states_array)],
    }
    for i, qkey in enumerate(quantile_keys):
        new_stats["observation.state"][qkey] = obs_quantiles[i].tolist()
    
    # Compute action stats
    action_quantiles = np.quantile(actions_array, quantiles, axis=0)
    new_stats["action"] = {
        "min": actions_array.min(axis=0).tolist(),
        "max": actions_array.max(axis=0).tolist(),
        "mean": actions_array.mean(axis=0).tolist(),
        "std": actions_array.std(axis=0).tolist(),
        "count": [len(actions_array)],
    }
    for i, qkey in enumerate(quantile_keys):
        new_stats["action"][qkey] = action_quantiles[i].tolist()
    
    write_stats(new_stats, dst_meta.root)


# =============================================================================
# SECTION 6: CLI Entry Point
# =============================================================================

def main():
    """Main entry point with tyro CLI parsing."""
    cfg = tyro.cli(ConversionConfig, prog="convert_hands_to_gripper")
    convert_hands_to_gripper(cfg)


if __name__ == "__main__":
    main()
