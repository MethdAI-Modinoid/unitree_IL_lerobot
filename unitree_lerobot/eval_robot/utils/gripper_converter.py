"""
Gripper-to-Hand State Converter
===============================

This module provides a middleware layer that automatically converts between:
- Gripper-based representation: 1-2 values (gripper) where 0.0=open, 1.0=closed
- Full hand representation: 7-14 values (7 hand joints per arm)

Supports both single-arm and dual-arm configurations:

Dual Arm (default):
- 28D state/action: 14 arm joints + 14 hand joints (direct passthrough)
- 16D state/action: 14 arm joints + 2 gripper states (converts to 14 hand joints)

Single Arm:
- 14D state/action: 7 arm joints + 7 hand joints (direct passthrough)
- 8D state/action: 7 arm joints + 1 gripper state (converts to 7 hand joints)

Usage:
    converter = GripperConverter(ee_type="dex3")
    
    # For actions from policy
    full_action = converter.expand_action(action_np, arm_dof=14)
    
    # For single arm
    full_action = converter.expand_action(action_np, arm_dof=7)
    
    # Get EE actions (automatically handles single/dual arm)
    left_ee, right_ee = converter.get_ee_actions(action_np, arm_dof=14)
    ee_action = converter.get_ee_actions(action_np, arm_dof=7)  # Single arm returns (ee,) tuple
"""

import numpy as np
from typing import Literal, Tuple, Union

import logging_mp

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


# =============================================================================
# Hand Pose Constants (from sample_gripper_control.py)
# =============================================================================

DEX3_NUM_MOTORS = 7

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
# Core Conversion Functions
# =============================================================================

def interpolate_hand_pose(gripper_value: float, hand: str = "left") -> np.ndarray:
    """
    Convert a gripper value (0.0-1.0) to full 7-DOF hand joint positions.
    
    Args:
        gripper_value: Float from 0.0 (fully open) to 1.0 (fully closed)
        hand: "left" or "right"
        
    Returns:
        np.ndarray: 7-dimensional array of joint angles in radians
    """
    gripper_value = np.clip(gripper_value, 0.0, 1.0)
    
    open_pose = HAND_POSES[hand]["open"]
    close_pose = HAND_POSES[hand]["close"]
    
    # Linear interpolation between open and closed
    return open_pose + (close_pose - open_pose) * gripper_value


def estimate_hand_openness(joint_positions: np.ndarray, hand: str = "left") -> float:
    """
    Estimate gripper value (0.0-1.0) from 7-DOF hand joint positions.
    
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
        if abs(delta) < 1e-6:
            continue
        t_i = (joint_positions[i] - open_pose[i]) / delta
        t_i = np.clip(t_i, 0.0, 1.0)
        t_values.append(t_i)
    
    if len(t_values) == 0:
        return 0.0
    
    return float(np.mean(t_values))


# =============================================================================
# Gripper Converter Class
# =============================================================================

class GripperConverter:
    """
    Middleware for converting between gripper and full hand representations.
    
    Automatically detects the representation based on action/state dimensions
    and converts as needed for robot execution.
    
    Supports:
    - Dual arm (arm_dof=14): 28D full or 16D gripper
    - Single arm (arm_dof=7): 14D full or 8D gripper
    """
    
    # Expected dimensions for different end-effector types
    EE_FULL_DOF = {
        "dex3": 7,      # 7 joints per hand
        "dex1": 1,      # 1 joint per gripper
        "inspire1": 6,  # 6 joints per hand
        "brainco": 6,   # 6 joints per hand
    }
    
    def __init__(self, ee_type: str = "dex3"):
        """
        Initialize the converter.
        
        Args:
            ee_type: End-effector type ("dex3", "dex1", "inspire1", "brainco")
        """
        self.ee_type = ee_type.lower()
        self.full_ee_dof = self.EE_FULL_DOF.get(self.ee_type, 7)
        self._is_gripper_mode = None  # Will be auto-detected
        self._is_single_arm = None  # Will be auto-detected
        
    def detect_mode(self, total_dim: int, arm_dof: int = 14) -> tuple:
        """
        Detect if the data uses gripper mode and single/dual arm based on dimensions.
        
        Args:
            total_dim: Total dimension of state/action vector
            arm_dof: Number of arm DOFs (14 for dual arm, 7 for single arm)
            
        Returns:
            tuple: (is_gripper_mode: bool, is_single_arm: bool)
        """
        # Dual arm expectations
        dual_full = 14 + 2 * self.full_ee_dof      # e.g., 14 + 14 = 28
        dual_gripper = 14 + 2                       # e.g., 14 + 2 = 16
        
        # Single arm expectations
        single_full = 7 + self.full_ee_dof         # e.g., 7 + 7 = 14
        single_gripper = 7 + 1                      # e.g., 7 + 1 = 8
        
        if total_dim == dual_gripper:
            return (True, False)   # Gripper mode, dual arm
        elif total_dim == dual_full:
            return (False, False)  # Full mode, dual arm
        elif total_dim == single_gripper:
            return (True, True)    # Gripper mode, single arm
        elif total_dim == single_full:
            return (False, True)   # Full mode, single arm
        else:
            logger_mp.warning(
                f"Unexpected dimension {total_dim}. Expected: "
                f"dual_full={dual_full}, dual_gripper={dual_gripper}, "
                f"single_full={single_full}, single_gripper={single_gripper}. "
                f"Assuming full mode with arm_dof={arm_dof}."
            )
            return (False, arm_dof == 7)
    
    def is_gripper_mode(self, action_or_state: np.ndarray, arm_dof: int = 14) -> bool:
        """Check if the given action/state is in gripper mode."""
        is_gripper, _ = self.detect_mode(len(action_or_state), arm_dof)
        return is_gripper
    
    def is_single_arm(self, action_or_state: np.ndarray, arm_dof: int = 14) -> bool:
        """Check if the given action/state is for single arm."""
        _, single_arm = self.detect_mode(len(action_or_state), arm_dof)
        return single_arm
    
    def expand_action(
        self, 
        action: np.ndarray, 
        arm_dof: int = 14,
        force_mode: Literal["auto", "gripper", "full"] = "auto"
    ) -> np.ndarray:
        """
        Expand action from gripper representation to full hand representation.
        
        If action is already in full representation, returns as-is.
        If action is in gripper representation, expands gripper values to full hand joints.
        
        Args:
            action: Action vector (gripper or full mode)
            arm_dof: Number of arm DOFs (14 for dual, 7 for single)
            force_mode: "auto" to detect, "gripper" to force expansion, "full" to skip
            
        Returns:
            np.ndarray: Action with full hand representation
        """
        if force_mode == "full":
            return action
        
        is_gripper, is_single = self.detect_mode(len(action), arm_dof)
        
        if force_mode == "gripper":
            is_gripper = True
        elif force_mode == "auto" and not is_gripper:
            return action
        
        if not is_gripper:
            return action
        
        # Extract components from gripper action
        if is_single:
            # Single arm: [arm(7), gripper(1)]
            # Single arm uses RIGHT arm/hand
            actual_arm_dof = 7
            arm_action = action[:actual_arm_dof]
            gripper_value = action[actual_arm_dof]
            
            # Convert gripper to full hand (single arm uses RIGHT hand)
            hand_joints = interpolate_hand_pose(gripper_value, hand="right")
            
            full_action = np.concatenate([arm_action, hand_joints])
        else:
            # Dual arm: [arm(14), left_gripper(1), right_gripper(1)]
            actual_arm_dof = 14
            arm_action = action[:actual_arm_dof]
            left_gripper = action[actual_arm_dof]
            right_gripper = action[actual_arm_dof + 1]
            
            # Convert gripper values to full hand joint positions
            left_hand_joints = interpolate_hand_pose(left_gripper, hand="left")
            right_hand_joints = interpolate_hand_pose(right_gripper, hand="right")
            
            full_action = np.concatenate([arm_action, left_hand_joints, right_hand_joints])
        
        return full_action.astype(np.float32)
    
    def compact_state(
        self,
        state: np.ndarray,
        arm_dof: int = 14,
        force_mode: Literal["auto", "gripper", "full"] = "auto"
    ) -> np.ndarray:
        """
        Compact state from full hand representation to gripper representation.
        
        Useful for sending observations to a policy trained on gripper data.
        
        Args:
            state: State vector (full mode)
            arm_dof: Number of arm DOFs
            force_mode: "auto" to detect, "gripper" to force compaction, "full" to skip
            
        Returns:
            np.ndarray: State with gripper representation if applicable
        """
        is_gripper, is_single = self.detect_mode(len(state), arm_dof)
        
        if force_mode == "gripper":
            if is_gripper:
                return state  # Already compact
        elif force_mode == "full" or (force_mode == "auto" and is_gripper):
            return state  # Already compact or should stay full
        
        if is_single:
            # Single arm full -> gripper
            # Single arm uses RIGHT arm/hand
            actual_arm_dof = 7
            arm_state = state[:actual_arm_dof]
            hand_joints = state[actual_arm_dof:actual_arm_dof + self.full_ee_dof]
            
            gripper_value = estimate_hand_openness(hand_joints, hand="right")
            
            compact = np.concatenate([arm_state, np.array([gripper_value], dtype=np.float32)])
        else:
            # Dual arm full -> gripper
            actual_arm_dof = 14
            arm_state = state[:actual_arm_dof]
            left_hand = state[actual_arm_dof:actual_arm_dof + self.full_ee_dof]
            right_hand = state[actual_arm_dof + self.full_ee_dof:actual_arm_dof + 2 * self.full_ee_dof]
            
            left_gripper = estimate_hand_openness(left_hand, hand="left")
            right_gripper = estimate_hand_openness(right_hand, hand="right")
            
            compact = np.concatenate([
                arm_state,
                np.array([left_gripper, right_gripper], dtype=np.float32)
            ])
        
        return compact.astype(np.float32)
    
    def get_ee_actions(
        self,
        action: np.ndarray,
        arm_dof: int = 14
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray]]:
        """
        Extract end-effector actions, expanding if needed.
        
        Always returns full hand joint arrays suitable for robot execution.
        
        Args:
            action: Action vector (gripper or full mode)
            arm_dof: Number of arm DOFs (14 for dual, 7 for single)
            
        Returns:
            For dual arm: tuple (left_ee_action, right_ee_action) each with full DOF
            For single arm: tuple (ee_action,) with full DOF
        """
        full_action = self.expand_action(action, arm_dof)
        
        _, is_single = self.detect_mode(len(action), arm_dof)
        
        if is_single:
            actual_arm_dof = 7
            ee_action = full_action[actual_arm_dof:actual_arm_dof + self.full_ee_dof]
            return (ee_action,)
        else:
            actual_arm_dof = 14
            left_ee = full_action[actual_arm_dof:actual_arm_dof + self.full_ee_dof]
            right_ee = full_action[actual_arm_dof + self.full_ee_dof:actual_arm_dof + 2 * self.full_ee_dof]
            return (left_ee, right_ee)


def create_converter(ee_type: str = "dex3") -> GripperConverter:
    """Factory function to create a gripper converter."""
    return GripperConverter(ee_type=ee_type)


# =============================================================================
# Utility Functions for Dataset Detection
# =============================================================================

def detect_dataset_mode(dataset_meta) -> dict:
    """
    Detect dataset configuration from metadata.
    
    Args:
        dataset_meta: LeRobotDataset metadata object
        
    Returns:
        dict with keys:
            - "mode": "gripper" or "full"
            - "arm_type": "single" or "dual"
            - "state_dim": dimension of observation.state
            - "action_dim": dimension of action
    """
    result = {
        "mode": "full",
        "arm_type": "dual",
        "state_dim": 28,
        "action_dim": 28,
    }
    
    if "observation.state" in dataset_meta.features:
        state_shape = dataset_meta.features["observation.state"]["shape"]
        state_dim = state_shape[0] if isinstance(state_shape, (list, tuple)) else state_shape
        result["state_dim"] = state_dim
        
        # Detect based on dimension
        if state_dim == 16:
            result["mode"] = "gripper"
            result["arm_type"] = "dual"
        elif state_dim == 28:
            result["mode"] = "full"
            result["arm_type"] = "dual"
        elif state_dim == 8:
            result["mode"] = "gripper"
            result["arm_type"] = "single"
        elif state_dim == 14:
            result["mode"] = "full"
            result["arm_type"] = "single"
    
    if "action" in dataset_meta.features:
        action_shape = dataset_meta.features["action"]["shape"]
        action_dim = action_shape[0] if isinstance(action_shape, (list, tuple)) else action_shape
        result["action_dim"] = action_dim
    
    return result
