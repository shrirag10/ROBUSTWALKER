"""
Locomotion reward functions for Go1 training.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward function weights."""
    
    # Positive rewards
    velocity_tracking: float = 1.0
    alive_bonus: float = 0.1
    
    # Penalties (negative coefficients)
    torque_penalty: float = 0.001
    action_rate_penalty: float = 0.1
    joint_acc_penalty: float = 0.0001
    stumble_penalty: float = 2.0
    termination_penalty: float = 5.0
    orientation_penalty: float = 0.5
    joint_limit_penalty: float = 1.0
    
    # Velocity tracking parameters
    target_velocity: tuple[float, float, float] = (0.8, 0.0, 0.0)  # (vx, vy, omega_z)
    velocity_tracking_sigma: float = 0.25


class LocomotionReward:
    """
    Computes reward for Go1 locomotion task.
    
    Reward = velocity_tracking + alive_bonus 
           - torque_penalty - action_rate_penalty 
           - stumble_penalty - orientation_penalty
    """
    
    def __init__(self, config: RewardConfig | None = None):
        """
        Initialize reward function.
        
        Args:
            config: Reward weights configuration
        """
        self.config = config or RewardConfig()
        self._last_action = None
        self._last_joint_vel = None
        
    def reset(self) -> None:
        """Reset reward state at episode start."""
        self._last_action = None
        self._last_joint_vel = None
        
    def compute(
        self,
        base_linear_vel: np.ndarray,
        base_angular_vel: np.ndarray,
        joint_torques: np.ndarray,
        action: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_limits: tuple[np.ndarray, np.ndarray],
        projected_gravity: np.ndarray,
        foot_contacts: dict[str, bool],
        body_contacts: bool,
        commands: np.ndarray | None = None,
        terminated: bool = False,
    ) -> tuple[float, dict]:
        """
        Compute total reward for current step.
        
        Args:
            base_linear_vel: Base linear velocity (3,) in body frame
            base_angular_vel: Base angular velocity (3,) in body frame
            joint_torques: Applied joint torques (12,)
            action: Current action (12,)
            joint_positions: Current joint positions (12,)
            joint_velocities: Current joint velocities (12,)
            joint_limits: Tuple of (lower, upper) joint limits
            projected_gravity: Gravity vector in body frame (3,)
            foot_contacts: Dict mapping foot name to contact boolean
            body_contacts: Whether body (non-foot) is in contact
            commands: Optional velocity commands (vx, vy, omega_z)
            terminated: Whether episode terminated due to fall
            
        Returns:
            Tuple of (total_reward, info_dict with individual components)
        """
        cfg = self.config
        
        # Use commanded velocity or default target
        target_vel = np.array(commands) if commands is not None else np.array(cfg.target_velocity)
        
        # Individual reward components
        rewards = {}
        
        # 1. Velocity tracking reward
        vel_error_x = (base_linear_vel[0] - target_vel[0]) ** 2
        vel_error_y = (base_linear_vel[1] - target_vel[1]) ** 2
        vel_error_yaw = (base_angular_vel[2] - target_vel[2]) ** 2
        
        vel_error = vel_error_x + vel_error_y + 0.5 * vel_error_yaw
        rewards['velocity_tracking'] = cfg.velocity_tracking * np.exp(-vel_error / cfg.velocity_tracking_sigma)
        
        # 2. Alive bonus
        rewards['alive'] = cfg.alive_bonus
        
        # 3. Torque penalty (energy efficiency)
        torque_cost = np.sum(joint_torques ** 2)
        rewards['torque'] = -cfg.torque_penalty * torque_cost
        
        # 4. Action rate penalty (smoothness)
        if self._last_action is not None:
            action_rate = np.sum((action - self._last_action) ** 2)
            rewards['action_rate'] = -cfg.action_rate_penalty * action_rate
        else:
            rewards['action_rate'] = 0.0
        
        # 5. Joint acceleration penalty
        if self._last_joint_vel is not None:
            joint_acc = np.sum((joint_velocities - self._last_joint_vel) ** 2)
            rewards['joint_acc'] = -cfg.joint_acc_penalty * joint_acc
        else:
            rewards['joint_acc'] = 0.0
            
        # 6. Stumble penalty (foot-body collision or unexpected contact)
        rewards['stumble'] = -cfg.stumble_penalty if body_contacts else 0.0
        
        # 7. Orientation penalty (keep upright)
        # Penalize deviation from upright (gravity should point down in body frame)
        orientation_error = projected_gravity[0] ** 2 + projected_gravity[1] ** 2
        rewards['orientation'] = -cfg.orientation_penalty * orientation_error
        
        # 8. Joint limit penalty
        lower_limits, upper_limits = joint_limits
        lower_violation = np.sum(np.maximum(lower_limits - joint_positions, 0) ** 2)
        upper_violation = np.sum(np.maximum(joint_positions - upper_limits, 0) ** 2)
        rewards['joint_limit'] = -cfg.joint_limit_penalty * (lower_violation + upper_violation)
        
        # 9. Termination penalty
        rewards['termination'] = -cfg.termination_penalty if terminated else 0.0
        
        # Update state
        self._last_action = action.copy()
        self._last_joint_vel = joint_velocities.copy()
        
        # Total reward
        total_reward = sum(rewards.values())
        
        return total_reward, rewards
    
    def set_command(self, vx: float, vy: float, omega_z: float) -> None:
        """
        Set velocity command.
        
        Args:
            vx: Forward velocity command (m/s)
            vy: Lateral velocity command (m/s)
            omega_z: Yaw rate command (rad/s)
        """
        self.config.target_velocity = (vx, vy, omega_z)


def compute_gait_reward(
    foot_contacts: dict[str, bool],
    phase: float,
    gait_type: str = "trot"
) -> float:
    """
    Compute reward for maintaining proper gait pattern.
    
    Args:
        foot_contacts: Dict mapping foot name to contact boolean
        phase: Current gait phase [0, 1)
        gait_type: One of 'trot', 'walk', 'gallop'
        
    Returns:
        Gait reward value
    """
    # Define expected contact patterns for trot gait
    # Diagonal pairs move together
    if gait_type == "trot":
        # Phase 0-0.5: FR+RL stance, FL+RR swing
        # Phase 0.5-1: FL+RR stance, FR+RL swing
        if phase < 0.5:
            expected = {'FR': True, 'RL': True, 'FL': False, 'RR': False}
        else:
            expected = {'FR': False, 'RL': False, 'FL': True, 'RR': True}
    else:
        # Default: just reward having some feet in contact
        return float(sum(foot_contacts.values()) >= 2)
    
    # Compute match score
    matches = sum(1 for k, v in foot_contacts.items() if v == expected[k])
    return matches / 4.0
