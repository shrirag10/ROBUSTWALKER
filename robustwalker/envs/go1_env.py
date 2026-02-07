"""
Go1 Gymnasium Environment for Blind Locomotion Training.

This environment trains a quadruped robot to walk on rough terrain
using only proprioceptive sensing (no vision/lidar).
"""

import os
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from robustwalker.envs.domain_rand import DomainRandomizer, DomainRandomizationConfig
from robustwalker.rewards.locomotion import LocomotionReward, RewardConfig
from robustwalker.utils.mujoco_utils import (
    get_joint_positions,
    get_joint_velocities,
    get_base_velocity,
    get_projected_gravity,
    get_body_rotation_matrix,
    get_foot_contact_forces,
)


# Joint order in Go1 model
JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Default standing pose
DEFAULT_JOINT_ANGLES = np.array([
    0.0, 0.9, -1.8,  # FR
    0.0, 0.9, -1.8,  # FL
    0.0, 0.9, -1.8,  # RR
    0.0, 0.9, -1.8,  # RL
])


class Go1Env(gym.Env):
    """
    Gymnasium environment for Unitree Go1 blind locomotion.
    
    Observation Space (57 dim):
        - Joint positions (12)
        - Joint velocities (12) 
        - Base angular velocity (3)
        - Projected gravity (3)
        - Velocity commands (3)
        - Action history (12 * 2 = 24)
    
    Action Space (12 dim):
        - Joint position targets (position mode)
        - or Joint torques (torque mode)
        - or Position deltas (delta mode)
        
    Control Modes:
        - 'position': PD control to target joint angles
        - 'torque': Direct torque control
        - 'delta': Position delta from current
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        render_mode: str | None = None,
        control_mode: str = "position",
        control_freq: float = 50.0,
        episode_length: int = 1000,
        enable_domain_rand: bool = True,
        reward_config: RewardConfig | None = None,
        domain_rand_config: DomainRandomizationConfig | None = None,
        command_curriculum: bool = True,
    ):
        """
        Initialize Go1 environment.
        
        Args:
            render_mode: 'human' for viewer, 'rgb_array' for image, None for headless
            control_mode: One of 'position', 'torque', 'delta'
            control_freq: Control frequency in Hz
            episode_length: Maximum episode length in steps
            enable_domain_rand: Whether to enable domain randomization
            reward_config: Custom reward configuration
            domain_rand_config: Custom domain randomization configuration
            command_curriculum: Whether to use curriculum on velocity commands
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.control_mode = control_mode
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.episode_length = episode_length
        self.enable_domain_rand = enable_domain_rand
        self.command_curriculum = command_curriculum
        
        # Load MuJoCo model
        model_path = self._get_model_path()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        # Compute simulation steps per control step
        self.model.opt.timestep = 0.002  # 500 Hz physics
        self.sim_steps_per_control = int(self.dt / self.model.opt.timestep)
        
        # Get joint limits
        self.joint_limits_lower = self.model.jnt_range[1:, 0].copy()  # Skip freejoint
        self.joint_limits_upper = self.model.jnt_range[1:, 1].copy()
        
        # Initialize components
        self.reward_fn = LocomotionReward(reward_config)
        
        if enable_domain_rand:
            self.domain_rand = DomainRandomizer(domain_rand_config)
        else:
            self.domain_rand = None
            
        # Action history for observation
        self.action_history_len = 2
        self.action_history = np.zeros((self.action_history_len, 12))
        
        # Velocity commands
        self.commands = np.zeros(3)  # vx, vy, omega_z
        
        # Episode tracking
        self.step_count = 0
        self.sim_time = 0.0
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Rendering
        self.viewer = None
        self.renderer = None
        
    def _get_model_path(self) -> Path:
        """Get path to Go1 MuJoCo model with scene (ground, lighting)."""
        # Look in project assets directory
        project_root = Path(__file__).parent.parent.parent
        
        # Use scene.xml which includes go1.xml plus ground plane and lighting
        model_path = project_root / "assets" / "go1" / "scene.xml"
        
        if not model_path.exists():
            # Fallback to go1.xml if scene doesn't exist
            model_path = project_root / "assets" / "go1" / "go1.xml"
            
        if not model_path.exists():
            raise FileNotFoundError(
                f"Go1 model not found at {model_path}. "
                "Please ensure the model is downloaded to assets/go1/"
            )
        return model_path
    
    def _setup_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation space
        obs_dim = (
            12 +  # joint positions
            12 +  # joint velocities
            3 +   # base angular velocity
            3 +   # projected gravity
            3 +   # commands
            12 * self.action_history_len  # action history
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space depends on control mode
        if self.control_mode == "torque":
            # Torque limits from model
            action_low = self.model.actuator_forcerange[:, 0]
            action_high = self.model.actuator_forcerange[:, 1]
        else:
            # Position or delta mode: use joint limits
            action_low = np.full(12, -1.0)
            action_high = np.full(12, 1.0)
            
        self.action_space = spaces.Box(
            low=action_low.astype(np.float32),
            high=action_high.astype(np.float32),
            dtype=np.float32,
        )
        
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Optional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose from keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        
        # Add small random perturbation to initial joint positions
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.1, 0.1, 12)
            self.data.qpos[7:] += noise
            
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset action history
        self.action_history = np.zeros((self.action_history_len, 12))
        
        # Sample new velocity command
        self._sample_commands()
        
        # Apply domain randomization
        info = {}
        if self.domain_rand is not None:
            rand_info = self.domain_rand.reset(self.model, self.data, self.sim_time)
            info['domain_rand'] = rand_info
            
        # Reset reward function
        self.reward_fn.reset()
        
        # Reset counters
        self.step_count = 0
        self.sim_time = 0.0
        
        return self._get_obs(), info
    
    def step(
        self, 
        action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action array (12,)
            
        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action based on control mode
        if self.control_mode == "position":
            # Scale from [-1, 1] to joint limits
            target_pos = self._scale_action_to_joints(action)
            self.data.ctrl[:] = target_pos
            
        elif self.control_mode == "torque":
            self.data.ctrl[:] = action
            
        elif self.control_mode == "delta":
            # Apply delta to current position
            current_pos = get_joint_positions(self.model, self.data)
            delta_scale = 0.1  # Scale factor for deltas
            target_pos = current_pos + action * delta_scale
            target_pos = np.clip(target_pos, self.joint_limits_lower, self.joint_limits_upper)
            self.data.ctrl[:] = target_pos
            
        # Step simulation
        for _ in range(self.sim_steps_per_control):
            # Apply domain randomization perturbations
            if self.domain_rand is not None:
                self.domain_rand.step(self.model, self.data, self.sim_time)
                
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.model.opt.timestep
            
        # Update action history
        self.action_history = np.roll(self.action_history, -1, axis=0)
        self.action_history[-1] = action
        
        # Get observation
        obs = self._get_obs()
        
        # Check termination
        terminated = self._check_termination()
        
        # Compute reward
        reward, reward_info = self._compute_reward(action, terminated)
        
        # Check truncation
        self.step_count += 1
        truncated = self.step_count >= self.episode_length
        
        # Build info dict
        info = {
            'reward_components': reward_info,
            'base_velocity': get_base_velocity(self.model, self.data)[0],
            'commands': self.commands.copy(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Joint positions (normalized)
        joint_pos = get_joint_positions(self.model, self.data)
        joint_pos_normalized = self._normalize_joints(joint_pos)
        
        # Joint velocities (scaled)
        joint_vel = get_joint_velocities(self.model, self.data)
        joint_vel_scaled = joint_vel * 0.05  # Scale factor
        
        # Base angular velocity in body frame
        _, base_angvel_world = get_base_velocity(self.model, self.data)
        rot_mat = get_body_rotation_matrix(self.model, self.data, "trunk")
        base_angvel_body = rot_mat.T @ base_angvel_world
        base_angvel_scaled = base_angvel_body * 0.25
        
        # Projected gravity (already in body frame)
        projected_gravity = get_projected_gravity(self.model, self.data)
        
        # Flatten action history
        action_history_flat = self.action_history.flatten()
        
        # Concatenate all observations
        obs = np.concatenate([
            joint_pos_normalized,
            joint_vel_scaled,
            base_angvel_scaled,
            projected_gravity,
            self.commands,
            action_history_flat,
        ])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, action: np.ndarray, terminated: bool) -> tuple[float, dict]:
        """Compute reward for current step."""
        # Get required quantities
        base_linvel, base_angvel = get_base_velocity(self.model, self.data)
        rot_mat = get_body_rotation_matrix(self.model, self.data, "trunk")
        
        # Transform velocities to body frame
        base_linvel_body = rot_mat.T @ base_linvel
        base_angvel_body = rot_mat.T @ base_angvel
        
        joint_pos = get_joint_positions(self.model, self.data)
        joint_vel = get_joint_velocities(self.model, self.data)
        projected_gravity = get_projected_gravity(self.model, self.data)
        
        # Get joint torques
        joint_torques = self.data.actuator_force.copy()
        
        # Foot contacts
        foot_forces = get_foot_contact_forces(self.model, self.data)
        foot_contacts = {k: np.linalg.norm(v) > 1.0 for k, v in foot_forces.items()}
        
        # Body contact check (non-foot collision)
        body_contacts = self._check_body_contact()
        
        return self.reward_fn.compute(
            base_linear_vel=base_linvel_body,
            base_angular_vel=base_angvel_body,
            joint_torques=joint_torques,
            action=action,
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            joint_limits=(self.joint_limits_lower, self.joint_limits_upper),
            projected_gravity=projected_gravity,
            foot_contacts=foot_contacts,
            body_contacts=body_contacts,
            commands=self.commands,
            terminated=terminated,
        )
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate (robot fell)."""
        # Get base height
        base_height = self.data.qpos[2]
        
        # Get orientation (projected gravity z component)
        projected_gravity = get_projected_gravity(self.model, self.data)
        
        # Terminate if:
        # 1. Base too low (fell)
        # 2. Body tilted too much
        if base_height < 0.15:
            return True
        if projected_gravity[2] > -0.5:  # Should be close to -1 when upright
            return True
            
        return False
    
    def _check_body_contact(self) -> bool:
        """Check if body (non-foot) is in contact with ground."""
        foot_geoms = ["FR", "FL", "RR", "RL"]
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            # If neither geom is a foot and one is ground-related
            if geom1 not in foot_geoms and geom2 not in foot_geoms:
                # This is a body contact
                if geom1 is not None and geom2 is not None:
                    return True
                    
        return False
    
    def _scale_action_to_joints(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to joint position limits."""
        # Map [-1, 1] to [lower, upper]
        scaled = (action + 1) / 2  # [0, 1]
        return self.joint_limits_lower + scaled * (self.joint_limits_upper - self.joint_limits_lower)
    
    def _normalize_joints(self, joint_pos: np.ndarray) -> np.ndarray:
        """Normalize joint positions to [-1, 1]."""
        normalized = (joint_pos - self.joint_limits_lower) / (
            self.joint_limits_upper - self.joint_limits_lower
        )
        return 2 * normalized - 1
    
    def _sample_commands(self) -> None:
        """Sample new velocity commands."""
        if self.command_curriculum and self.np_random is not None:
            # Sample random commands
            vx = self.np_random.uniform(0.0, 1.0)  # Forward velocity
            vy = self.np_random.uniform(-0.3, 0.3)  # Lateral velocity
            omega = self.np_random.uniform(-0.5, 0.5)  # Yaw rate
            self.commands = np.array([vx, vy, omega])
        else:
            # Default command: walk forward
            self.commands = np.array([0.8, 0.0, 0.0])
            
    def set_commands(self, vx: float, vy: float, omega_z: float) -> None:
        """Manually set velocity commands."""
        self.commands = np.array([vx, vy, omega_z])
        
    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode is None:
            return None
            
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)
            
        self.renderer.update_scene(self.data)
        
        if self.render_mode == "rgb_array":
            return self.renderer.render()
        elif self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
            
    def close(self) -> None:
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer = None


# Register environment with Gymnasium
gym.register(
    id="Go1-v0",
    entry_point="robustwalker.envs.go1_env:Go1Env",
    max_episode_steps=1000,
)
