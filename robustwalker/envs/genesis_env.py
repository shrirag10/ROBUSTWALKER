"""
Go1 Genesis Environment for GPU-Accelerated Locomotion Training.

This environment uses the Genesis physics simulator for massively-parallel
training of blind locomotion policies on the Unitree Go1 quadruped robot.

Inspired by Genesis's official Go2 locomotion example and adapted for Go1.
Compatible with rsl-rl OnPolicyRunner for PPO training.
"""

import math
from pathlib import Path

import numpy as np
import torch
import genesis as gs

from robustwalker.utils.genesis_utils import (
    gs_rand,
    quat_rotate,
    inv_quat,
    quat_to_euler,
)

# Go1 joint names in robot order
GO1_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Default standing joint angles for Go1 (from MJCF keyframe "home")
GO1_DEFAULT_JOINT_ANGLES = {
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.9,
    "FR_calf_joint": -1.8,
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.9,
    "FL_calf_joint": -1.8,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.9,
    "RR_calf_joint": -1.8,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.9,
    "RL_calf_joint": -1.8,
}


class Go1GenesisEnv:
    """
    Genesis-based GPU-accelerated environment for Go1 locomotion.

    Uses Genesis's batched simulation for massively-parallel RL training.
    Compatible with rsl-rl's OnPolicyRunner.

    Key features:
        - GPU-parallel simulation (up to 30000+ environments)
        - Built-in PD controller for joint position control
        - Batched observation, reward, and reset operations
        - Action latency simulation for sim-to-real transfer
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        show_viewer: bool = False,
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        # Timing
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02  # 50 Hz control frequency (matches real robot)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # Store configs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ─── Create Genesis scene ────────────────────────────────────
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        # ─── Add ground plane ────────────────────────────────────────
        self.scene.add_entity(gs.morphs.Plane())

        # ─── Add Go1 robot ───────────────────────────────────────────
        model_path = self._get_model_path()
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(model_path),
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        # ─── Build scene (n_envs parallel) ───────────────────────────
        self.scene.build(n_envs=num_envs)

        # ─── Set neutral qpos to standing pose (fixes joint limits warning) ─
        # Genesis defaults to all-zero joint DOFs, but calf joints have range
        # [-2.818, -0.888] so 0.0 is outside limits. Set the standing pose.
        standing_qpos = np.zeros(self.robot.n_qs)
        # Base position and quaternion (first 7 values)
        standing_qpos[0:3] = self.env_cfg["base_init_pos"]
        standing_qpos[3:7] = self.env_cfg["base_init_quat"]
        # Set joint DOFs to standing angles
        for joint in self.robot.joints[1:]:  # skip freejoint
            if joint.name in self.env_cfg["default_joint_angles"]:
                standing_qpos[joint.dof_start] = self.env_cfg["default_joint_angles"][joint.name]
        self.robot.init_qpos = standing_qpos
        self.robot.set_qpos(self.robot.init_qpos)

        # ─── Map joint names to DOF indices ──────────────────────────
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start
             for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int, device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # ─── Map foot link indices for contact tracking ──────────────
        foot_link_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        self.foot_link_indices = []
        for name in foot_link_names:
            link = self.robot.get_link(name)
            self.foot_link_indices.append(link.idx - self.robot.link_start)
        self.num_feet = len(self.foot_link_indices)
        # Foot tip is ~0.213m below calf link center in Go1
        self.foot_tip_offset = 0.213
        self.feet_contact_threshold = self.env_cfg.get("feet_contact_threshold", 0.02)

        # ─── Configure PD controller ─────────────────────────────────
        self.robot.set_dofs_kp(
            [self.env_cfg["kp"]] * self.num_actions,
            self.motors_dof_idx,
        )
        self.robot.set_dofs_kv(
            [self.env_cfg["kd"]] * self.num_actions,
            self.motors_dof_idx,
        )

        # ─── Global constants ────────────────────────────────────────
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device,
        )

        # ─── Initial state ───────────────────────────────────────────
        self.init_base_pos = torch.tensor(
            self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device,
        )
        self.init_base_quat = torch.tensor(
            self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device,
        )
        self.inv_base_init_quat = inv_quat(
            self.init_base_quat.unsqueeze(0)
        ).squeeze(0)

        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name]
             for joint in self.robot.joints[1:]],  # skip freejoint
            dtype=gs.tc_float, device=gs.device,
        )
        self.init_qpos = torch.concatenate(
            (self.init_base_pos, self.init_base_quat, self.init_dof_pos)
        )

        # Initial projected gravity
        self.init_projected_gravity = quat_rotate(
            inv_quat(self.init_base_quat.unsqueeze(0)),
            self.global_gravity.unsqueeze(0),
        ).squeeze(0)

        # ─── Allocate state buffers ──────────────────────────────────
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.obs_buf = torch.empty((self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        # Commands: [vx, vy, omega_z]
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device, dtype=gs.tc_float,
        )
        self.commands_limits = [
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        ]

        # Action buffers
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)

        # DOF state buffers
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        # Base state
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)

        # Feet state for gait cycle tracking
        self.feet_pos = torch.empty((self.num_envs, self.num_feet, 3), dtype=gs.tc_float, device=gs.device)
        self.feet_air_time = torch.zeros((self.num_envs, self.num_feet), dtype=gs.tc_float, device=gs.device)
        self.last_feet_contact = torch.ones((self.num_envs, self.num_feet), dtype=gs.tc_bool, device=gs.device)
        self.first_contact = torch.zeros((self.num_envs, self.num_feet), dtype=gs.tc_bool, device=gs.device)
        self.landed_air_time = torch.zeros((self.num_envs, self.num_feet), dtype=gs.tc_float, device=gs.device)

        # Default DOF positions (for action offset)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float, device=gs.device,
        )

        # Extra info for logging
        self.extras = dict()
        self.extras["observations"] = dict()

        # ─── Register reward functions ───────────────────────────────
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        # ─── Domain Randomization ────────────────────────────────────
        self.domain_rand_cfg = self.env_cfg.get("domain_rand", {})
        self.randomize_kp = self.domain_rand_cfg.get("randomize_kp", False)
        self.push_robots = self.domain_rand_cfg.get("push_robots", False)
        self.obs_noise_enabled = self.domain_rand_cfg.get("obs_noise", False)
        self.action_noise_enabled = self.domain_rand_cfg.get("action_noise", False)

        # PD gain ranges
        self.kp_range = self.domain_rand_cfg.get("kp_range", [40.0, 60.0])
        self.kd_range = self.domain_rand_cfg.get("kd_range", [0.8, 1.2])

        # Push force config
        push_interval = self.domain_rand_cfg.get("push_interval_s", [5.0, 10.0])
        self.push_interval_range = [
            int(push_interval[0] / self.dt),
            int(push_interval[1] / self.dt),
        ]
        self.push_force_range = self.domain_rand_cfg.get("push_force_range", [0.0, 15.0])
        self.push_duration_steps = int(
            self.domain_rand_cfg.get("push_duration_s", 0.1) / self.dt
        )

        # Push timer buffer (counts down steps until next push)
        self.push_timer = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)

        # Noise scales
        self.obs_noise_level = self.domain_rand_cfg.get("obs_noise_level", 0.05)
        self.action_noise_scale = self.domain_rand_cfg.get("action_noise_scale", 0.02)

    # ─── Model path ──────────────────────────────────────────────────

    def _get_model_path(self) -> str:
        """Get absolute path to Go1 MJCF model."""
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "assets" / "go1" / "go1.xml"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Go1 model not found at {model_path}. "
                "Please ensure the model is downloaded to assets/go1/"
            )
        return str(model_path)

    # ─── Core API (compatible with rsl-rl) ───────────────────────────

    def reset(self):
        """Reset all environments. Returns (observations, None)."""
        self._reset_idx()
        self._update_observation()
        return self.obs_buf, None

    def step(self, actions):
        """
        Take one step in all environments.

        Returns:
            Tuple of (obs, rewards, dones, extras)
        """
        # Clip actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # Add action noise for domain randomization
        if self.action_noise_enabled:
            action_noise = torch.randn_like(self.actions) * self.action_noise_scale
            noisy_actions = self.actions + action_noise
        else:
            noisy_actions = self.actions

        # Simulate action latency (1-step delay as on real robot)
        exec_actions = self.last_actions if self.simulate_action_latency else noisy_actions

        # Convert to joint position targets: action * scale + default_pos
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # Send control commands (reorder to simulator DOF order)
        self.robot.control_dofs_position(
            target_dof_pos[:, self.actions_dof_idx],
            self.motors_dof_idx,
        )

        # ─── Apply push forces (domain randomization) ────────────────
        if self.push_robots:
            self._apply_push_forces()

        # Step physics
        self.scene.step()

        # ─── Update state buffers ────────────────────────────────────
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_euler(self.base_quat)

        # Transform velocities to body frame
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = quat_rotate(inv_base_quat, self.robot.get_vel())
        self.base_ang_vel = quat_rotate(inv_base_quat, self.robot.get_ang())
        self.projected_gravity = quat_rotate(
            inv_base_quat, self.global_gravity.expand(self.num_envs, -1),
        )

        # Joint state
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # Feet state — track foot z-positions for contact detection
        all_link_pos = self.robot.get_links_pos()
        for i, link_idx in enumerate(self.foot_link_indices):
            self.feet_pos[:, i, :] = all_link_pos[:, link_idx, :]
        feet_z = self.feet_pos[:, :, 2]  # (num_envs, 4)
        # Estimate foot tip z-position (calf link center is ~0.213m above foot)
        feet_tip_z = feet_z - self.foot_tip_offset
        self.feet_contact = feet_tip_z < self.feet_contact_threshold

        # Track feet air time (for gait cycle reward)
        self.first_contact = self.feet_contact & ~self.last_feet_contact  # just landed
        self.feet_air_time += self.dt
        # Save air time at landing BEFORE zeroing — reward function needs this
        self.landed_air_time = self.feet_air_time * self.first_contact
        self.feet_air_time *= ~self.feet_contact  # reset when in contact
        self.last_feet_contact = self.feet_contact.clone()

        # ─── Compute rewards ─────────────────────────────────────────
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ─── Resample commands periodically ──────────────────────────
        self._resample_commands(
            self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0
        )

        # ─── Check termination ───────────────────────────────────────
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        # Timeout flag for value bootstrapping in rsl-rl
        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        # Reset terminated environments
        self._reset_idx(self.reset_buf)

        # Update observations
        self._update_observation()

        # Store historical state
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        """Return current observations (rsl-rl compatibility)."""
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        """No privileged observations in this env."""
        return None

    # ─── Internal helpers ────────────────────────────────────────────

    def _resample_commands(self, envs_idx):
        """Sample new velocity commands for specified environments."""
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def _reset_idx(self, envs_idx=None):
        """Reset environments at specified indices."""
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.feet_air_time.zero_()
            self.last_feet_contact.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            self.feet_air_time.masked_fill_(envs_idx[:, None], 0.0)
            self.last_feet_contact.masked_fill_(envs_idx[:, None], True)

        # ─── Domain Randomization on Reset ────────────────────────────
        self._domain_rand_on_reset(envs_idx)

        # Log episode reward sums
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
                value.zero_()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
                value.masked_fill_(envs_idx, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]

        self._resample_commands(envs_idx)

    def _domain_rand_on_reset(self, envs_idx=None):
        """Apply per-episode domain randomization on reset."""
        # ── PD gain randomization ─────────────────────────────────────
        if self.randomize_kp:
            if envs_idx is None:
                # All envs reset: single random kp/kd for the batch
                rand_kp = torch.empty(1, dtype=gs.tc_float, device=gs.device).uniform_(
                    self.kp_range[0], self.kp_range[1]
                ).item()
                rand_kd = torch.empty(1, dtype=gs.tc_float, device=gs.device).uniform_(
                    self.kd_range[0], self.kd_range[1]
                ).item()
                kp_vals = torch.full((self.num_actions,), rand_kp, dtype=gs.tc_float, device=gs.device)
                kd_vals = torch.full((self.num_actions,), rand_kd, dtype=gs.tc_float, device=gs.device)
                self.robot.set_dofs_kp(kp_vals, self.motors_dof_idx)
                self.robot.set_dofs_kv(kd_vals, self.motors_dof_idx)
            else:
                # Only a few envs reset: set per-env random gains
                n_reset = envs_idx.sum().item()
                if n_reset > 0:
                    reset_indices = torch.nonzero(envs_idx).squeeze(-1)
                    for idx in reset_indices:
                        rand_kp = torch.empty(1, dtype=gs.tc_float, device=gs.device).uniform_(
                            self.kp_range[0], self.kp_range[1]
                        ).item()
                        rand_kd = torch.empty(1, dtype=gs.tc_float, device=gs.device).uniform_(
                            self.kd_range[0], self.kd_range[1]
                        ).item()
                        kp_vals = torch.full((self.num_actions,), rand_kp, dtype=gs.tc_float, device=gs.device)
                        kd_vals = torch.full((self.num_actions,), rand_kd, dtype=gs.tc_float, device=gs.device)
                        self.robot.set_dofs_kp(kp_vals, self.motors_dof_idx, envs_idx=[idx.item()])
                        self.robot.set_dofs_kv(kd_vals, self.motors_dof_idx, envs_idx=[idx.item()])

        # ── Reset push timers for reset envs ──────────────────────────
        if self.push_robots:
            if envs_idx is None:
                self.push_timer.random_(
                    self.push_interval_range[0], self.push_interval_range[1]
                )
            else:
                new_timers = torch.empty_like(self.push_timer).random_(
                    self.push_interval_range[0], self.push_interval_range[1]
                )
                self.push_timer = torch.where(envs_idx, new_timers, self.push_timer)

    def _apply_push_forces(self):
        """Apply random velocity impulses to robot base (vectorized).

        Instead of external forces (solver-only API), directly adds velocity
        kicks to the robot's base. This is the standard approach used in
        IsaacGym/IsaacLab for push perturbations.
        """
        # Count down to next push
        self.push_timer -= 1

        # Apply impulse only on the first frame of a push event
        apply_push = (self.push_timer <= 0)
        if apply_push.any():
            n_push = apply_push.sum().item()

            # Convert force to velocity impulse: Δv = F * dt / m
            # Go1 mass ≈ 12 kg, so vel_impulse = force * dt / 12
            robot_mass = 12.0
            force_mag = torch.empty(n_push, dtype=gs.tc_float, device=gs.device).uniform_(
                self.push_force_range[0], self.push_force_range[1]
            )
            vel_impulse = force_mag * self.push_duration_steps * self.dt / robot_mass

            # Random direction in XY plane
            angle = torch.empty(n_push, dtype=gs.tc_float, device=gs.device).uniform_(0, 2 * math.pi)
            vx = vel_impulse * torch.cos(angle)
            vy = vel_impulse * torch.sin(angle)

            # Get current world-frame velocity and add impulse
            cur_vel = self.robot.get_vel()  # (num_envs, 3)
            new_vel = cur_vel.clone()
            push_indices = torch.nonzero(apply_push).squeeze(-1)
            new_vel[push_indices, 0] += vx
            new_vel[push_indices, 1] += vy

            # Apply velocity change via qpos manipulation:
            # Set base velocity DOFs (indices 0,1,2 of qvel = vx,vy,vz)
            # We use set_dofs_velocity on the base freejoint (first 6 DOFs)
            base_vel_dof_idx = torch.tensor([0, 1], dtype=gs.tc_int, device=gs.device)
            vel_update = torch.stack([vx, vy], dim=1)  # (n_push, 2)
            cur_base_vel = self.robot.get_dofs_velocity(base_vel_dof_idx)
            new_base_vel = cur_base_vel.clone()
            new_base_vel[push_indices] += vel_update
            self.robot.set_dofs_velocity(new_base_vel, base_vel_dof_idx)

            # Reset timer for next push
            new_timers = torch.empty(n_push, dtype=gs.tc_int, device=gs.device).random_(
                self.push_interval_range[0], self.push_interval_range[1]
            )
            self.push_timer[apply_push] = new_timers

    def _update_observation(self):
        """Build observation buffer from current state."""
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],                       # 3
                self.projected_gravity,                                                # 3
                self.commands * self.commands_scale,                                    # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],    # 12
                self.dof_vel * self.obs_scales["dof_vel"],                             # 12
                self.actions,                                                          # 12
            ),
            dim=-1,
        )
        # Add observation noise for domain randomization
        if self.obs_noise_enabled:
            self.obs_buf += torch.randn_like(self.obs_buf) * self.obs_noise_level

    # ─── Reward functions (each returns tensor of shape (num_envs,)) ─

    def _reward_tracking_lin_vel(self):
        """Reward for tracking commanded linear velocity (xy)."""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """Reward for tracking commanded angular velocity (yaw)."""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        """Penalize vertical (z) base velocity."""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        """Penalize changes in actions (smoothness)."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """Penalize joint positions far from default standing pose."""
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        """Penalize base height away from target."""
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_torque(self):
        """Penalize large joint torques (energy efficiency)."""
        torques = self.robot.get_dofs_control_force(self.motors_dof_idx)
        return torch.sum(torch.square(torques), dim=1)

    def _reward_orientation(self):
        """Penalize non-upright orientation."""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_alive(self):
        """Constant alive bonus per step."""
        return torch.ones((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    def _reward_feet_air_time(self):
        """Reward feet air time to encourage proper trot gait cycle.

        This is the key reward for natural quadruped locomotion.
        Rewards each foot for spending ~0.25s in the air (half of trot cycle).
        Uses self.landed_air_time which captures the air time at the moment
        of landing, before it gets zeroed.
        """
        target_air_time = self.reward_cfg.get("target_feet_air_time", 0.25)
        # landed_air_time has non-zero values only for feet that just landed
        air_time_reward = torch.sum(
            (self.landed_air_time - target_air_time * self.first_contact.to(dtype=gs.tc_float)).clamp(min=0.0),
            dim=1,
        )
        return air_time_reward

    def _reward_dof_acc(self):
        """Penalize joint accelerations (smoother gait)."""
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """Penalize body contacts other than feet (e.g. knee/thigh hitting ground)."""
        # Penalize if base is too low (body scraping ground)
        return (self.base_pos[:, 2] < 0.20).to(dtype=gs.tc_float)

    def _reward_feet_contact_forces(self):
        """Penalize high foot contact forces (gentler foot placement)."""
        # Use base z-velocity as proxy for impact forces
        return torch.square(self.base_lin_vel[:, 2]).clamp(max=1.0)

    def _reward_ang_vel_xy(self):
        """Penalize xy-axis angular velocity (body rolling/pitching)."""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_hip_deviation(self):
        """Strongly penalize hip abduction/adduction away from 0.

        Hip joints (FR_hip, FL_hip, RR_hip, RL_hip) at config indices [0,3,6,9]
        should stay near 0.0 to keep legs under the body.
        """
        hip_indices = [0, 3, 6, 9]  # indices in config joint_names order
        hip_pos = self.dof_pos[:, hip_indices]
        hip_default = self.default_dof_pos[hip_indices]
        return torch.sum(torch.square(hip_pos - hip_default), dim=1)

    def _reward_symmetry(self):
        """Reward diagonal symmetry for trot gait pattern.

        In a trot gait, DIAGONAL pairs move together:
          - FR (indices 1,2) with RL (indices 10,11)
          - FL (indices 4,5) with RR (indices 7,8)
        This allows proper alternating leg movement while maintaining
        coordination.
        Config order: [FR_hip,FR_thigh,FR_calf, FL_hip,FL_thigh,FL_calf,
                        RR_hip,RR_thigh,RR_calf, RL_hip,RL_thigh,RL_calf]
        """
        # Diagonal pair 1: FR with RL (move together in trot)
        fr_thigh_calf = self.dof_pos[:, [1, 2]]
        rl_thigh_calf = self.dof_pos[:, [10, 11]]
        diag1_error = torch.sum(torch.square(fr_thigh_calf - rl_thigh_calf), dim=1)

        # Diagonal pair 2: FL with RR (move together in trot)
        fl_thigh_calf = self.dof_pos[:, [4, 5]]
        rr_thigh_calf = self.dof_pos[:, [7, 8]]
        diag2_error = torch.sum(torch.square(fl_thigh_calf - rr_thigh_calf), dim=1)

        return diag1_error + diag2_error

    def _reward_stumble(self):
        """Penalize base getting too low (body dragging on ground)."""
        return (self.base_pos[:, 2] < 0.22).to(dtype=gs.tc_float)
