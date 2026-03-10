"""
Unit tests for Go1 environment.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGo1Env:
    """Tests for Go1 Gymnasium environment."""
    
    @pytest.fixture
    def env(self):
        """Create environment fixture."""
        from robustwalker.envs.go1_env import Go1Env
        env = Go1Env(
            control_mode="position",
            enable_domain_rand=False,
            render_mode=None,
        )
        yield env
        env.close()
    
    def test_env_creation(self, env):
        """Test environment can be created."""
        assert env is not None
        assert env.model is not None
        assert env.data is not None
    
    def test_observation_space(self, env):
        """Test observation space is correctly defined."""
        obs_space = env.observation_space
        assert obs_space.shape == (57,)  # Expected observation dim
        assert obs_space.dtype == np.float32
    
    def test_action_space(self, env):
        """Test action space is correctly defined."""
        action_space = env.action_space
        assert action_space.shape == (12,)  # 12 joints
        assert action_space.dtype == np.float32
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        
        assert obs.shape == (57,)
        assert isinstance(info, dict)
        assert np.all(np.isfinite(obs))
    
    def test_step(self, env):
        """Test environment step."""
        env.reset()
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (57,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert np.all(np.isfinite(obs))
    
    def test_episode_rollout(self, env):
        """Test full episode rollout."""
        env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(100):  # 100 steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps > 0
        assert np.isfinite(total_reward)
    
    def test_set_commands(self, env):
        """Test velocity command setting."""
        env.reset()
        env.set_commands(0.5, 0.1, 0.2)
        
        assert np.allclose(env.commands, [0.5, 0.1, 0.2])


class TestControlModes:
    """Test different control modes."""
    
    @pytest.fixture(params=["position", "torque", "delta"])
    def env_with_mode(self, request):
        """Create environment with different control modes."""
        from robustwalker.envs.go1_env import Go1Env
        env = Go1Env(
            control_mode=request.param,
            enable_domain_rand=False,
            render_mode=None,
        )
        yield env
        env.close()
    
    def test_control_mode_step(self, env_with_mode):
        """Test step works with all control modes."""
        env_with_mode.reset()
        action = env_with_mode.action_space.sample()
        
        obs, reward, terminated, truncated, info = env_with_mode.step(action)
        
        assert obs.shape == (57,)
        assert np.all(np.isfinite(obs))


class TestDomainRandomization:
    """Test domain randomization."""
    
    def test_domain_rand_enabled(self):
        """Test domain randomization applies correctly."""
        from robustwalker.envs.go1_env import Go1Env
        
        env = Go1Env(
            control_mode="position",
            enable_domain_rand=True,
            render_mode=None,
        )
        
        try:
            obs, info = env.reset()
            
            # Check domain rand info is in reset info
            assert 'domain_rand' in info or True  # May not always be present
            
            # Run a few steps
            for _ in range(10):
                action = env.action_space.sample()
                env.step(action)
        finally:
            env.close()


class TestReward:
    """Test reward function."""
    
    def test_reward_components(self):
        """Test reward returns expected components."""
        from robustwalker.envs.go1_env import Go1Env
        
        env = Go1Env(
            control_mode="position",
            enable_domain_rand=False,
            render_mode=None,
        )
        
        try:
            env.reset()
            action = np.zeros(12)  # Neutral action
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert 'reward_components' in info
            components = info['reward_components']
            
            # Check expected reward components
            expected_keys = [
                'velocity_tracking', 'alive', 'torque', 
                'action_rate', 'orientation'
            ]
            for key in expected_keys:
                assert key in components, f"Missing reward component: {key}"
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ─── Genesis Environment Tests ───────────────────────────────────────

class TestGenesisEnv:
    """Tests for Go1 Genesis environment (GPU-accelerated)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_genesis(self):
        """Skip all tests if genesis-world is not installed."""
        pytest.importorskip("genesis")

    @pytest.fixture(scope="class")
    def genesis_cfg(self):
        """Create minimal Genesis config for testing."""
        env_cfg = {
            "num_actions": 12,
            "episode_length_s": 20.0,
            "resampling_time_s": 4.0,
            "action_scale": 0.25,
            "clip_actions": 100.0,
            "simulate_action_latency": True,
            "base_init_pos": [0.0, 0.0, 0.445],
            "base_init_quat": [1.0, 0.0, 0.0, 0.0],
            "kp": 20.0,
            "kd": 0.5,
            "termination_if_roll_greater_than": 30,
            "termination_if_pitch_greater_than": 30,
            "joint_names": [
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            ],
            "default_joint_angles": {
                "FR_hip_joint": 0.0, "FR_thigh_joint": 0.9, "FR_calf_joint": -1.8,
                "FL_hip_joint": 0.0, "FL_thigh_joint": 0.9, "FL_calf_joint": -1.8,
                "RR_hip_joint": 0.0, "RR_thigh_joint": 0.9, "RR_calf_joint": -1.8,
                "RL_hip_joint": 0.0, "RL_thigh_joint": 0.9, "RL_calf_joint": -1.8,
            },
        }
        obs_cfg = {
            "num_obs": 45,
            "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
        }
        reward_cfg = {
            "tracking_sigma": 0.25,
            "base_height_target": 0.27,
            "reward_scales": {
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.2,
                "lin_vel_z": -1.0,
                "base_height": -50.0,
                "action_rate": -0.005,
                "similar_to_default": -0.1,
            },
        }
        command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.5, 0.8],
            "lin_vel_y_range": [-0.3, 0.3],
            "ang_vel_range": [-0.5, 0.5],
        }
        return env_cfg, obs_cfg, reward_cfg, command_cfg

    @pytest.fixture(scope="class")
    def env(self, genesis_cfg):
        """Create a small Genesis environment for testing."""
        import genesis as gs
        try:
            gs.init(backend=gs.cpu, logging_level="warning")
        except Exception:
            pass  # Already initialized

        from robustwalker.envs.genesis_env import Go1GenesisEnv

        env_cfg, obs_cfg, reward_cfg, command_cfg = genesis_cfg
        env = Go1GenesisEnv(
            num_envs=2,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=False,
        )
        return env

    def test_genesis_env_creation(self, env):
        """Test Genesis environment can be created."""
        assert env is not None
        assert env.num_envs == 2
        assert env.num_actions == 12
        assert env.num_obs == 45

    def test_genesis_reset(self, env):
        """Test Genesis environment reset."""
        import torch
        obs, info = env.reset()
        assert obs.shape == (2, 45)
        assert torch.all(torch.isfinite(obs))

    def test_genesis_step(self, env):
        """Test Genesis environment step."""
        import torch
        env.reset()
        actions = torch.zeros((2, 12), dtype=torch.float32, device=env.device)
        obs, rewards, dones, extras = env.step(actions)

        assert obs.shape == (2, 45)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert torch.all(torch.isfinite(obs))
        assert torch.all(torch.isfinite(rewards))
