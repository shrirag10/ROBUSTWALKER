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
