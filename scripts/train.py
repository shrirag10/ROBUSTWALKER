#!/usr/bin/env python3
"""
Training script for Go1 locomotion using PPO.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/custom.yaml
    python scripts/train.py --timesteps 5000000 --n-envs 4
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from robustwalker.envs.go1_env import Go1Env
from robustwalker.envs.domain_rand import DomainRandomizationConfig
from robustwalker.rewards.locomotion import RewardConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(
    rank: int,
    seed: int,
    control_mode: str = "position",
    enable_domain_rand: bool = True,
    domain_rand_config: dict | None = None,
    reward_config: dict | None = None,
) -> callable:
    """
    Create environment factory function for vectorized envs.
    """
    def _init():
        # Create domain randomization config
        dr_config = None
        if enable_domain_rand and domain_rand_config:
            dr_config = DomainRandomizationConfig(**domain_rand_config)
        
        # Create reward config
        rw_config = None
        if reward_config:
            rw_config = RewardConfig(**reward_config)
        
        env = Go1Env(
            control_mode=control_mode,
            enable_domain_rand=enable_domain_rand,
            domain_rand_config=dr_config,
            reward_config=rw_config,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    return _init


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.timesteps:
        config['total_timesteps'] = args.timesteps
    if args.n_envs:
        config['n_envs'] = args.n_envs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
        
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"go1_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    print(f"=" * 60)
    print(f"RobustWalker Training")
    print(f"=" * 60)
    print(f"Log directory: {log_dir}")
    print(f"Timesteps: {config['total_timesteps']:,}")
    print(f"Environments: {config['n_envs']}")
    print(f"Control mode: {config.get('control_mode', 'position')}")
    print(f"=" * 60)
    
    # Create vectorized environment
    n_envs = config['n_envs']
    seed = args.seed
    
    env_fns = [
        make_env(
            rank=i,
            seed=seed,
            control_mode=config.get('control_mode', 'position'),
            enable_domain_rand=config.get('domain_rand', {}).get('enabled', True),
            domain_rand_config=config.get('domain_rand', {}),
            reward_config=config.get('rewards', {}),
        )
        for i in range(n_envs)
    ]
    
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    
    # Wrap with VecNormalize for observation/reward normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create evaluation environment
    eval_env_fn = make_env(
        rank=0,
        seed=seed + 1000,
        control_mode=config.get('control_mode', 'position'),
        enable_domain_rand=False,  # No randomization for eval
    )
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
    )
    
    # Configure policy
    policy_kwargs = config.get('policy_kwargs', {})
    if 'net_arch' in policy_kwargs:
        # Convert config format to SB3 format
        net_arch = policy_kwargs['net_arch']
        if isinstance(net_arch, dict):
            policy_kwargs['net_arch'] = dict(
                pi=net_arch.get('pi', [256, 256]),
                vf=net_arch.get('vf', [256, 256]),
            )
    
    # Map activation function
    if policy_kwargs.get('activation_fn') == 'elu':
        policy_kwargs['activation_fn'] = torch.nn.ELU
    elif policy_kwargs.get('activation_fn') == 'relu':
        policy_kwargs['activation_fn'] = torch.nn.ReLU
    elif policy_kwargs.get('activation_fn') == 'tanh':
        policy_kwargs['activation_fn'] = torch.nn.Tanh
        
    # Create PPO model
    model = PPO(
        policy=config.get('policy', 'MlpPolicy'),
        env=env,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 64),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        seed=seed,
        device='auto',
    )
    
    print(f"\nModel architecture:")
    print(model.policy)
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('save_freq', 100000) // n_envs,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="go1",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=config.get('eval_freq', 50000) // n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    # Train
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_path = log_dir / "final_model"
    model.save(str(final_path / "model"))
    env.save(str(final_path / "vecnormalize.pkl"))
    
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {log_dir / 'best_model'}")
    print(f"TensorBoard logs: tensorboard --logdir={log_dir / 'tensorboard'}")
    print(f"{'=' * 60}")
    
    # Cleanup
    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Go1 locomotion policy")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run short training for testing",
    )
    
    args = parser.parse_args()
    
    # Test mode: short training run
    if args.test_mode:
        args.timesteps = 10000
        args.n_envs = 2
        print("Running in test mode with 10k steps...")
    
    train(args)


if __name__ == "__main__":
    main()
