#!/usr/bin/env python3
"""
Training script for Go1 locomotion using Genesis + rsl-rl PPO.

Uses Genesis's GPU-accelerated parallel simulation for massively-parallel training.

Usage:
    python scripts/genesis_train.py
    python scripts/genesis_train.py -e my-experiment -B 4096 --max_iterations 300
    python scripts/genesis_train.py --config configs/genesis.yaml
"""

import argparse
import os
import pickle
import shutil
import sys
from importlib import metadata
from pathlib import Path

import yaml

# Verify rsl-rl installation
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please install 'rsl-rl-lib==2.2.4': pip install rsl-rl-lib==2.2.4"
    ) from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robustwalker.envs.genesis_env import Go1GenesisEnv


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_cfgs_from_yaml(config: dict) -> tuple:
    """Extract env, obs, reward, command configs from YAML."""
    env_cfg = config["env"]
    obs_cfg = config["obs"]
    reward_cfg = config["reward"]
    command_cfg = config["command"]
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_train_cfg(config: dict, exp_name: str, max_iterations: int) -> dict:
    """Build rsl-rl training config from YAML."""
    training = config.get("training", {})

    train_cfg = {
        "algorithm": training.get("algorithm", {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        }),
        "init_member_classes": {},
        "policy": training.get("policy", {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        }),
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": training.get("runner", {}).get("log_interval", 1),
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": training.get("runner", {}).get("num_steps_per_env", 24),
        "save_interval": training.get("runner", {}).get("save_interval", 50),
        "empirical_normalization": None,
        "seed": training.get("seed", 42),
    }

    return train_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Train Go1 locomotion with Genesis + rsl-rl PPO"
    )
    parser.add_argument(
        "-e", "--exp_name", type=str, default="go1-walking",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "-B", "--num_envs", type=int, default=None,
        help="Number of parallel environments (overrides config)",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None,
        help="Maximum training iterations (overrides config)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/genesis.yaml",
        help="Path to Genesis config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume/fine-tune from (e.g. logs/go1-walking-v8/model_1400.pt)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs_from_yaml(config)

    # Apply CLI overrides
    num_envs = args.num_envs or config.get("training", {}).get("num_envs", 4096)
    max_iterations = args.max_iterations or config.get("training", {}).get("max_iterations", 300)
    seed = config.get("training", {}).get("seed", 42)

    train_cfg = get_train_cfg(config, args.exp_name, max_iterations)

    # Setup logging directory
    log_dir = f"logs/{args.exp_name}"
    if not args.resume:  # Don't wipe logs when fine-tuning
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Save configs for later evaluation
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    print("=" * 60)
    print("RobustWalker Genesis Training")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Environments: {num_envs}")
    print(f"Max iterations: {max_iterations}")
    print(f"Log directory: {log_dir}")
    print("=" * 60)

    # Initialize Genesis
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=seed,
    )

    # Create environment
    env = Go1GenesisEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    # Create rsl-rl runner and train
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Load checkpoint for fine-tuning if specified
    if args.resume:
        print(f"\nLoading checkpoint for fine-tuning: {args.resume}")
        runner.load(args.resume)

    print("\nStarting training...")
    print("Monitor with: tensorboard --logdir logs/")
    runner.learn(
        num_learning_iterations=max_iterations,
        init_at_random_ep_len=True,
    )

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Checkpoints saved to: {log_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
