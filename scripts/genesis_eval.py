#!/usr/bin/env python3
"""
Evaluation script for Go1 locomotion trained with Genesis.

Loads a trained policy and runs it in the Genesis viewer or headless mode.

Usage:
    # Interactive viewer (requires display)
    python scripts/genesis_eval.py -e go1-walking --ckpt 50

    # Headless mode (no display needed, prints metrics)
    python scripts/genesis_eval.py -e go1-walking --ckpt 50 --headless

    # Run for specific number of steps
    python scripts/genesis_eval.py -e go1-walking --ckpt 50 --headless --steps 500
"""

import argparse
import os
import pickle
import sys
from importlib import metadata
from pathlib import Path

import torch

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Go1 locomotion policy trained with Genesis"
    )
    parser.add_argument(
        "-e", "--exp_name", type=str, default="go1-walking",
        help="Experiment name (must match training run)",
    )
    parser.add_argument(
        "--ckpt", type=int, default=100,
        help="Checkpoint iteration to load",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help="Number of evaluation environments",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without viewer (headless mode, prints metrics)",
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of steps to run in headless mode",
    )
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"

    # Load saved configs
    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfg_path):
        print(f"Error: Config not found at {cfg_path}")
        if os.path.exists("logs"):
            print(f"Available experiments: {os.listdir('logs/')}")
        return 1

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(cfg_path, "rb")
    )

    # Disable reward scales for evaluation (no training signal needed)
    reward_cfg["reward_scales"] = {}

    show_viewer = not args.headless

    print("=" * 60)
    print("RobustWalker Genesis Evaluation")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Mode:       {'Interactive viewer' if show_viewer else 'Headless'}")
    print(f"Log directory: {log_dir}")
    print("=" * 60)

    # Initialize Genesis
    gs.init(backend=gs.cpu)

    # Create environment
    env = Go1GenesisEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )

    # Load trained policy
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(resume_path):
        print(f"Error: Checkpoint not found at {resume_path}")
        available = [f for f in os.listdir(log_dir) if f.startswith("model_")]
        print(f"Available checkpoints: {available}")
        return 1

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    print(f"\nLoaded policy from: {resume_path}")

    # Run evaluation loop
    obs, _ = env.reset()

    if show_viewer:
        print("Running evaluation... Press Ctrl+C to exit.\n")
        with torch.no_grad():
            try:
                while True:
                    actions = policy(obs)
                    obs, rews, dones, infos = env.step(actions)
            except KeyboardInterrupt:
                print("\nEvaluation stopped.")
    else:
        # Headless mode: run fixed steps and report metrics
        print(f"Running {args.steps} steps in headless mode...\n")
        total_reward = 0.0
        num_episodes = 0
        episode_lengths = []
        base_heights = []

        with torch.no_grad():
            for step in range(args.steps):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                total_reward += rews.sum().item()

                # Track base height
                base_heights.append(env.base_pos[:, 2].mean().item())

                if dones.any():
                    num_episodes += dones.sum().item()

                # Print progress every 100 steps
                if (step + 1) % 100 == 0:
                    avg_height = sum(base_heights[-100:]) / min(100, len(base_heights))
                    print(
                        f"  Step {step+1:4d}/{args.steps} | "
                        f"avg_height={avg_height:.3f}m | "
                        f"episodes_done={num_episodes}"
                    )

        avg_height = sum(base_heights) / len(base_heights)
        print(f"\n{'=' * 60}")
        print("Evaluation Summary")
        print(f"{'=' * 60}")
        print(f"  Total steps:     {args.steps}")
        print(f"  Total reward:    {total_reward:.2f}")
        print(f"  Avg base height: {avg_height:.4f} m")
        print(f"  Episodes done:   {num_episodes}")
        print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
