#!/usr/bin/env python3
"""
Record a GIF of the Go1 robot walking in Genesis.

Usage:
    python scripts/record_genesis_gif.py -e go1-walking-v10 --ckpt 1800 --steps 200 --output assets/genesis_trot.gif
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
import numpy as np

try:
    from importlib import metadata
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib==2.2.4'") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from robustwalker.envs.genesis_env import Go1GenesisEnv


def main():
    parser = argparse.ArgumentParser(description="Record Go1 trotting GIF")
    parser.add_argument("-e", "--exp_name", type=str, default="go1-walking-v10")
    parser.add_argument("--ckpt", type=int, default=1800)
    parser.add_argument("--steps", type=int, default=200, help="Frames to record")
    parser.add_argument("--output", type=str, default="assets/genesis_trot.gif")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.exists(cfg_path):
        print(f"Error: {cfg_path} not found")
        return 1

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(cfg_path, "rb")
    )
    reward_cfg["reward_scales"] = {}

    # Init Genesis with viewer for rendering
    gs.init(backend=gs.cpu)

    env = Go1GenesisEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Load policy
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)
    print(f"Loaded policy from: {resume_path}")

    # Run and capture frames
    obs, _ = env.reset()
    frames = []

    print(f"Recording {args.steps} frames...")
    with torch.no_grad():
        for step in range(args.steps):
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)

            # Capture frame from viewer
            frame = env.scene.viewer.render()
            if frame is not None:
                frames.append(frame)

            if (step + 1) % 50 == 0:
                print(f"  Frame {step+1}/{args.steps}")

    # Save as GIF
    if frames:
        try:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in frames]
            duration = int(1000 / args.fps)
            pil_frames[0].save(
                args.output,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration,
                loop=0,
                optimize=True,
            )
            print(f"\nSaved GIF: {args.output} ({len(frames)} frames, {args.fps} FPS)")
        except ImportError:
            print("Pillow not installed, saving frames as numpy array")
            np.save(args.output.replace(".gif", ".npy"), np.stack(frames))
    else:
        print("Warning: No frames captured")

    return 0


if __name__ == "__main__":
    sys.exit(main())
