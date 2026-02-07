#!/usr/bin/env python3
"""
Visualization script for Go1 locomotion policy.

Usage:
    python scripts/visualize.py --checkpoint logs/best_model/model.zip
    python scripts/visualize.py --checkpoint logs/best_model/model.zip --record video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import mujoco
import mujoco.viewer

from robustwalker.envs.go1_env import Go1Env


def visualize_policy(
    checkpoint_path: str,
    n_episodes: int = 5,
    record_path: str | None = None,
    target_velocity: float = 0.8,
) -> None:
    """
    Visualize trained policy with MuJoCo viewer.
    
    Args:
        checkpoint_path: Path to trained model
        n_episodes: Number of episodes to visualize
        record_path: Optional path to save video
        target_velocity: Forward velocity command
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Create environment
    env = Go1Env(
        control_mode="position",
        enable_domain_rand=False,
        render_mode="human",
    )
    
    # Wrap for compatibility with SB3
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if exists
    vecnorm_path = checkpoint_path.parent / "vecnormalize.pkl"
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load model
    model = PPO.load(str(checkpoint_path))
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Visualizing {n_episodes} episodes...")
    print(f"Target velocity: {target_velocity} m/s")
    print(f"Press Ctrl+C to exit")
    
    # Setup recording if requested
    frames = [] if record_path else None
    
    try:
        for episode in range(n_episodes):
            obs = vec_env.reset()
            env.set_commands(target_velocity, 0.0, 0.0)
            
            total_reward = 0
            step = 0
            done = False
            
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            
            while not done:
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = vec_env.step(action)
                total_reward += reward[0]
                step += 1
                
                # Render
                env.render()
                
                # Record frame if recording
                if frames is not None:
                    # Get RGB frame
                    frame = env.renderer.render() if env.renderer else None
                    if frame is not None:
                        frames.append(frame)
                
                # Print velocity info periodically
                if step % 50 == 0 and 'base_velocity' in info[0]:
                    vel = info[0]['base_velocity']
                    print(f"  Step {step}: vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})")
                
                # Control visualization speed
                time.sleep(1.0 / 50.0)  # 50 Hz
                
                done = done[0]
            
            print(f"  Episode finished: reward={total_reward:.2f}, length={step}")
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted")
    
    finally:
        vec_env.close()
        
        # Save video if recording
        if frames and record_path:
            save_video(frames, record_path)


def visualize_untrained(n_steps: int = 500) -> None:
    """Visualize environment with random actions."""
    print("Visualizing untrained policy (random actions)...")
    
    env = Go1Env(
        control_mode="position",
        enable_domain_rand=True,
        render_mode="human",
    )
    
    try:
        obs, info = env.reset()
        
        for step in range(n_steps):
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            time.sleep(1.0 / 50.0)
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nVisualization interrupted")
    
    finally:
        env.close()


def save_video(frames: list, path: str, fps: int = 50) -> None:
    """Save frames as video."""
    try:
        import imageio
        print(f"Saving video to {path}...")
        imageio.mimsave(path, frames, fps=fps)
        print(f"Video saved ({len(frames)} frames)")
    except ImportError:
        print("imageio not installed. Install with: pip install imageio imageio-ffmpeg")


def main():
    parser = argparse.ArgumentParser(description="Visualize Go1 locomotion policy")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to save video recording",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=0.8,
        help="Target forward velocity (m/s)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Visualize with random actions (no checkpoint needed)",
    )
    
    args = parser.parse_args()
    
    if args.random:
        visualize_untrained()
    elif args.checkpoint:
        visualize_policy(
            args.checkpoint,
            n_episodes=args.n_episodes,
            record_path=args.record,
            target_velocity=args.velocity,
        )
    else:
        print("Error: Either --checkpoint or --random must be specified")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
