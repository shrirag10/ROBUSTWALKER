#!/usr/bin/env python3
"""
Record video of Go1 locomotion simulation (memory-efficient version).

Writes frames directly to video file instead of storing in memory.

Usage:
    python scripts/record_video.py --checkpoint logs/best_model/model.zip --output video.mp4
    python scripts/record_video.py --random --output random_policy.mp4 --duration 120
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robustwalker.envs.go1_env import Go1Env


def record_video(
    checkpoint_path: str | None = None,
    output_path: str = "simulation.mp4",
    duration: float = 10.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera: str | None = None,
    target_velocity: float = 0.8,
    random_policy: bool = False,
) -> None:
    """
    Record video of Go1 simulation (memory-efficient streaming).
    """
    import imageio
    
    print(f"Recording Go1 Simulation Video")
    print(f"=" * 50)
    print(f"Output: {output_path}")
    print(f"Duration: {duration}s @ {fps} FPS")
    print(f"Resolution: {width}x{height}")
    
    if checkpoint_path and not random_policy:
        print(f"Model: {checkpoint_path}")
    else:
        print(f"Policy: Random actions")
    print(f"=" * 50)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = Go1Env(
        control_mode="position",
        enable_domain_rand=False,
        render_mode=None,
    )
    
    # Set offscreen framebuffer size
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    
    # Setup renderer
    renderer = mujoco.Renderer(env.model, width=width, height=height)
    
    # Load model if provided
    model = None
    vec_env = None
    
    if checkpoint_path and not random_policy:
        checkpoint_path = Path(checkpoint_path)
        
        vec_env = DummyVecEnv([lambda: Go1Env(
            control_mode="position",
            enable_domain_rand=False,
        )])
        
        vecnorm_path = checkpoint_path.parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        
        model = PPO.load(str(checkpoint_path))
        print(f"Loaded model from {checkpoint_path}")
    
    # Calculate timing
    control_dt = 1.0 / 50.0
    render_dt = 1.0 / fps
    total_frames = int(duration * fps)
    
    # Reset environment
    if vec_env:
        obs = vec_env.reset()
        raw_env = vec_env.envs[0]
        if hasattr(raw_env, 'env'):
            raw_env = raw_env.env
    else:
        obs, _ = env.reset()
        raw_env = env
    
    raw_env.set_commands(target_velocity, 0.0, 0.0)
    
    print(f"\nRecording {total_frames} frames (streaming to disk)...")
    
    # Use imageio writer for streaming (memory efficient)
    # Support multiple formats: .gif, .mp4, .avi
    ext = output_path.suffix.lower()
    if ext == '.gif':
        writer = imageio.get_writer(str(output_path), fps=fps, mode='I')
    elif ext == '.webm':
        writer = imageio.get_writer(str(output_path), fps=fps, codec='libvpx-vp9')
    else:
        # Default: use simple format without h264
        writer = imageio.get_writer(str(output_path), fps=fps)
    
    sim_time = 0.0
    last_render_time = 0.0
    frame_count = 0
    
    try:
        while frame_count < total_frames:
            # Get action
            if model and vec_env:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = vec_env.step(action)
                if done[0]:
                    obs = vec_env.reset()
                    raw_env.set_commands(target_velocity, 0.0, 0.0)
            else:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
                    env.set_commands(target_velocity, 0.0, 0.0)
            
            sim_time += control_dt
            
            # Render at specified FPS
            if sim_time - last_render_time >= render_dt:
                data_to_use = raw_env.data if vec_env else env.data
                renderer.update_scene(data_to_use, camera=camera or "tracking")
                
                frame = renderer.render()
                writer.append_data(frame)
                
                last_render_time = sim_time
                frame_count += 1
                
                # Progress every 10 seconds of video
                if frame_count % (fps * 10) == 0:
                    progress = frame_count / total_frames * 100
                    print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    except KeyboardInterrupt:
        print("\nRecording interrupted")
    
    finally:
        writer.close()
        env.close()
        if vec_env:
            vec_env.close()
    
    print(f"\nVideo saved: {output_path}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {frame_count / fps:.2f}s")
    if output_path.exists():
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Record Go1 simulation video")
    
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default="simulation.mp4")
    parser.add_argument("--duration", "-d", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--velocity", type=float, default=0.8)
    parser.add_argument("--random", action="store_true")
    
    args = parser.parse_args()
    
    record_video(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera=args.camera,
        target_velocity=args.velocity,
        random_policy=args.random,
    )


if __name__ == "__main__":
    main()
