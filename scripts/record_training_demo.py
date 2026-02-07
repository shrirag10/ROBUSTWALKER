#!/usr/bin/env python3
"""
Record video showing Go1 in training environment with terrain and domain randomization.

Shows the complete training setup with:
- Rough terrain (heightfield)
- Domain randomization (friction, payload, pushes)
- Training overlay showing progress

Usage:
    python scripts/record_training_demo.py --output training_demo.webm --duration 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mujoco

from robustwalker.envs.go1_env import Go1Env
from robustwalker.envs.domain_rand import DomainRandomizationConfig


def add_overlay(frame: np.ndarray, step: int, episode: int, reward: float, 
                velocity: float, friction: float, payload: float) -> np.ndarray:
    """Add training info overlay to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Try to use a monospace font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Semi-transparent background
    overlay_height = 120
    overlay = Image.new('RGBA', (220, overlay_height), (0, 0, 0, 180))
    img.paste(overlay, (10, 10), overlay)
    
    # Training info
    y = 15
    draw.text((15, y), "ROBUSTWALKER TRAINING", fill=(0, 255, 100), font=title_font)
    y += 22
    draw.text((15, y), f"Episode: {episode}", fill=(255, 255, 255), font=font)
    y += 18
    draw.text((15, y), f"Step: {step:,}", fill=(255, 255, 255), font=font)
    y += 18
    draw.text((15, y), f"Reward: {reward:.2f}", fill=(255, 255, 0), font=font)
    y += 18
    draw.text((15, y), f"Velocity: {velocity:.2f} m/s", fill=(100, 200, 255), font=font)
    
    # Domain randomization info (bottom right)
    dr_overlay = Image.new('RGBA', (180, 60), (0, 0, 0, 180))
    img.paste(dr_overlay, (frame.shape[1] - 190, 10), dr_overlay)
    
    draw.text((frame.shape[1] - 185, 15), "DOMAIN RAND", fill=(255, 100, 100), font=title_font)
    draw.text((frame.shape[1] - 185, 35), f"Friction: {friction:.2f}", fill=(255, 255, 255), font=font)
    draw.text((frame.shape[1] - 185, 50), f"Payload: {payload:.1f} kg", fill=(255, 255, 255), font=font)
    
    return np.array(img)


def record_training_demo(
    output_path: str = "training_demo.webm",
    duration: float = 60.0,
    fps: int = 30,
    width: int = 800,
    height: int = 600,
    checkpoint_path: str | None = None,
) -> None:
    """
    Record demonstration of training environment.
    """
    print(f"Recording Training Environment Demo")
    print(f"=" * 50)
    print(f"Output: {output_path}")
    print(f"Duration: {duration}s @ {fps} FPS")
    print(f"Features: Terrain + Domain Randomization + Overlay")
    print(f"=" * 50)
    
    # Ensure output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create domain randomization config
    dr_config = DomainRandomizationConfig(
        friction_range=(0.5, 1.2),
        payload_range=(0.0, 4.0),
        motor_strength_range=(0.9, 1.1),
        push_force_range=(0.0, 15.0),
        push_interval=(3.0, 8.0),
        randomize_friction=True,
        randomize_payload=True,
        randomize_motor_strength=True,
        randomize_pushes=True,
    )
    
    # Create environment with domain randomization
    env = Go1Env(
        control_mode="position",
        enable_domain_rand=True,
        domain_rand_config=dr_config,
        render_mode=None,
    )
    
    # Set framebuffer size
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    
    # Setup renderer
    renderer = mujoco.Renderer(env.model, width=width, height=height)
    
    # Load model if checkpoint provided
    model = None
    if checkpoint_path:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        
        checkpoint_path = Path(checkpoint_path)
        vec_env = DummyVecEnv([lambda: Go1Env(control_mode="position", enable_domain_rand=True)])
        
        vecnorm_path = checkpoint_path.parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        
        model = PPO.load(str(checkpoint_path))
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    # Video writer
    ext = output_path.suffix.lower()
    if ext == '.gif':
        writer = imageio.get_writer(str(output_path), fps=fps, mode='I')
    elif ext == '.webm':
        writer = imageio.get_writer(str(output_path), fps=fps, codec='libvpx-vp9')
    else:
        writer = imageio.get_writer(str(output_path), fps=fps)
    
    # Timing
    control_dt = 1.0 / 50.0
    render_dt = 1.0 / fps
    total_frames = int(duration * fps)
    
    # Tracking
    current_friction = 0.8
    current_payload = 0.0
    episode = 0
    total_reward = 0.0
    episode_steps = 0
    global_steps = 0
    
    # Reset
    obs, info = env.reset()
    if 'domain_rand' in info:
        current_friction = info['domain_rand'].get('friction', 0.8)
        current_payload = info['domain_rand'].get('payload', 0.0)
    
    env.set_commands(0.8, 0.0, 0.0)  # Walk forward at 0.8 m/s
    
    print(f"\nRecording {total_frames} frames...")
    
    sim_time = 0.0
    last_render_time = 0.0
    frame_count = 0
    
    try:
        while frame_count < total_frames:
            # Get action
            if model:
                # Use trained policy
                action, _ = model.predict(obs, deterministic=False)  # Stochastic for natural look
            else:
                # Use random/demonstrative actions
                t = sim_time * 2 * np.pi * 0.5  # Slow oscillation
                action = 0.3 * np.sin(t + np.arange(12) * 0.5)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1
            global_steps += 1
            
            # Get velocity
            velocity = info.get('base_velocity', np.zeros(3))[0]
            
            sim_time += control_dt
            
            # Handle episode end
            if terminated or truncated:
                episode += 1
                obs, info = env.reset()
                if 'domain_rand' in info:
                    current_friction = info['domain_rand'].get('friction', 0.8)
                    current_payload = info['domain_rand'].get('payload', 0.0)
                env.set_commands(0.8, 0.0, 0.0)
                total_reward = 0.0
                episode_steps = 0
            
            # Render
            if sim_time - last_render_time >= render_dt:
                renderer.update_scene(env.data, camera="tracking")
                frame = renderer.render()
                
                # Add overlay
                frame = add_overlay(
                    frame,
                    step=global_steps,
                    episode=episode + 1,
                    reward=total_reward,
                    velocity=velocity,
                    friction=current_friction,
                    payload=current_payload,
                )
                
                writer.append_data(frame)
                
                last_render_time = sim_time
                frame_count += 1
                
                if frame_count % (fps * 10) == 0:
                    progress = frame_count / total_frames * 100
                    print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    except KeyboardInterrupt:
        print("\nRecording interrupted")
    
    finally:
        writer.close()
        env.close()
    
    print(f"\nVideo saved: {output_path}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {frame_count / fps:.2f}s")
    if output_path.exists():
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Record training environment demo")
    parser.add_argument("--output", "-o", type=str, default="videos/training_demo.webm")
    parser.add_argument("--duration", "-d", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Use trained policy (optional)")
    
    args = parser.parse_args()
    
    record_training_demo(
        output_path=args.output,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
