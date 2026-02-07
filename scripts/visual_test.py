#!/usr/bin/env python3
"""
Visual evaluation of Go1 policy with MuJoCo viewer.

Shows the robot running acceptance tests in real-time visualization.
"""

import argparse
import sys
from pathlib import Path
import time

import numpy as np
import mujoco
import mujoco.viewer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robustwalker.envs.go1_env import Go1Env


def visual_velocity_test(
    checkpoint_path: str | None = None,
    target_velocity: float = 0.8,
    duration: float = 20.0,
):
    """Run velocity tracking test with MuJoCo visualization."""
    print(f"\n{'='*60}")
    print(f"VISUAL TEST: Velocity Tracking")
    print(f"Target: {target_velocity} m/s for {duration}s")
    print(f"{'='*60}")
    
    # Create environment for visualization
    vis_env = Go1Env(
        control_mode="position",
        enable_domain_rand=False,
        render_mode=None,
    )
    
    # Load model and VecNormalize if checkpoint provided
    model = None
    vec_env = None
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        
        # Create wrapped environment
        vec_env = DummyVecEnv([lambda: Go1Env(control_mode="position")])
        
        vecnorm_path = checkpoint_path.parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"Loaded VecNormalize: {vecnorm_path}")
        else:
            print("WARNING: VecNormalize not found, using raw observations")
            
        model = PPO.load(str(checkpoint_path))
        print(f"Loaded model: {checkpoint_path}")
    else:
        print("Running with RANDOM actions (no checkpoint)")
    
    # Reset both environments
    vis_obs, _ = vis_env.reset()
    vis_env.set_commands(target_velocity, 0.0, 0.0)
    
    if vec_env:
        vec_env.reset()
    
    # Launch viewer
    print("\n>>> MuJoCo viewer opened - watch the robot! <<<")
    print("Press ESC or close window to exit\n")
    
    velocities = []
    step = 0
    control_freq = 50.0
    max_steps = int(duration * control_freq)
    
    with mujoco.viewer.launch_passive(vis_env.model, vis_env.data) as viewer:
        while viewer.is_running() and step < max_steps:
            step_start = time.time()
            
            # Get action
            if model and vec_env:
                # Use VecNormalize to normalize observation
                norm_obs = vec_env.normalize_obs(vis_obs.reshape(1, -1))
                action, _ = model.predict(norm_obs, deterministic=True)
                action = action[0]
            elif model:
                action, _ = model.predict(vis_obs.reshape(1, -1), deterministic=True)
                action = action[0]
            else:
                action = vis_env.action_space.sample()
            
            # Step visualization environment
            vis_obs, reward, terminated, truncated, info = vis_env.step(action)
            
            # Track velocity
            if 'base_velocity' in info:
                vx = info['base_velocity'][0]
                velocities.append(vx)
                
                # Print progress every 2 seconds
                if step % 100 == 0:
                    avg_vel = np.mean(velocities[-50:]) if velocities else 0
                    print(f"Step {step:4d} | Vel: {vx:.2f} m/s | Avg: {avg_vel:.2f} | Reward: {reward:.2f}")
            
            step += 1
            
            # Check termination
            if terminated or truncated:
                print(f"\n>>> Episode ended at step {step} <<<")
                vis_obs, _ = vis_env.reset()
                vis_env.set_commands(target_velocity, 0.0, 0.0)
            
            # Sync viewer
            viewer.sync()
            
            # Real-time pacing
            elapsed = time.time() - step_start
            sleep_time = (1.0 / control_freq) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # Results
    actual_duration = step / control_freq
    mean_velocity = np.mean(velocities) if velocities else 0.0
    
    print(f"\n{'='*60}")
    print(f"TEST COMPLETE")
    print(f"Duration: {actual_duration:.2f}s")
    print(f"Mean velocity: {mean_velocity:.3f} m/s (target: {target_velocity})")
    print(f"{'='*60}")
    
    vis_env.close()
    if vec_env:
        vec_env.close()


def visual_push_test(
    checkpoint_path: str | None = None,
    push_force: float = 15.0,
    num_pushes: int = 3,
):
    """Run push recovery test with MuJoCo visualization."""
    print(f"\n{'='*60}")
    print(f"VISUAL TEST: Push Recovery")
    print(f"Push force: {push_force}N, {num_pushes} pushes")
    print(f"{'='*60}")
    
    # Create environment for visualization
    vis_env = Go1Env(
        control_mode="position",
        enable_domain_rand=False,
        render_mode=None,
    )
    
    # Load model and VecNormalize if checkpoint provided
    model = None
    vec_env = None
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        
        vec_env = DummyVecEnv([lambda: Go1Env(control_mode="position")])
        
        vecnorm_path = checkpoint_path.parent / "vecnormalize.pkl"
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"Loaded VecNormalize: {vecnorm_path}")
        else:
            print("WARNING: VecNormalize not found, using raw observations")
            
        model = PPO.load(str(checkpoint_path))
        print(f"Loaded model: {checkpoint_path}")
    else:
        print("Running with RANDOM actions (no checkpoint)")
    
    # Reset
    vis_obs, _ = vis_env.reset()
    vis_env.set_commands(0.0, 0.0, 0.0)  # Stand still
    
    if vec_env:
        vec_env.reset()
    
    print("\n>>> MuJoCo viewer opened - watch for pushes! <<<")
    print("Robot will be pushed laterally every few seconds\n")
    
    trunk_id = mujoco.mj_name2id(vis_env.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    
    control_freq = 50.0
    push_interval = 3.0  # seconds between pushes
    push_duration = 0.1  # seconds
    
    step = 0
    push_count = 0
    next_push_step = int(2.0 * control_freq)  # First push after 2s
    push_end_step = 0
    current_force = np.zeros(3)
    
    with mujoco.viewer.launch_passive(vis_env.model, vis_env.data) as viewer:
        while viewer.is_running() and push_count <= num_pushes:
            step_start = time.time()
            
            # Check if time to push
            if step == next_push_step and push_count < num_pushes:
                # Random lateral push
                angle = np.random.uniform(0, 2 * np.pi)
                current_force = np.array([
                    push_force * np.cos(angle),
                    push_force * np.sin(angle),
                    0.0
                ])
                push_end_step = step + int(push_duration * control_freq)
                push_count += 1
                print(f"\n>>> PUSH {push_count}! Force: [{current_force[0]:.1f}, {current_force[1]:.1f}] N <<<")
                next_push_step = step + int(push_interval * control_freq)
            
            # Apply or clear force
            if step < push_end_step:
                vis_env.data.xfrc_applied[trunk_id, 0:3] = current_force
            else:
                vis_env.data.xfrc_applied[trunk_id, 0:3] = 0
            
            # Get action with VecNormalize
            if model and vec_env:
                norm_obs = vec_env.normalize_obs(vis_obs.reshape(1, -1))
                action, _ = model.predict(norm_obs, deterministic=True)
                action = action[0]
            elif model:
                action, _ = model.predict(vis_obs.reshape(1, -1), deterministic=True)
                action = action[0]
            else:
                action = vis_env.action_space.sample()
            
            # Step
            vis_obs, reward, terminated, truncated, info = vis_env.step(action)
            step += 1
            
            # Check if fell
            if terminated or truncated:
                if push_count > 0:
                    print(f"   Robot FELL after push {push_count}!")
                vis_obs, _ = vis_env.reset()
                vis_env.set_commands(0.0, 0.0, 0.0)
            
            # Sync viewer
            viewer.sync()
            
            # Real-time pacing
            elapsed = time.time() - step_start
            sleep_time = (1.0 / control_freq) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print(f"\n{'='*60}")
    print(f"PUSH TEST COMPLETE - {num_pushes} pushes applied")
    print(f"{'='*60}")
    
    vis_env.close()
    if vec_env:
        vec_env.close()


def main():
    parser = argparse.ArgumentParser(description="Visual evaluation of Go1 policy")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model (optional, uses random if not provided)")
    parser.add_argument("--test", type=str, choices=["velocity", "push", "both"],
                        default="both", help="Which test to run")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Duration for velocity test (seconds)")
    parser.add_argument("--push-force", type=float, default=15.0,
                        help="Push force in Newtons")
    parser.add_argument("--num-pushes", type=int, default=3,
                        help="Number of pushes")
    
    args = parser.parse_args()
    
    if args.test in ["velocity", "both"]:
        visual_velocity_test(
            checkpoint_path=args.checkpoint,
            duration=args.duration,
        )
    
    if args.test in ["push", "both"]:
        visual_push_test(
            checkpoint_path=args.checkpoint,
            push_force=args.push_force,
            num_pushes=args.num_pushes,
        )


if __name__ == "__main__":
    main()
