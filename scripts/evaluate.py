#!/usr/bin/env python3
"""
Evaluation script for Go1 locomotion policy.

Tests acceptance criteria:
1. Track 0.8 m/s on rough terrain for >15s
2. Recover from 15N lateral push

Usage:
    python scripts/evaluate.py --checkpoint logs/best_model/model.zip
    python scripts/evaluate.py --checkpoint logs/best_model/model.zip --test-velocity --test-push
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mujoco

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robustwalker.envs.go1_env import Go1Env
from robustwalker.envs.domain_rand import DomainRandomizationConfig


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    test_name: str
    passed: bool
    duration: float
    mean_velocity: float
    details: dict


def create_env(enable_domain_rand: bool = False) -> tuple:
    """Create environment for evaluation."""
    env = Go1Env(
        control_mode="position",
        enable_domain_rand=enable_domain_rand,
        render_mode=None,
    )
    vec_env = DummyVecEnv([lambda: env])
    return vec_env, env


def load_model(checkpoint_path: str, vec_env) -> tuple[PPO, VecNormalize | None]:
    """Load trained model and optional VecNormalize."""
    checkpoint_path = Path(checkpoint_path)
    
    # Try to load VecNormalize stats
    vecnorm_path = checkpoint_path.parent / "vecnormalize.pkl"
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load model
    model = PPO.load(str(checkpoint_path), env=vec_env)
    
    return model, vec_env


def test_velocity_tracking(
    model: PPO,
    env: VecNormalize,
    target_velocity: float = 0.8,
    duration: float = 15.0,
    tolerance: float = 0.1,
) -> EvaluationResult:
    """
    Test 1: Track target velocity on rough terrain for specified duration.
    
    Acceptance criterion: Robot tracks 0.8 m/s on rough terrain for >15s.
    """
    print(f"\n{'='*60}")
    print(f"Test: Velocity Tracking")
    print(f"Target: {target_velocity} m/s for {duration}s")
    print(f"{'='*60}")
    
    control_freq = 50.0
    max_steps = int(duration * control_freq * 1.5)  # 50% extra margin
    
    obs = env.reset()
    
    # Set velocity command
    raw_env = env.envs[0]
    if hasattr(raw_env, 'env'):
        raw_env = raw_env.env
    raw_env.set_commands(target_velocity, 0.0, 0.0)
    
    velocities = []
    step = 0
    
    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Get actual velocity
        if 'base_velocity' in info[0]:
            vx = info[0]['base_velocity'][0]
            velocities.append(vx)
        
        step += 1
        
        if done[0]:
            print(f"Episode terminated at step {step}")
            break
    
    # Analyze results
    actual_duration = step / control_freq
    mean_velocity = np.mean(velocities) if velocities else 0.0
    std_velocity = np.std(velocities) if velocities else 0.0
    
    # Pass if maintained velocity for required duration
    passed = actual_duration >= duration and abs(mean_velocity - target_velocity) < tolerance
    
    result = EvaluationResult(
        test_name="Velocity Tracking",
        passed=passed,
        duration=actual_duration,
        mean_velocity=mean_velocity,
        details={
            'target_velocity': target_velocity,
            'std_velocity': std_velocity,
            'min_velocity': min(velocities) if velocities else 0,
            'max_velocity': max(velocities) if velocities else 0,
            'total_steps': step,
        }
    )
    
    print(f"\nResults:")
    print(f"  Duration: {actual_duration:.2f}s (required: {duration}s)")
    print(f"  Mean velocity: {mean_velocity:.3f} m/s (target: {target_velocity})")
    print(f"  Velocity std: {std_velocity:.3f}")
    print(f"  PASSED: {passed}")
    
    return result


def test_push_recovery(
    model: PPO,
    env: VecNormalize,
    push_force: float = 15.0,
    recovery_time: float = 2.0,
    num_pushes: int = 5,
) -> EvaluationResult:
    """
    Test 2: Recover from lateral push.
    
    Acceptance criterion: Robot recovers from 15N lateral push.
    """
    print(f"\n{'='*60}")
    print(f"Test: Push Recovery")
    print(f"Push force: {push_force}N, {num_pushes} pushes")
    print(f"{'='*60}")
    
    control_freq = 50.0
    push_duration = 0.1  # 100ms push
    recovery_steps = int(recovery_time * control_freq)
    push_steps = int(push_duration * control_freq)
    
    recoveries = []
    
    for push_idx in range(num_pushes):
        obs = env.reset()
        
        # Get raw environment for force application
        raw_env = env.envs[0]
        if hasattr(raw_env, 'env'):
            raw_env = raw_env.env
        
        # Let robot stabilize first (2 seconds)
        for _ in range(int(2.0 * control_freq)):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done[0]:
                break
        
        if done[0]:
            recoveries.append(False)
            print(f"  Push {push_idx+1}: Failed (fell during stabilization)")
            continue
        
        # Apply push in random lateral direction
        angle = np.random.uniform(0, 2 * np.pi)
        force = np.array([
            push_force * np.cos(angle),
            push_force * np.sin(angle),
            0.0
        ])
        
        # Apply force for push duration
        trunk_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        for _ in range(push_steps):
            # Apply force directly to trunk body
            raw_env.data.xfrc_applied[trunk_id, 0:3] = force
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done[0]:
                break
        
        # Clear force
        raw_env.data.xfrc_applied[trunk_id, 0:3] = 0
        
        if done[0]:
            recoveries.append(False)
            print(f"  Push {push_idx+1}: Failed (fell during push)")
            continue
        
        # Check if robot recovers
        recovered = True
        for step in range(recovery_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            if done[0]:
                recovered = False
                break
        
        recoveries.append(recovered)
        status = "Recovered" if recovered else "Failed"
        print(f"  Push {push_idx+1}: {status}")
    
    # Analyze results
    recovery_rate = sum(recoveries) / len(recoveries)
    passed = recovery_rate >= 0.8  # 80% recovery rate threshold
    
    result = EvaluationResult(
        test_name="Push Recovery",
        passed=passed,
        duration=0.0,  # N/A
        mean_velocity=0.0,  # N/A
        details={
            'push_force': push_force,
            'num_pushes': num_pushes,
            'num_recovered': sum(recoveries),
            'recovery_rate': recovery_rate,
        }
    )
    
    print(f"\nResults:")
    print(f"  Recovery rate: {recovery_rate*100:.1f}% ({sum(recoveries)}/{len(recoveries)})")
    print(f"  PASSED: {passed}")
    
    return result


def run_general_evaluation(
    model: PPO,
    env: VecNormalize,
    n_episodes: int = 10,
) -> dict:
    """Run general evaluation episodes."""
    print(f"\n{'='*60}")
    print(f"General Evaluation ({n_episodes} episodes)")
    print(f"{'='*60}")
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = done[0]
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"  Episode {ep+1}: reward={total_reward:.2f}, length={steps}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
    }
    
    print(f"\nSummary:")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean length: {results['mean_length']:.1f} steps")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Go1 locomotion policy")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--test-velocity",
        action="store_true",
        help="Run velocity tracking test",
    )
    parser.add_argument(
        "--test-push",
        action="store_true",
        help="Run push recovery test",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of general evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    # Create environment
    vec_env, raw_env = create_env(enable_domain_rand=False)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, vec_env = load_model(args.checkpoint, vec_env)
    
    results = []
    
    # Run tests
    if args.test_velocity or (not args.test_velocity and not args.test_push):
        result = test_velocity_tracking(model, vec_env)
        results.append(result)
    
    if args.test_push or (not args.test_velocity and not args.test_push):
        result = test_push_recovery(model, vec_env)
        results.append(result)
    
    # General evaluation
    general_results = run_general_evaluation(model, vec_env, args.n_episodes)
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for result in results:
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"  {result.test_name}: {status}")
        all_passed = all_passed and result.passed
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")
    
    # Cleanup
    vec_env.close()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
