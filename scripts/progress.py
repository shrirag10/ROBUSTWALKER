#!/usr/bin/env python3
"""Live progress bar for training monitoring."""

import sys
import time
import re
from pathlib import Path

def get_latest_progress(log_path: str) -> tuple[int, int, float]:
    """Parse log file for latest progress."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Find total_timesteps
        matches = re.findall(r'total_timesteps[|\s]+(\d+)', content)
        current = int(matches[-1]) if matches else 0
        
        # Find ep_rew_mean
        rew_matches = re.findall(r'ep_rew_mean[|\s]+(-?[\d.]+)', content)
        reward = float(rew_matches[-1]) if rew_matches else 0.0
        
        # Get target from start of log
        target_match = re.search(r'Timesteps:\s+([\d,]+)', content)
        target = int(target_match.group(1).replace(',', '')) if target_match else 1000000
        
        return current, target, reward
    except:
        return 0, 1000000, 0.0

def draw_progress_bar(current: int, target: int, reward: float, width: int = 40):
    """Draw ASCII progress bar."""
    pct = min(current / target, 1.0)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    
    print(f"\r[{bar}] {pct*100:5.1f}% | {current:,}/{target:,} | Reward: {reward:+.1f}   ", end='', flush=True)

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "training_1m.log"
    
    print(f"📊 Live Training Progress - {log_path}")
    print("=" * 60)
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            current, target, reward = get_latest_progress(log_path)
            draw_progress_bar(current, target, reward)
            
            if current >= target:
                print("\n\n✅ Training Complete!")
                break
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == "__main__":
    main()
