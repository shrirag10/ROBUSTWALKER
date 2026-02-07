# RobustWalker: Blind Locomotion for Unitree Go1

Train a PPO policy in MuJoCo for quadruped locomotion with domain randomization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python scripts/train.py

# Evaluate
python scripts/evaluate.py --checkpoint logs/best_model.zip

# Visualize
python scripts/visualize.py --checkpoint logs/best_model.zip
```

## Project Structure

```
├── assets/go1/          # MuJoCo model
├── robustwalker/        # Core package
│   ├── envs/            # Gymnasium environment
│   ├── rewards/         # Reward functions
│   └── utils/           # Helpers
├── scripts/             # Training & evaluation
└── configs/             # Hyperparameters
```

## Acceptance Criteria

- [ ] Track 0.8 m/s on rough terrain for >15s
- [ ] Recover from 15N lateral push
