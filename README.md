# RobustWalker

Deep reinforcement learning for blind locomotion on the Unitree Go1 quadruped. Trains a PPO policy using proprioceptive sensing only (joint encoders + IMU) — no cameras, no lidar. Supports both MuJoCo (CPU) and Genesis (GPU) backends, with Genesis delivering ~85x throughput gains.

## Key Results

| Metric              | MuJoCo         | Genesis              |
|---------------------|----------------|----------------------|
| Total reward        | 30.83 ± 3.59   | **51.20**            |
| Episode length      | 11 steps       | **1001 steps (20s)** |
| Best velocity       | 0.32 m/s       | **1.41 m/s**         |
| Training throughput | ~2K steps/s    | **169K steps/s**     |

## Tech Stack

- **Simulation**: MuJoCo, Genesis
- **RL**: PPO via custom training loop
- **Robot**: Unitree Go1 (MuJoCo MJCF model)
- **Python**: 3.10+

## Architecture

```
robustwalker/
├── envs/
│   ├── go1_env.py          # Core Gymnasium env (57-dim obs, 12-dim action)
│   ├── genesis_env.py      # Genesis-native parallel env
│   ├── domain_rand.py      # Friction, payload, motor randomization
│   └── terrain.py          # Terrain generation
├── rewards/
│   └── locomotion.py       # Velocity tracking, energy, symmetry, survival
└── utils/
    ├── mujoco_utils.py
    └── genesis_utils.py
scripts/
├── train.py                # MuJoCo training
├── genesis_train.py        # Genesis GPU training (recommended)
├── evaluate.py
└── visualize.py
```

## Observation & Action Space

- **Observations (57-dim)**: Joint positions (12), joint velocities (12), base angular velocity (3), projected gravity (3), velocity commands (3), action history (24)
- **Actions (12-dim)**: Target joint positions (position control mode)

## Critical Design Note: Trot Symmetry

Naive left-right symmetry (FR=FL, RR=RL) prevents walking by locking both sides to identical positions. The reward uses **diagonal trot symmetry** instead — penalizing differences between diagonal pairs (FR<->RL, FL<->RR) — which produces natural quadruped gait.

## Setup

```bash
git clone https://github.com/shrirag10/ROBUSTWALKER.git
cd ROBUSTWALKER
pip install -r requirements.txt
```

### Train (Genesis — recommended)

```bash
python scripts/genesis_train.py -e go1-walking -B 4096
```

### Train (MuJoCo)

```bash
python scripts/train.py
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint <path>
```

### Visualize

```bash
python scripts/visualize.py
```

## License

MIT
