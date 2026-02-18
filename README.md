# 🤖 RobustWalker

**Blind Locomotion for Unitree Go1 using Deep Reinforcement Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0+-green.svg)](https://mujoco.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

RobustWalker trains a **PPO-based RL policy** to control the Unitree Go1 quadruped robot using **only proprioceptive sensing** (no cameras or LiDAR). The robot learns to walk robustly on rough terrain and recover from external disturbances through **domain randomization** during training.

### 🎥 Training Demo

[![Training Demo](https://img.shields.io/badge/▶️_Watch_Training_Video-Google_Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1CoJNGUmFYeM_CfP4g9liCz5-LFeYYQjU/view?usp=sharing)

> **[Click here to watch the trained policy in action →](https://drive.google.com/file/d/1CoJNGUmFYeM_CfP4g9liCz5-LFeYYQjU/view?usp=sharing)**

### Key Features

- 🏃 **Blind Locomotion**: Walks using only joint encoders and IMU—no vision required
- 🌍 **Domain Randomization**: Randomizes friction, payload, motor strength, and external pushes for sim-to-real transfer
- ⚡ **Parallel Training**: Vectorized environments for fast training with Stable-Baselines3
- 📊 **Comprehensive Rewards**: Multi-objective reward function balancing speed, efficiency, and stability

---

## 🏗️ Architecture

### Policy Network

```
┌─────────────────────────────────────────────────────────────┐
│                    Observation (57-dim)                      │
├─────────────────────────────────────────────────────────────┤
│  Joint Positions (12) │ Joint Velocities (12) │ IMU (6)     │
│  Velocity Commands (3) │ Action History (24)                │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MLP [256, 256]                           │
│                    Activation: ELU                           │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Joint Position Targets (12)                  │
│              (PD Controller → Torques → Robot)              │
└─────────────────────────────────────────────────────────────┘
```

### Observation Space (57 dimensions)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Joint Positions | 12 | Normalized encoder readings for all leg joints |
| Joint Velocities | 12 | Scaled velocity measurements |
| Base Angular Velocity | 3 | IMU gyroscope in body frame |
| Projected Gravity | 3 | Gravity vector in body frame (detects tilt) |
| Velocity Commands | 3 | Target (vx, vy, ωz) for command tracking |
| Action History | 24 | Last 2 actions for temporal context |

### Action Space (12 dimensions)

Joint position targets for all 12 actuators:
- **FR** (Front Right): hip, thigh, calf
- **FL** (Front Left): hip, thigh, calf  
- **RR** (Rear Right): hip, thigh, calf
- **RL** (Rear Left): hip, thigh, calf

---

## 🎯 Reward Function

The reward function balances multiple objectives:

```python
reward = velocity_tracking + alive_bonus 
       - torque_penalty - action_rate_penalty
       - stumble_penalty - orientation_penalty - termination_penalty
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Velocity Tracking** | 1.0 | Gaussian reward for matching commanded velocity |
| **Alive Bonus** | 0.1 | Small reward for each timestep survived |
| **Torque Penalty** | 0.001 | Minimizes energy consumption |
| **Action Rate Penalty** | 0.1 | Encourages smooth joint motion |
| **Stumble Penalty** | 2.0 | Penalizes body-ground contact |
| **Orientation Penalty** | 0.5 | Keeps robot upright |
| **Termination Penalty** | 5.0 | Large penalty for falling |

---

## 🔀 Domain Randomization

For robust sim-to-real transfer, we randomize:

| Parameter | Range | Applied When |
|-----------|-------|--------------|
| **Ground Friction** | [0.5, 1.2] | Per episode |
| **Payload Mass** | [0, 4] kg | Per episode |
| **Motor Strength** | [0.9, 1.1]× | Per episode |
| **External Pushes** | [0, 15] N | Every 5-10 seconds |

---

## 📁 Project Structure

```
ROBUSTWALKER/
├── assets/go1/              # MuJoCo model files (URDF → MJCF)
│   ├── go1.xml              # Robot definition
│   └── scene.xml            # Scene with ground plane & lighting
├── robustwalker/            # Core Python package
│   ├── envs/
│   │   ├── go1_env.py       # Gymnasium environment
│   │   └── domain_rand.py   # Domain randomization
│   ├── rewards/
│   │   └── locomotion.py    # Multi-objective reward function
│   └── utils/
│       └── mujoco_utils.py  # MuJoCo helper functions
├── scripts/
│   ├── train.py             # PPO training script
│   ├── evaluate.py          # Policy evaluation
│   └── visualize.py         # Render trained policy
├── configs/
│   └── default.yaml         # Hyperparameters
└── tests/
    └── test_env.py          # Environment unit tests
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/shrirag10/ROBUSTWALKER.git
cd ROBUSTWALKER

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default config (~2M steps, ~2-3 hours on GPU)
python scripts/train.py

# Custom training
python scripts/train.py --timesteps 5000000 --n-envs 16

# Quick test (10k steps)
python scripts/train.py --test-mode
```

### Evaluation & Visualization

```bash
# Evaluate trained policy
python scripts/evaluate.py --checkpoint logs/best_model.zip

# Visualize in MuJoCo viewer
python scripts/visualize.py --checkpoint logs/best_model.zip

# Record video
python scripts/visualize.py --checkpoint logs/best_model.zip --record
```

---

## ⚙️ Configuration

Edit `configs/default.yaml` to customize training:

```yaml
# Training
total_timesteps: 2_000_000
n_envs: 8                    # Parallel environments
learning_rate: 3.0e-4
batch_size: 64

# Policy Network
policy_kwargs:
  net_arch:
    pi: [256, 256]
    vf: [256, 256]
  activation_fn: elu

# Domain Randomization
domain_rand:
  friction_range: [0.5, 1.2]
  payload_range: [0.0, 4.0]
  push_force_range: [0.0, 15.0]
```

---

## 📈 Training Progress

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

Key metrics to track:
- `rollout/ep_rew_mean` - Average episode reward
- `train/loss` - Policy loss
- `rollout/ep_len_mean` - Episode length (longer = more stable)

---

## 🎯 Acceptance Criteria

- [ ] Track 0.8 m/s forward velocity on rough terrain for >15s
- [ ] Recover from 15N lateral push without falling
- [ ] Maintain stable trot gait pattern

---

## 🔬 Technical Details

### MuJoCo Simulation

- **Physics timestep**: 2ms (500 Hz)
- **Control frequency**: 50 Hz (20ms per action)
- **Episode length**: 1000 steps (20 seconds)

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Rollout buffer | 2048 steps/env |
| Minibatch size | 64 |
| Epochs per update | 10 |
| Discount (γ) | 0.99 |
| GAE (λ) | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |

---

## 📚 References

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978) - Rudin et al., 2022
- [Unitree Go1 Documentation](https://www.unitree.com/products/go1)
- [MuJoCo Physics Engine](https://mujoco.org/)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Srinivasan Shriram**  
[GitHub](https://github.com/shrirag10)
