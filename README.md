# MARL-IoTP: Hierarchical Multi-Agent Reinforcement Learning for Joint Perception Model Selection and Resource Orchestration in IoT Edge Networks

A comprehensive Multi-Agent Reinforcement Learning framework for intelligent perception and resource orchestration in Edge-IoT networks.

## Architecture

![MARL-IoTP System Architecture](docs/system_architecture.png)

The framework employs the **Centralized Training with Decentralized Execution (CTDE)** paradigm with:
- **Centralized Critic**: Receives global observations during training for stable value estimation
- **Multi-Head Attention**: Aggregates messages from perception agents for informed orchestration decisions
- **Heterogeneous Agents**: Perception agents (IoT devices) and Orchestration agents (edge servers)

## Overview

MARL-IoTP addresses the challenge of joint optimization of perception model selection and resource allocation in IoT edge computing environments. The framework achieves:

- **91.7% Classification Accuracy** with 101ms average latency
- **44% Improvement** in cumulative reward over Independent PPO
- **87% Reduction** in deadline violations compared to heuristic baselines
- **Near-linear Scalability** from 10 to 50 IoT devices

### Key Innovations

1. **Heterogeneous Agent Architecture**: Specialized agents reflecting the natural IoT-edge hierarchy
2. **Learned Communication Protocol**: Differentiable message encoding for cross-boundary coordination
3. **Attention-Based Message Aggregation**: Dynamic weighting of device information based on server state

## Project Structure

```
MARL_for_IoT/
├── configs/           # Configuration files (YAML)
│   ├── mappo.yaml     # MAPPO training configuration
│   ├── maddpg.yaml    # MADDPG baseline configuration
│   └── default.yaml   # Default environment settings
├── envs/              # Environment implementations
│   ├── iot_env.py     # Main IoT Edge environment
│   ├── edge_server.py # Edge server simulation
│   ├── network_model.py # Wireless channel model
│   └── perception_task.py # Task generation
├── agents/            # Agent implementations
│   ├── perception_agent.py    # Perception agent (model selection, frame rate)
│   └── orchestration_agent.py # Orchestration agent (offloading, resource allocation)
├── algorithms/        # RL algorithm implementations
│   ├── mappo.py       # Multi-Agent PPO
│   ├── maddpg.py      # Multi-Agent DDPG
│   └── buffer.py      # Experience replay buffer
├── utils/             # Utility functions
│   ├── logger.py      # Training logger
│   ├── metrics.py     # Evaluation metrics
│   └── visualization.py # Plotting utilities
├── scripts/           # Training and evaluation scripts
│   ├── train.py       # Main training script
│   ├── evaluate.py    # Evaluation script
│   └── run_baselines.py # Baseline comparisons
├── experiments/       # Experiment scripts
│   ├── exp_scalability.py # Scalability experiments
│   ├── exp_ablation.py    # Ablation studies
│   └── run_all_experiments.py # Full experiment suite
├── docs/              # Documentation and figures
└── requirements.txt   # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/MHHamdan/MARL_for_IoT.git
cd MARL_for_IoT

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, Gymnasium, PyYAML
- Matplotlib, Seaborn (for visualization)

## Quick Start

### Training

```bash
# Train with default MAPPO settings
python scripts/train.py --config configs/mappo.yaml

# Train with custom settings
python scripts/train.py --config configs/mappo.yaml \
    --episodes 5000 \
    --num_devices 20 \
    --num_servers 3 \
    --seed 42

# Train with GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/mappo.yaml
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint results/mappo_best.pt

# Run baseline comparisons
python scripts/run_baselines.py --algorithms random greedy ippo maddpg
```

### Experiments

```bash
# Scalability experiments (10-50 devices)
python experiments/exp_scalability.py --device_counts 10 20 30 50

# Ablation studies
python experiments/exp_ablation.py --ablation communication
python experiments/exp_ablation.py --ablation attention
python experiments/exp_ablation.py --ablation reward_weights

# Run all experiments
python experiments/run_all_experiments.py --mode full
```

## Environment Details

### IoT Devices
- CPU: 1.5 GHz ARM (Raspberry Pi 4 / Jetson Nano class)
- Battery: 5000 mAh
- Transmit Power: 20 dBm
- Task Queue: 10 tasks maximum

### Perception Models
| Model | Accuracy | Latency Factor | MFLOPS |
|-------|----------|----------------|--------|
| MobileNet-Tiny | 0.65 | 0.3 | 50 |
| MobileNet-Small | 0.75 | 0.5 | 150 |
| MobileNet-Large | 0.85 | 0.8 | 300 |
| ResNet-Edge | 0.92 | 1.0 | 500 |

### Edge Servers
- CPU: 3.0 GHz (8 cores)
- Memory: 16 GB RAM
- Max Concurrent Tasks: 10

### Wireless Channel
- Bandwidth: 20 MHz
- Path Loss Exponent: 3.5 (urban environment)
- Rayleigh fading model

## Results

### Performance Comparison

| Method | Reward | Latency (ms) | Accuracy | Violations |
|--------|--------|--------------|----------|------------|
| Random | -1500.2 | 210.5 | 35.1% | 45.8 |
| Greedy | -950.4 | 168.2 | 55.8% | 28.4 |
| IPPO | -680.5 | 142.8 | 72.3% | 18.2 |
| MADDPG | -520.3 | 125.4 | 78.5% | 12.8 |
| **MARL-IoTP** | **-379.2** | **101.1** | **91.7%** | **4.8** |

### Ablation Results

- **Communication**: 61% of performance gains from inter-agent communication
- **Attention**: 22% improvement from multi-head attention over mean pooling
- **Scalability**: Maintains 78.5% accuracy at 50 devices (5x scale)

## Configuration

Key hyperparameters in `configs/mappo.yaml`:

```yaml
# Training
gamma: 0.99           # Discount factor
gae_lambda: 0.95      # GAE parameter
clip_epsilon: 0.2     # PPO clipping
lr: 0.0003            # Learning rate
ppo_epochs: 10        # PPO update epochs

# Architecture
hidden_dim: 128       # Perception agent hidden dimension
critic_hidden_dim: 256 # Critic hidden dimension
message_dim: 8        # Inter-agent message dimension
attention_heads: 2    # Multi-head attention heads

# Reward weights
latency: 0.4          # Latency penalty weight
energy: 0.3           # Energy penalty weight
accuracy: 0.3         # Accuracy reward weight
```

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{hamdan2026marliotp,
  title={MARL-IoTP: Hierarchical Multi-Agent Reinforcement Learning for Joint Perception Model Selection and Resource Orchestration in IoT Edge Networks},
  author={Hamdan, Mohammad H. and [Co-authors]},
  booktitle={International Conference on Pattern Recognition and Artificial Intelligence (ICPRAI)},
  year={2026}
}
```

## References

- [MAPPO](https://arxiv.org/abs/2103.01955) - Multi-Agent PPO
- [MobileNets](https://arxiv.org/abs/1704.04861) - Efficient CNNs for Mobile
- [MADDPG](https://arxiv.org/abs/1706.02275) - Multi-Agent DDPG

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
