#!/usr/bin/env python3
"""
Main Experiment for MARL-IoTP Paper (Table 1)

Runs the primary MAPPO training with default configuration.
Supports: Table 1 - Main Performance Comparison

Usage:
    python experiments/exp_main.py --episodes 5000 --seed 42
    CUDA_VISIBLE_DEVICES=0 python experiments/exp_main.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train import main as train_main


def parse_args():
    parser = argparse.ArgumentParser(description='MARL-IoTP Main Experiment (Table 1)')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_devices', type=int, default=20,
                        help='Number of IoT devices')
    parser.add_argument('--num_servers', type=int, default=3,
                        help='Number of edge servers')
    parser.add_argument('--save_dir', type=str, default='results/raw/main_experiment',
                        help='Directory to save results')
    parser.add_argument('--config', type=str, default='configs/mappo.yaml',
                        help='Configuration file path')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MARL-IoTP Main Experiment")
    print("Supports: Table 1 - Main Performance Comparison")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print(f"Devices: {args.num_devices}")
    print(f"Servers: {args.num_servers}")
    print("=" * 60)

    # Build command line args for train.py
    sys.argv = [
        'train.py',
        '--config', args.config,
        '--episodes', str(args.episodes),
        '--seed', str(args.seed),
        '--num_devices', str(args.num_devices),
        '--num_servers', str(args.num_servers),
        '--save_dir', args.save_dir,
        '--experiment_name', 'main_experiment'
    ]

    # Run training
    train_main()

    print("\n" + "=" * 60)
    print("Main Experiment Complete")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
