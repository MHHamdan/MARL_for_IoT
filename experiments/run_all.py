#!/usr/bin/env python3
"""
Run All Experiments for MARL-IoTP Research Paper

This script runs all experiments needed for the ICPRAI 2026 paper:
1. Baseline MAPPO training
2. MADDPG baseline comparison
3. Ablation studies (communication, attention, architecture, reward weights)
4. Scalability experiments

Usage:
    python experiments/run_all_experiments.py --quick   # Quick test (100 episodes)
    python experiments/run_all_experiments.py --full    # Full experiments (5000 episodes)
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='Run all MARL-IoTP experiments')
    parser.add_argument(
        '--mode', type=str, default='quick',
        choices=['quick', 'medium', 'full'],
        help='Experiment mode: quick (100 ep), medium (1000 ep), full (5000 ep)'
    )
    parser.add_argument(
        '--experiments', type=str, nargs='+',
        default=['mappo', 'maddpg', 'ablation', 'scalability'],
        help='Which experiments to run'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/paper_experiments')
    return parser.parse_args()


def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"WARNING: {description} returned non-zero exit code: {result.returncode}")
        return False
    return True


def run_mappo_baseline(episodes, save_dir, seed):
    """Run MAPPO baseline experiment."""
    cmd = [
        'python', 'scripts/train.py',
        '--config', 'configs/mappo.yaml',
        '--episodes', str(episodes),
        '--seed', str(seed),
        '--experiment_name', 'mappo_baseline'
    ]
    return run_command(cmd, "MAPPO Baseline Training")


def run_maddpg_baseline(episodes, save_dir, seed):
    """Run MADDPG baseline experiment."""
    cmd = [
        'python', 'scripts/run_baselines.py',
        '--algorithm', 'maddpg',
        '--episodes', str(episodes),
        '--seed', str(seed)
    ]
    return run_command(cmd, "MADDPG Baseline Training")


def run_ablation_studies(episodes, eval_episodes, save_dir, seed):
    """Run all ablation studies."""
    cmd = [
        'python', 'experiments/exp_ablation.py',
        '--ablation', 'all',
        '--episodes', str(episodes),
        '--eval_episodes', str(eval_episodes),
        '--seed', str(seed),
        '--save_dir', str(save_dir / 'ablation')
    ]
    return run_command(cmd, "Ablation Studies")


def run_scalability_experiments(episodes, save_dir, seed):
    """Run scalability experiments."""
    cmd = [
        'python', 'experiments/exp_scalability.py',
        '--episodes', str(episodes),
        '--seed', str(seed),
        '--save_dir', str(save_dir / 'scalability')
    ]
    return run_command(cmd, "Scalability Experiments")


def generate_summary_report(save_dir):
    """Generate a summary report of all experiments."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {}
    }

    # Collect results from all experiments
    results_dirs = list(save_dir.glob('**/'))

    for result_dir in results_dirs:
        json_files = list(result_dir.glob('*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    report['experiments'][json_file.stem] = data
            except Exception as e:
                print(f"Could not read {json_file}: {e}")

    # Save summary
    summary_path = save_dir / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    return report


def main():
    args = parse_args()

    # Set episode counts based on mode
    episode_counts = {
        'quick': {'train': 100, 'eval': 10},
        'medium': {'train': 1000, 'eval': 50},
        'full': {'train': 5000, 'eval': 100}
    }

    episodes = episode_counts[args.mode]['train']
    eval_episodes = episode_counts[args.mode]['eval']

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"{args.mode}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# MARL-IoTP Complete Experiment Suite")
    print(f"# Mode: {args.mode}")
    print(f"# Episodes: {episodes}")
    print(f"# Experiments: {args.experiments}")
    print(f"# Save Directory: {save_dir}")
    print(f"{'#'*60}")

    # Change to project root
    os.chdir(project_root)

    results = {}

    # Run selected experiments
    if 'mappo' in args.experiments:
        print("\n" + "="*60)
        print("EXPERIMENT 1: MAPPO Baseline")
        print("="*60)
        success = run_mappo_baseline(episodes, save_dir, args.seed)
        results['mappo'] = 'success' if success else 'failed'

    if 'maddpg' in args.experiments:
        print("\n" + "="*60)
        print("EXPERIMENT 2: MADDPG Baseline")
        print("="*60)
        success = run_maddpg_baseline(episodes, save_dir, args.seed)
        results['maddpg'] = 'success' if success else 'failed'

    if 'ablation' in args.experiments:
        print("\n" + "="*60)
        print("EXPERIMENT 3: Ablation Studies")
        print("="*60)
        success = run_ablation_studies(episodes, eval_episodes, save_dir, args.seed)
        results['ablation'] = 'success' if success else 'failed'

    if 'scalability' in args.experiments:
        print("\n" + "="*60)
        print("EXPERIMENT 4: Scalability Experiments")
        print("="*60)
        success = run_scalability_experiments(episodes, save_dir, args.seed)
        results['scalability'] = 'success' if success else 'failed'

    # Generate summary report
    print("\n" + "="*60)
    print("Generating Summary Report...")
    print("="*60)
    generate_summary_report(save_dir)

    # Print final status
    print(f"\n{'#'*60}")
    print("# EXPERIMENT SUITE COMPLETE")
    print(f"{'#'*60}")
    for exp, status in results.items():
        status_icon = "✓" if status == 'success' else "✗"
        print(f"  {status_icon} {exp}: {status}")
    print(f"\nResults saved to: {save_dir}")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
