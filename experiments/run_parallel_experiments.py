#!/usr/bin/env python3
"""
Run Experiments in Parallel on Multiple GPUs

This script runs all experiments needed for the ICPRAI 2026 paper in parallel,
utilizing all available GPUs:
- GPU 0: MAPPO baseline training
- GPU 1: MADDPG baseline training
- GPU 2: Ablation studies
- GPU 3: Scalability experiments

Usage:
    python experiments/run_parallel_experiments.py --mode full
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='Run MARL-IoTP experiments in parallel on multiple GPUs')
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=['quick', 'medium', 'full'],
        help='Experiment mode: quick (100 ep), medium (1000 ep), full (5000 ep)'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/paper_experiments')
    return parser.parse_args()


def run_experiment_on_gpu(gpu_id, cmd, name, log_file):
    """Run an experiment on a specific GPU."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"\n{'='*60}")
    print(f"Starting {name} on GPU {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(project_root)
        )

    return process


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
    save_dir = Path(args.save_dir) / f"parallel_{args.mode}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# MARL-IoTP Parallel Experiment Suite")
    print(f"# Mode: {args.mode}")
    print(f"# Episodes: {episodes}")
    print(f"# Using 4 GPUs in parallel")
    print(f"# Save Directory: {save_dir}")
    print(f"{'#'*60}")

    # Define experiments for each GPU
    experiments = [
        {
            'gpu': 0,
            'name': 'MAPPO Baseline',
            'cmd': [
                'python', 'scripts/train.py',
                '--config', 'configs/mappo.yaml',
                '--episodes', str(episodes),
                '--seed', str(args.seed),
                '--experiment_name', 'mappo_baseline_gpu0'
            ],
            'log': save_dir / 'mappo_baseline.log'
        },
        {
            'gpu': 1,
            'name': 'MADDPG Baseline',
            'cmd': [
                'python', 'scripts/run_baselines.py',
                '--algorithm', 'maddpg',
                '--episodes', str(episodes),
                '--seed', str(args.seed)
            ],
            'log': save_dir / 'maddpg_baseline.log'
        },
        {
            'gpu': 2,
            'name': 'Ablation Studies',
            'cmd': [
                'python', 'experiments/exp_ablation.py',
                '--ablation', 'all',
                '--episodes', str(episodes),
                '--eval_episodes', str(eval_episodes),
                '--seed', str(args.seed),
                '--save_dir', str(save_dir / 'ablation')
            ],
            'log': save_dir / 'ablation_studies.log'
        },
        {
            'gpu': 3,
            'name': 'Scalability Experiments',
            'cmd': [
                'python', 'experiments/exp_scalability.py',
                '--episodes', str(episodes),
                '--seed', str(args.seed),
                '--save_dir', str(save_dir / 'scalability')
            ],
            'log': save_dir / 'scalability_experiments.log'
        }
    ]

    # Launch all experiments in parallel
    processes = {}
    for exp in experiments:
        proc = run_experiment_on_gpu(
            exp['gpu'],
            exp['cmd'],
            exp['name'],
            str(exp['log'])
        )
        processes[exp['name']] = {
            'process': proc,
            'gpu': exp['gpu'],
            'log': exp['log'],
            'start_time': datetime.now()
        }

    print(f"\n{'='*60}")
    print("All experiments launched! Monitoring progress...")
    print(f"{'='*60}")

    # Monitor progress
    completed = set()
    while len(completed) < len(processes):
        time.sleep(60)  # Check every minute

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update:")
        print("-" * 40)

        for name, info in processes.items():
            proc = info['process']
            poll_result = proc.poll()

            if poll_result is None:
                # Still running - show last line of log
                try:
                    with open(info['log'], 'r') as f:
                        lines = f.readlines()
                        # Find last training progress line
                        for line in reversed(lines):
                            if 'Training:' in line or 'Episode' in line:
                                status = line.strip()[-60:]
                                break
                        else:
                            status = "Running..."
                except:
                    status = "Running..."

                elapsed = datetime.now() - info['start_time']
                print(f"  GPU {info['gpu']} | {name}: {status} (elapsed: {elapsed})")
            else:
                if name not in completed:
                    completed.add(name)
                    elapsed = datetime.now() - info['start_time']
                    status = "DONE" if poll_result == 0 else f"FAILED (code {poll_result})"
                    print(f"  GPU {info['gpu']} | {name}: {status} (total time: {elapsed})")

    # Final summary
    print(f"\n{'#'*60}")
    print("# ALL EXPERIMENTS COMPLETE")
    print(f"{'#'*60}")

    for name, info in processes.items():
        elapsed = datetime.now() - info['start_time']
        return_code = info['process'].returncode
        status = "SUCCESS" if return_code == 0 else f"FAILED (code {return_code})"
        print(f"  GPU {info['gpu']} | {name}: {status}")
        print(f"         Time: {elapsed}")
        print(f"         Log: {info['log']}")

    print(f"\nResults saved to: {save_dir}")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
