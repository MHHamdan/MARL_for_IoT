#!/usr/bin/env python3
"""
Evaluation Script for MARL-IoTP

Evaluates trained models and generates visualizations.

Usage:
    python scripts/evaluate.py --checkpoint results/mappo_best.pt --num_episodes 100
"""

import argparse
import yaml
import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.iot_env import IoTEdgeEnv
from agents.perception_agent import PerceptionAgent
from agents.orchestration_agent import OrchestrationAgent
from algorithms.mappo import MAPPO
from utils.metrics import compute_metrics, MetricsTracker
from utils.visualization import (
    plot_training_curves,
    plot_latency_accuracy_tradeoff,
    plot_offload_distribution,
    plot_agent_behavior
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MARL-IoTP')

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config (uses checkpoint config if not specified)'
    )
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results/evaluation/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--render', action='store_true')

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: str):
    """Load checkpoint and extract config."""
    state = torch.load(checkpoint_path, map_location=device)
    config = state.get('config', {})
    return state, config


def create_and_load_agents(config: dict, checkpoint_dir: Path, device: str):
    """Create agents and load their weights."""
    num_devices = config['num_iot_devices']
    num_servers = config['num_edge_servers']

    perception_agents = []
    for i in range(num_devices):
        agent = PerceptionAgent(i, config, device=device)
        agent_path = checkpoint_dir / f'perception_agent_{i}_best.pt'
        if agent_path.exists():
            agent.load(str(agent_path))
        perception_agents.append(agent)

    orchestration_agents = []
    for i in range(num_servers):
        agent = OrchestrationAgent(i, config, device=device)
        agent_path = checkpoint_dir / f'orchestration_agent_{i}_best.pt'
        if agent_path.exists():
            agent.load(str(agent_path))
        orchestration_agents.append(agent)

    return perception_agents, orchestration_agents


def run_evaluation(
    env: IoTEdgeEnv,
    perception_agents: list,
    orchestration_agents: list,
    config: dict,
    num_episodes: int,
    render: bool = False
):
    """
    Run evaluation episodes and collect detailed metrics.

    Returns:
        Dictionary with evaluation results
    """
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'latencies': [],
        'energies': [],
        'accuracies': [],
        'deadline_violations': [],
        'offload_counts': {'local': 0, 'edge': 0, 'cloud': 0},
        'agent_actions': {i: [] for i in range(len(perception_agents))}
    }

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(config['episode_length']):
            # Get perception actions
            perception_actions = []
            messages = []
            for i, agent in enumerate(perception_agents):
                action, _ = agent.get_action(
                    obs['perception'][i], deterministic=True
                )
                perception_actions.append(action)
                messages.append(agent.encode_message(obs['perception'][i]))
                results['agent_actions'][i].append(action)

            # Organize messages for orchestration agents
            devices_per_server = config.get('devices_per_server', 7)
            messages_per_server = []
            for i in range(len(orchestration_agents)):
                start_idx = i * devices_per_server
                end_idx = min(start_idx + devices_per_server, len(messages))
                messages_per_server.append(messages[start_idx:end_idx])

            # Get orchestration actions
            orchestration_actions = []
            for i, agent in enumerate(orchestration_agents):
                action, _ = agent.get_action(
                    obs['orchestration'][i],
                    messages_per_server[i],
                    deterministic=True
                )
                orchestration_actions.append(action)

            # Step environment
            actions = {
                'perception': perception_actions,
                'orchestration': orchestration_actions
            }
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            episode_reward += rewards['total']
            episode_length += 1

            # Collect metrics
            results['latencies'].append(info.get('avg_latency', 0))
            results['energies'].append(info.get('avg_energy', 0))
            results['accuracies'].append(info.get('avg_accuracy', 0))
            results['deadline_violations'].append(info.get('deadline_violations', 0))

            # Track offload distribution
            if 'offload_distribution' in info:
                for key in ['local', 'edge', 'cloud']:
                    results['offload_counts'][key] += info['offload_distribution'].get(key, 0)

            if render:
                env.render()

            obs = next_obs

            if terminated or truncated:
                break

        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)

    return results


def compute_statistics(results: dict) -> dict:
    """Compute summary statistics from results."""
    stats = {
        'reward': {
            'mean': np.mean(results['episode_rewards']),
            'std': np.std(results['episode_rewards']),
            'min': np.min(results['episode_rewards']),
            'max': np.max(results['episode_rewards'])
        },
        'latency': {
            'mean': np.mean(results['latencies']),
            'std': np.std(results['latencies']),
            'p50': np.percentile(results['latencies'], 50),
            'p95': np.percentile(results['latencies'], 95),
            'p99': np.percentile(results['latencies'], 99)
        },
        'energy': {
            'mean': np.mean(results['energies']),
            'std': np.std(results['energies'])
        },
        'accuracy': {
            'mean': np.mean(results['accuracies']),
            'std': np.std(results['accuracies'])
        },
        'deadline_violation_rate': np.mean(
            [v > 0 for v in results['deadline_violations']]
        ),
        'offload_distribution': {
            k: v / sum(results['offload_counts'].values())
            for k, v in results['offload_counts'].items()
        } if sum(results['offload_counts'].values()) > 0 else results['offload_counts']
    }
    return stats


def generate_visualizations(results: dict, stats: dict, save_dir: Path):
    """Generate and save visualization plots."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Offload distribution
    plot_offload_distribution(
        results['offload_counts'],
        save_path=str(save_dir / 'offload_distribution.png')
    )

    # Latency-accuracy trade-off
    plot_latency_accuracy_tradeoff(
        {'MAPPO': (results['latencies'], results['accuracies'])},
        save_path=str(save_dir / 'latency_accuracy_tradeoff.png')
    )

    # Agent behavior (first 3 agents)
    agent_actions_subset = {
        i: results['agent_actions'][i]
        for i in list(results['agent_actions'].keys())[:3]
    }
    plot_agent_behavior(
        agent_actions_subset,
        save_path=str(save_dir / 'agent_behavior.png')
    )

    print(f"Visualizations saved to {save_dir}")


def main():
    args = parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.is_file():
        state, config = load_checkpoint(str(checkpoint_path), args.device)
        checkpoint_dir = checkpoint_path.parent
    else:
        # Assume it's a directory
        checkpoint_dir = checkpoint_path
        trainer_path = checkpoint_dir / 'trainer_best.pt'
        state, config = load_checkpoint(str(trainer_path), args.device)

    # Load external config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Create environment
    print("Creating environment...")
    env = IoTEdgeEnv(config)

    # Create and load agents
    print("Loading agents...")
    perception_agents, orchestration_agents = create_and_load_agents(
        config, checkpoint_dir, args.device
    )

    # Run evaluation
    print(f"Running evaluation for {args.num_episodes} episodes...")
    results = run_evaluation(
        env, perception_agents, orchestration_agents,
        config, args.num_episodes, args.render
    )

    # Compute statistics
    stats = compute_statistics(results)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nReward:")
    print(f"  Mean: {stats['reward']['mean']:.2f} ± {stats['reward']['std']:.2f}")
    print(f"  Min: {stats['reward']['min']:.2f}, Max: {stats['reward']['max']:.2f}")
    print(f"\nLatency (ms):")
    print(f"  Mean: {stats['latency']['mean']:.2f} ± {stats['latency']['std']:.2f}")
    print(f"  P50: {stats['latency']['p50']:.2f}, P95: {stats['latency']['p95']:.2f}, P99: {stats['latency']['p99']:.2f}")
    print(f"\nEnergy (J):")
    print(f"  Mean: {stats['energy']['mean']:.4f} ± {stats['energy']['std']:.4f}")
    print(f"\nAccuracy:")
    print(f"  Mean: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
    print(f"\nDeadline Violation Rate: {stats['deadline_violation_rate']:.2%}")
    print(f"\nOffload Distribution:")
    for k, v in stats['offload_distribution'].items():
        print(f"  {k}: {v:.2%}")
    print("=" * 50)

    # Generate visualizations
    generate_visualizations(results, stats, save_dir)

    # Save results
    import json
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        # Convert numpy types to Python types
        json_stats = {}
        for k, v in stats.items():
            if isinstance(v, dict):
                json_stats[k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                    for kk, vv in v.items()
                }
            else:
                json_stats[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        json.dump(json_stats, f, indent=2)

    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
