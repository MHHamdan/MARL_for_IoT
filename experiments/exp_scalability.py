#!/usr/bin/env python3
"""
Scalability Experiment for MARL-IoTP

Tests how the framework scales with increasing number of IoT devices.

Usage:
    python experiments/exp_scalability.py --device_counts 10 20 30 50
"""

import argparse
import yaml
import torch
import numpy as np
import random
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.iot_env import IoTEdgeEnv
from agents.perception_agent import PerceptionAgent
from agents.orchestration_agent import OrchestrationAgent
from algorithms.mappo import MAPPO
from utils.visualization import plot_scalability


def parse_args():
    parser = argparse.ArgumentParser(description='MARL-IoTP Scalability Experiment')

    parser.add_argument(
        '--config', type=str, default='configs/mappo.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--device_counts', nargs='+', type=int, default=[10, 20, 30, 50],
        help='Device counts to test'
    )
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/scalability/')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return parser.parse_args()


def run_experiment_for_device_count(
    base_config: dict,
    num_devices: int,
    episodes: int,
    eval_episodes: int,
    device: str
):
    """
    Run scalability experiment for a specific device count.

    Returns:
        Dictionary with results
    """
    # Update config for this device count
    config = base_config.copy()
    config['num_iot_devices'] = num_devices

    # Scale servers with devices (roughly 7 devices per server)
    config['num_edge_servers'] = max(1, num_devices // 7)
    config['devices_per_server'] = int(
        np.ceil(num_devices / config['num_edge_servers'])
    )

    print(f"\n{'='*60}")
    print(f"Testing with {num_devices} devices, {config['num_edge_servers']} servers")
    print(f"{'='*60}")

    # Create environment
    env = IoTEdgeEnv(config)

    # Create agents
    perception_agents = [
        PerceptionAgent(i, config, device=device)
        for i in range(num_devices)
    ]
    orchestration_agents = [
        OrchestrationAgent(i, config, device=device)
        for i in range(config['num_edge_servers'])
    ]

    # Create trainer
    trainer = MAPPO(
        perception_agents=perception_agents,
        orchestration_agents=orchestration_agents,
        config=config,
        device=device
    )

    # Training
    rewards = []
    update_interval = config.get('update_interval', 10)
    steps_per_update = config.get('episode_length', 200) * update_interval

    episode = 0
    pbar = tqdm(total=episodes, desc=f"{num_devices} devices")

    import time
    start_time = time.time()

    while episode < episodes:
        rollout_stats = trainer.collect_rollout(env, steps_per_update)
        trainer.update()

        for ep_reward in rollout_stats.get('episode_rewards', []):
            rewards.append(ep_reward)
            episode += 1
            pbar.update(1)

            if episode >= episodes:
                break

    training_time = time.time() - start_time
    pbar.close()

    # Evaluation
    print("Running evaluation...")
    eval_rewards = []
    eval_latencies = []
    eval_accuracies = []
    eval_throughputs = []

    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_tasks = 0

        for step in range(config['episode_length']):
            # Get actions
            perception_actions = []
            messages = []
            for i, agent in enumerate(perception_agents):
                action, _ = agent.get_action(obs['perception'][i], deterministic=True)
                perception_actions.append(action)
                messages.append(agent.encode_message(obs['perception'][i]))

            devices_per_server = config['devices_per_server']
            messages_per_server = []
            for i in range(len(orchestration_agents)):
                start_idx = i * devices_per_server
                end_idx = min(start_idx + devices_per_server, len(messages))
                messages_per_server.append(messages[start_idx:end_idx])

            orchestration_actions = []
            for i, agent in enumerate(orchestration_agents):
                action, _ = agent.get_action(
                    obs['orchestration'][i],
                    messages_per_server[i],
                    deterministic=True
                )
                orchestration_actions.append(action)

            actions = {
                'perception': perception_actions,
                'orchestration': orchestration_actions
            }
            next_obs, reward_dict, terminated, truncated, info = env.step(actions)

            episode_reward += reward_dict['total']
            eval_latencies.append(info.get('avg_latency', 0))
            eval_accuracies.append(info.get('avg_accuracy', 0))
            episode_tasks += info.get('tasks_processed', 0)

            obs = next_obs
            if terminated or truncated:
                break

        eval_rewards.append(episode_reward)
        eval_throughputs.append(episode_tasks)

    results = {
        'num_devices': num_devices,
        'num_servers': config['num_edge_servers'],
        'training_time': training_time,
        'training_rewards': rewards,
        'eval_reward_mean': np.mean(eval_rewards),
        'eval_reward_std': np.std(eval_rewards),
        'eval_latency_mean': np.mean(eval_latencies),
        'eval_accuracy_mean': np.mean(eval_accuracies),
        'eval_throughput_mean': np.mean(eval_throughputs),
        'convergence_episode': find_convergence(rewards)
    }

    return results


def find_convergence(rewards, window=100, threshold=0.95):
    """Find episode where training converged."""
    if len(rewards) < window:
        return len(rewards)

    final_reward = np.mean(rewards[-window:])

    for i in range(window, len(rewards)):
        current_avg = np.mean(rewards[i-window:i])
        if current_avg >= threshold * final_reward:
            return i

    return len(rewards)


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    # Create save directory
    save_dir = Path(args.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = save_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = {}

    for num_devices in args.device_counts:
        results = run_experiment_for_device_count(
            base_config, num_devices, args.episodes, args.eval_episodes, args.device
        )
        all_results[num_devices] = results

        # Print interim results
        print(f"\nResults for {num_devices} devices:")
        print(f"  Training time: {results['training_time']:.1f}s")
        print(f"  Eval reward: {results['eval_reward_mean']:.2f} ± {results['eval_reward_std']:.2f}")
        print(f"  Eval latency: {results['eval_latency_mean']:.2f}ms")
        print(f"  Eval accuracy: {results['eval_accuracy_mean']:.4f}")
        print(f"  Throughput: {results['eval_throughput_mean']:.1f} tasks/episode")
        print(f"  Convergence: episode {results['convergence_episode']}")

    # Generate visualization
    scalability_data = {
        'MAPPO': {
            n: {
                'reward': r['eval_reward_mean'],
                'latency': r['eval_latency_mean'],
                'throughput': r['eval_throughput_mean'],
                'training_time': r['training_time']
            }
            for n, r in all_results.items()
        }
    }

    plot_scalability(
        device_counts=args.device_counts,
        results=scalability_data,
        metrics=['reward', 'latency', 'throughput'],
        save_path=str(save_dir / 'scalability.png')
    )

    # Save results
    json_results = {}
    for n, r in all_results.items():
        json_results[str(n)] = {
            'num_devices': r['num_devices'],
            'num_servers': r['num_servers'],
            'training_time': r['training_time'],
            'eval_reward_mean': r['eval_reward_mean'],
            'eval_reward_std': r['eval_reward_std'],
            'eval_latency_mean': r['eval_latency_mean'],
            'eval_accuracy_mean': r['eval_accuracy_mean'],
            'eval_throughput_mean': r['eval_throughput_mean'],
            'convergence_episode': r['convergence_episode']
        }

    with open(save_dir / 'scalability_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SCALABILITY EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
