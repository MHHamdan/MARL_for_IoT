#!/usr/bin/env python3
"""
Run All Baselines for MARL-IoTP

Trains and evaluates multiple baseline algorithms for comparison.

Usage:
    python scripts/run_baselines.py --config configs/default.yaml
    python scripts/run_baselines.py --algorithms random greedy mappo maddpg
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
from utils.logger import Logger
from utils.visualization import plot_comparison, create_results_table


def parse_args():
    parser = argparse.ArgumentParser(description='Run MARL-IoTP Baselines')

    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--algorithms', nargs='+',
        default=['random', 'greedy', 'round_robin', 'independent_ppo', 'mappo'],
        help='Algorithms to run'
    )
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/baselines/')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return parser.parse_args()


class RandomBaseline:
    """Random action baseline."""

    def __init__(self, env: IoTEdgeEnv, config: dict):
        self.env = env
        self.config = config
        self.num_models = config.get('num_perception_models', 4)
        self.devices_per_server = config.get('devices_per_server', 7)

    def get_actions(self, obs):
        """Get random actions for all agents."""
        perception_actions = []
        for i in range(self.config['num_iot_devices']):
            action = {
                'model_selection': np.random.randint(0, self.num_models),
                'frame_rate': np.random.uniform(0.1, 1.0, size=(1,))
            }
            perception_actions.append(action)

        orchestration_actions = []
        for i in range(self.config['num_edge_servers']):
            action = {
                'offload_decisions': np.random.randint(0, 3, size=self.devices_per_server),
                'resource_allocation': np.random.uniform(0.1, 1.0, size=(1,)),
                'bandwidth_allocation': np.random.uniform(0.1, 1.0, size=(1,))
            }
            orchestration_actions.append(action)

        return {
            'perception': perception_actions,
            'orchestration': orchestration_actions
        }

    def update(self):
        pass


class GreedyBaseline:
    """Greedy baseline - always picks best local option."""

    def __init__(self, env: IoTEdgeEnv, config: dict):
        self.env = env
        self.config = config
        self.num_models = config.get('num_perception_models', 4)
        self.devices_per_server = config.get('devices_per_server', 7)

    def get_actions(self, obs):
        """Get greedy actions."""
        perception_actions = []
        for i in range(self.config['num_iot_devices']):
            # Always use fastest model, full frame rate
            action = {
                'model_selection': 0,  # Fastest model
                'frame_rate': np.array([1.0])
            }
            perception_actions.append(action)

        orchestration_actions = []
        for i in range(self.config['num_edge_servers']):
            # Always offload to edge with max resources
            action = {
                'offload_decisions': np.ones(self.devices_per_server, dtype=np.int64),
                'resource_allocation': np.array([1.0]),
                'bandwidth_allocation': np.array([1.0])
            }
            orchestration_actions.append(action)

        return {
            'perception': perception_actions,
            'orchestration': orchestration_actions
        }

    def update(self):
        pass


class RoundRobinBaseline:
    """Round-robin baseline."""

    def __init__(self, env: IoTEdgeEnv, config: dict):
        self.env = env
        self.config = config
        self.num_models = config.get('num_perception_models', 4)
        self.devices_per_server = config.get('devices_per_server', 7)
        self.step = 0

    def get_actions(self, obs):
        """Get round-robin actions."""
        self.step += 1

        perception_actions = []
        for i in range(self.config['num_iot_devices']):
            # Cycle through models
            action = {
                'model_selection': (self.step + i) % self.num_models,
                'frame_rate': np.array([0.5])
            }
            perception_actions.append(action)

        orchestration_actions = []
        for i in range(self.config['num_edge_servers']):
            # Cycle through offload options
            offload = np.array([
                (self.step + j) % 3 for j in range(self.devices_per_server)
            ], dtype=np.int64)
            action = {
                'offload_decisions': offload,
                'resource_allocation': np.array([0.5]),
                'bandwidth_allocation': np.array([0.5])
            }
            orchestration_actions.append(action)

        return {
            'perception': perception_actions,
            'orchestration': orchestration_actions
        }

    def update(self):
        pass


class IndependentPPOBaseline:
    """Independent PPO (no communication between agents)."""

    def __init__(self, env: IoTEdgeEnv, config: dict, device: str):
        self.env = env
        self.config = config
        self.device = device

        # Create agents
        self.perception_agents = [
            PerceptionAgent(i, config, device=device)
            for i in range(config['num_iot_devices'])
        ]
        self.orchestration_agents = [
            OrchestrationAgent(i, config, device=device)
            for i in range(config['num_edge_servers'])
        ]

        # Create independent trainers (simplified - just update each agent)
        self.gamma = config.get('gamma', 0.99)
        self.devices_per_server = config.get('devices_per_server', 7)

    def get_actions(self, obs):
        """Get actions from independent agents (no message passing)."""
        perception_actions = []
        for i, agent in enumerate(self.perception_agents):
            action, _ = agent.get_action(obs['perception'][i])
            perception_actions.append(action)

        # No message passing - use zero messages
        zero_messages = [np.zeros(self.config.get('message_dim', 8))
                        for _ in range(self.devices_per_server)]

        orchestration_actions = []
        for i, agent in enumerate(self.orchestration_agents):
            action, _ = agent.get_action(obs['orchestration'][i], zero_messages)
            orchestration_actions.append(action)

        return {
            'perception': perception_actions,
            'orchestration': orchestration_actions
        }

    def update(self):
        # Simplified update - in practice would need proper buffer
        pass


def run_baseline(
    baseline_name: str,
    env: IoTEdgeEnv,
    config: dict,
    num_episodes: int,
    device: str
):
    """Run a single baseline algorithm."""
    print(f"\n{'='*50}")
    print(f"Running {baseline_name} baseline...")
    print(f"{'='*50}")

    # Create baseline
    if baseline_name == 'random':
        baseline = RandomBaseline(env, config)
    elif baseline_name == 'greedy':
        baseline = GreedyBaseline(env, config)
    elif baseline_name == 'round_robin':
        baseline = RoundRobinBaseline(env, config)
    elif baseline_name == 'independent_ppo':
        baseline = IndependentPPOBaseline(env, config, device)
    elif baseline_name == 'mappo':
        # Use full MAPPO
        perception_agents = [
            PerceptionAgent(i, config, device=device)
            for i in range(config['num_iot_devices'])
        ]
        orchestration_agents = [
            OrchestrationAgent(i, config, device=device)
            for i in range(config['num_edge_servers'])
        ]
        trainer = MAPPO(perception_agents, orchestration_agents, config, device)
        return run_mappo(trainer, env, config, num_episodes)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # Run episodes
    rewards = []
    metrics = {'latency': [], 'energy': [], 'accuracy': []}

    for ep in tqdm(range(num_episodes), desc=baseline_name):
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(config['episode_length']):
            actions = baseline.get_actions(obs)
            next_obs, reward_dict, terminated, truncated, info = env.step(actions)

            episode_reward += reward_dict['total']
            metrics['latency'].append(info.get('avg_latency', 0))
            metrics['energy'].append(info.get('avg_energy', 0))
            metrics['accuracy'].append(info.get('avg_accuracy', 0))

            obs = next_obs

            if terminated or truncated:
                break

        rewards.append(episode_reward)

    return {
        'rewards': rewards,
        'metrics': metrics,
        'final_reward': np.mean(rewards[-100:]),
        'final_latency': np.mean(metrics['latency'][-1000:]),
        'final_energy': np.mean(metrics['energy'][-1000:]),
        'final_accuracy': np.mean(metrics['accuracy'][-1000:])
    }


def run_mappo(trainer, env, config, num_episodes):
    """Run MAPPO training."""
    rewards = []
    metrics = {'latency': [], 'energy': [], 'accuracy': []}

    update_interval = config.get('update_interval', 10)
    steps_per_update = config.get('episode_length', 200) * update_interval

    episode = 0
    pbar = tqdm(total=num_episodes, desc="MAPPO")

    while episode < num_episodes:
        rollout_stats = trainer.collect_rollout(env, steps_per_update)
        trainer.update()

        for ep_reward in rollout_stats.get('episode_rewards', []):
            rewards.append(ep_reward)
            episode += 1
            pbar.update(1)

    pbar.close()

    return {
        'rewards': rewards[:num_episodes],
        'metrics': metrics,
        'final_reward': np.mean(rewards[-100:]) if rewards else 0,
        'final_latency': 0,
        'final_energy': 0,
        'final_accuracy': 0
    }


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['devices_per_server'] = int(
        np.ceil(config['num_iot_devices'] / config['num_edge_servers'])
    )

    # Create save directory
    save_dir = Path(args.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = save_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    print("Creating environment...")
    env = IoTEdgeEnv(config)

    # Run all baselines
    all_results = {}

    for algo_name in args.algorithms:
        try:
            results = run_baseline(
                algo_name, env, config, args.episodes, args.device
            )
            all_results[algo_name] = results

            print(f"\n{algo_name} Results:")
            print(f"  Final Reward: {results['final_reward']:.2f}")
            print(f"  Final Latency: {results['final_latency']:.2f}")
            print(f"  Final Accuracy: {results['final_accuracy']:.4f}")

        except Exception as e:
            print(f"Error running {algo_name}: {e}")
            continue

    # Generate comparison plots
    comparison_data = {
        algo: {'reward': results['rewards']}
        for algo, results in all_results.items()
    }

    plot_comparison(
        comparison_data,
        metric='reward',
        save_path=str(save_dir / 'comparison.png'),
        title='Algorithm Comparison'
    )

    # Create results table
    table_results = {
        algo: {
            'Final Reward': results['final_reward'],
            'Final Latency': results['final_latency'],
            'Final Energy': results['final_energy'],
            'Final Accuracy': results['final_accuracy']
        }
        for algo, results in all_results.items()
    }

    table_str = create_results_table(
        table_results,
        metrics=['Final Reward', 'Final Latency', 'Final Energy', 'Final Accuracy'],
        save_path=str(save_dir / 'comparison_table.csv')
    )
    print("\n" + "="*50)
    print("COMPARISON TABLE")
    print("="*50)
    print(table_str)

    # Save all results
    with open(save_dir / 'all_results.json', 'w') as f:
        json_results = {}
        for algo, results in all_results.items():
            json_results[algo] = {
                'final_reward': float(results['final_reward']),
                'final_latency': float(results['final_latency']),
                'final_energy': float(results['final_energy']),
                'final_accuracy': float(results['final_accuracy'])
            }
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
