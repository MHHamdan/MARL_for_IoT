#!/usr/bin/env python3
"""
Ablation Study for MARL-IoTP

Tests the impact of different components on performance.

Usage:
    python experiments/exp_ablation.py --ablation communication
    python experiments/exp_ablation.py --ablation all
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
from copy import deepcopy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.iot_env import IoTEdgeEnv
from agents.perception_agent import PerceptionAgent
from agents.orchestration_agent import OrchestrationAgent
from algorithms.mappo import MAPPO
from utils.visualization import plot_ablation


def parse_args():
    parser = argparse.ArgumentParser(description='MARL-IoTP Ablation Study')

    parser.add_argument(
        '--config', type=str, default='configs/mappo.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--ablation', type=str, default='all',
        choices=['all', 'communication', 'attention', 'architecture', 'reward_weights'],
        help='Ablation type to run'
    )
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/ablation/')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return parser.parse_args()


class NoCommunicationMARL:
    """MARL without inter-agent communication."""

    def __init__(self, perception_agents, orchestration_agents, config, device):
        self.perception_agents = perception_agents
        self.orchestration_agents = orchestration_agents
        self.config = config
        self.device = device
        self.devices_per_server = config.get('devices_per_server', 7)
        self.message_dim = config.get('message_dim', 8)

        # Create dummy MAPPO trainer
        self.trainer = MAPPO(
            perception_agents, orchestration_agents, config, device
        )

    def collect_rollout(self, env, num_steps):
        """Collect rollout without communication."""
        obs, _ = env.reset()
        episode_rewards = []
        current_episode_reward = 0

        for step in range(num_steps):
            # Get perception actions (NO message encoding)
            perception_actions = []
            for i, agent in enumerate(self.perception_agents):
                action, _ = agent.get_action(obs['perception'][i])
                perception_actions.append(action)

            # Zero messages (no communication)
            zero_messages = [
                np.zeros(self.message_dim) for _ in range(self.devices_per_server)
            ]

            orchestration_actions = []
            for i, agent in enumerate(self.orchestration_agents):
                action, _ = agent.get_action(obs['orchestration'][i], zero_messages)
                orchestration_actions.append(action)

            actions = {
                'perception': perception_actions,
                'orchestration': orchestration_actions
            }
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            current_episode_reward += rewards['total']
            obs = next_obs

            if terminated or truncated:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = env.reset()

        return {'episode_rewards': episode_rewards}

    def update(self):
        return self.trainer.update()


def run_variant(
    variant_name: str,
    config: dict,
    episodes: int,
    eval_episodes: int,
    device: str
):
    """Run a single ablation variant."""
    print(f"\n{'='*60}")
    print(f"Running variant: {variant_name}")
    print(f"{'='*60}")

    # Create environment
    env = IoTEdgeEnv(config)

    # Create agents
    num_devices = config['num_iot_devices']
    num_servers = config['num_edge_servers']

    perception_agents = [
        PerceptionAgent(i, config, device=device)
        for i in range(num_devices)
    ]
    orchestration_agents = [
        OrchestrationAgent(i, config, device=device)
        for i in range(num_servers)
    ]

    # Create trainer based on variant
    if variant_name == 'no_communication':
        trainer = NoCommunicationMARL(
            perception_agents, orchestration_agents, config, device
        )
    else:
        trainer = MAPPO(
            perception_agents, orchestration_agents, config, device
        )

    # Training
    rewards = []
    update_interval = config.get('update_interval', 10)
    steps_per_update = config.get('episode_length', 200) * update_interval

    episode = 0
    pbar = tqdm(total=episodes, desc=variant_name)

    while episode < episodes:
        rollout_stats = trainer.collect_rollout(env, steps_per_update)
        trainer.update()

        for ep_reward in rollout_stats.get('episode_rewards', []):
            rewards.append(ep_reward)
            episode += 1
            pbar.update(1)
            if episode >= episodes:
                break

    pbar.close()

    # Evaluation
    print("Running evaluation...")
    eval_results = evaluate_variant(
        env, perception_agents, orchestration_agents, config,
        eval_episodes, variant_name
    )

    return {
        'training_rewards': rewards,
        **eval_results
    }


def evaluate_variant(
    env, perception_agents, orchestration_agents, config,
    num_episodes, variant_name
):
    """Evaluate a variant."""
    rewards = []
    latencies = []
    accuracies = []

    devices_per_server = config.get('devices_per_server', 7)
    message_dim = config.get('message_dim', 8)

    for _ in tqdm(range(num_episodes), desc=f"Eval {variant_name}"):
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(config['episode_length']):
            perception_actions = []
            messages = []

            for i, agent in enumerate(perception_agents):
                action, _ = agent.get_action(obs['perception'][i], deterministic=True)
                perception_actions.append(action)
                if variant_name == 'no_communication':
                    messages.append(np.zeros(message_dim))
                else:
                    messages.append(agent.encode_message(obs['perception'][i]))

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
            latencies.append(info.get('avg_latency', 0))
            accuracies.append(info.get('avg_accuracy', 0))

            obs = next_obs
            if terminated or truncated:
                break

        rewards.append(episode_reward)

    return {
        'eval_reward_mean': np.mean(rewards),
        'eval_reward_std': np.std(rewards),
        'eval_latency_mean': np.mean(latencies),
        'eval_accuracy_mean': np.mean(accuracies)
    }


def run_communication_ablation(base_config, episodes, eval_episodes, device):
    """Run communication ablation study."""
    variants = {
        'full_communication': base_config,
        'no_communication': base_config
    }

    results = {}
    for name, config in variants.items():
        results[name] = run_variant(name, config, episodes, eval_episodes, device)

    return results


def run_reward_weight_ablation(base_config, episodes, eval_episodes, device):
    """Run reward weight ablation study."""
    variants = {
        'balanced': {'latency': 0.33, 'energy': 0.33, 'accuracy': 0.34},
        'latency_focused': {'latency': 0.6, 'energy': 0.2, 'accuracy': 0.2},
        'accuracy_focused': {'latency': 0.2, 'energy': 0.2, 'accuracy': 0.6},
        'energy_focused': {'latency': 0.2, 'energy': 0.6, 'accuracy': 0.2}
    }

    results = {}
    for name, weights in variants.items():
        config = deepcopy(base_config)
        config['reward_weights'] = weights
        results[name] = run_variant(name, config, episodes, eval_episodes, device)

    return results


def run_attention_ablation(base_config, episodes, eval_episodes, device):
    """Run attention mechanism ablation study."""
    results = {}

    # With attention (default)
    config_attention = deepcopy(base_config)
    config_attention['use_attention'] = True
    results['with_attention'] = run_variant('with_attention', config_attention, episodes, eval_episodes, device)

    # Without attention
    config_no_attention = deepcopy(base_config)
    config_no_attention['use_attention'] = False
    results['no_attention'] = run_variant('no_attention', config_no_attention, episodes, eval_episodes, device)

    return results


def run_architecture_ablation(base_config, episodes, eval_episodes, device):
    """Run architecture ablation study (shared vs separate critics)."""
    results = {}

    # Separate critics (default MAPPO)
    config_separate = deepcopy(base_config)
    results['separate_critics'] = run_variant('separate_critics', config_separate, episodes, eval_episodes, device)

    # Smaller hidden dimension
    config_small = deepcopy(base_config)
    config_small['hidden_dim'] = 64
    results['small_network'] = run_variant('small_network', config_small, episodes, eval_episodes, device)

    # Larger hidden dimension
    config_large = deepcopy(base_config)
    config_large['hidden_dim'] = 256
    results['large_network'] = run_variant('large_network', config_large, episodes, eval_episodes, device)

    return results


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    base_config['devices_per_server'] = int(
        np.ceil(base_config['num_iot_devices'] / base_config['num_edge_servers'])
    )

    # Create save directory
    save_dir = Path(args.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = save_dir / f"{args.ablation}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run ablation
    if args.ablation == 'communication' or args.ablation == 'all':
        print("\n" + "="*60)
        print("COMMUNICATION ABLATION")
        print("="*60)
        comm_results = run_communication_ablation(
            base_config, args.episodes, args.eval_episodes, args.device
        )

        # Visualize
        ablation_data = {
            name: {
                'reward': r['eval_reward_mean'],
                'latency': r['eval_latency_mean'],
                'accuracy': r['eval_accuracy_mean']
            }
            for name, r in comm_results.items()
        }
        plot_ablation(
            ablation_data,
            save_path=str(save_dir / 'communication_ablation.png')
        )

        # Save results
        with open(save_dir / 'communication_results.json', 'w') as f:
            json_results = {
                name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in r.items() if k != 'training_rewards'}
                for name, r in comm_results.items()
            }
            json.dump(json_results, f, indent=2)

    if args.ablation == 'reward_weights' or args.ablation == 'all':
        print("\n" + "="*60)
        print("REWARD WEIGHT ABLATION")
        print("="*60)
        weight_results = run_reward_weight_ablation(
            base_config, args.episodes, args.eval_episodes, args.device
        )

        # Visualize
        ablation_data = {
            name: {
                'reward': r['eval_reward_mean'],
                'latency': r['eval_latency_mean'],
                'accuracy': r['eval_accuracy_mean']
            }
            for name, r in weight_results.items()
        }
        plot_ablation(
            ablation_data,
            save_path=str(save_dir / 'reward_weight_ablation.png')
        )

        # Save results
        with open(save_dir / 'reward_weight_results.json', 'w') as f:
            json_results = {
                name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in r.items() if k != 'training_rewards'}
                for name, r in weight_results.items()
            }
            json.dump(json_results, f, indent=2)

    if args.ablation == 'attention' or args.ablation == 'all':
        print("\n" + "="*60)
        print("ATTENTION MECHANISM ABLATION")
        print("="*60)
        attention_results = run_attention_ablation(
            base_config, args.episodes, args.eval_episodes, args.device
        )

        # Visualize
        ablation_data = {
            name: {
                'reward': r['eval_reward_mean'],
                'latency': r['eval_latency_mean'],
                'accuracy': r['eval_accuracy_mean']
            }
            for name, r in attention_results.items()
        }
        plot_ablation(
            ablation_data,
            save_path=str(save_dir / 'attention_ablation.png')
        )

        # Save results
        with open(save_dir / 'attention_results.json', 'w') as f:
            json_results = {
                name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in r.items() if k != 'training_rewards'}
                for name, r in attention_results.items()
            }
            json.dump(json_results, f, indent=2)

    if args.ablation == 'architecture' or args.ablation == 'all':
        print("\n" + "="*60)
        print("ARCHITECTURE ABLATION")
        print("="*60)
        arch_results = run_architecture_ablation(
            base_config, args.episodes, args.eval_episodes, args.device
        )

        # Visualize
        ablation_data = {
            name: {
                'reward': r['eval_reward_mean'],
                'latency': r['eval_latency_mean'],
                'accuracy': r['eval_accuracy_mean']
            }
            for name, r in arch_results.items()
        }
        plot_ablation(
            ablation_data,
            save_path=str(save_dir / 'architecture_ablation.png')
        )

        # Save results
        with open(save_dir / 'architecture_results.json', 'w') as f:
            json_results = {
                name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in r.items() if k != 'training_rewards'}
                for name, r in arch_results.items()
            }
            json.dump(json_results, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to {save_dir}")


if __name__ == '__main__':
    main()
