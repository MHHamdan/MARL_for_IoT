#!/usr/bin/env python3
"""
Enhanced Experiments for MARL-IoTP Paper Revision

Includes:
1. DPTORA and MADOA baseline comparisons
2. Message dimension ablation study
3. Mutual information analysis
4. Enhanced metrics collection (fairness, tail latency)

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/run_enhanced_experiments.py --experiment baselines
    CUDA_VISIBLE_DEVICES=3 python experiments/run_enhanced_experiments.py --experiment message_dim
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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.iot_env import IoTEdgeEnv
from agents.perception_agent import PerceptionAgent
from agents.orchestration_agent import OrchestrationAgent
from algorithms.mappo import MAPPO
from algorithms.baselines import DPTORABaseline, MADOABaseline
from utils.logger import Logger
from utils.analysis import (
    EnhancedMetricsCollector,
    MutualInformationEstimator,
    AttentionAnalyzer,
    MessageDimensionAnalyzer,
    compute_jains_fairness_index,
    compute_tail_latencies,
    analyze_message_semantics
)


def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced MARL-IoTP Experiments')

    parser.add_argument(
        '--experiment', type=str, required=True,
        choices=['baselines', 'message_dim', 'communication_analysis', 'all'],
        help='Experiment type to run'
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--eval_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/enhanced/')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return parser.parse_args()


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_dptora_experiment(env, config, num_episodes, device, save_dir):
    """Run DPTORA baseline experiment."""
    print("\n" + "=" * 60)
    print("Running DPTORA Baseline Experiment")
    print("=" * 60)

    # Create perception agents (same as MAPPO)
    perception_agents = [
        PerceptionAgent(i, config, device=device)
        for i in range(config['num_iot_devices'])
    ]

    # Create DPTORA baseline
    dptora = DPTORABaseline(
        obs_dim=config.get('orchestration_obs_dim', 32),
        message_dim=config.get('message_dim', 8),
        num_devices=config['num_iot_devices'],
        num_servers=config['num_edge_servers'],
        config=config,
        device=device
    )

    # Create metrics collector
    metrics = EnhancedMetricsCollector()
    rewards = []

    for ep in tqdm(range(num_episodes), desc="DPTORA Training"):
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(config['episode_length']):
            # Get perception actions and messages
            perception_actions = []
            messages = []

            for i, agent in enumerate(perception_agents):
                action, _ = agent.get_action(obs['perception'][i])
                perception_actions.append(action)
                messages.append(agent.encode_message(obs['perception'][i]))

            # Organize messages per server
            devices_per_server = config.get('devices_per_server', 7)
            orchestration_actions = []

            for s in range(config['num_edge_servers']):
                start_idx = s * devices_per_server
                end_idx = min(start_idx + devices_per_server, len(messages))
                server_messages = messages[start_idx:end_idx]

                action = dptora.get_action(
                    obs['orchestration'][s],
                    server_messages,
                    server_id=s
                )
                orchestration_actions.append(action)

            # Step environment
            actions = {
                'perception': perception_actions,
                'orchestration': orchestration_actions
            }
            next_obs, reward_dict, terminated, truncated, info = env.step(actions)

            episode_reward += reward_dict['total']

            # Collect metrics
            metrics.add_step(
                latency=info.get('avg_latency', 0),
                energy=info.get('avg_energy', 0),
                accuracy=info.get('avg_accuracy', 0),
                reward=reward_dict['total'],
                violation=info.get('deadline_violated', False)
            )

            obs = next_obs

            if terminated or truncated:
                break

        rewards.append(episode_reward)

    # Get summary
    summary = metrics.get_summary()
    summary['final_reward'] = float(np.mean(rewards[-100:]))
    summary['reward_std'] = float(np.std(rewards[-100:]))

    # Save results
    results_path = save_dir / 'dptora_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDPTORA Results:")
    print(f"  Final Reward: {summary['final_reward']:.2f} +/- {summary['reward_std']:.2f}")
    print(f"  Mean Latency: {summary['mean_latency']:.2f} ms")
    print(f"  P95 Latency: {summary['p95_latency']:.2f} ms")
    print(f"  Mean Accuracy: {summary['mean_accuracy']:.4f}")
    print(f"  Fairness: {summary['reward_fairness']:.4f}")

    return summary


def run_madoa_experiment(env, config, num_episodes, device, save_dir):
    """Run MADOA baseline experiment."""
    print("\n" + "=" * 60)
    print("Running MADOA Baseline Experiment")
    print("=" * 60)

    perception_agents = [
        PerceptionAgent(i, config, device=device)
        for i in range(config['num_iot_devices'])
    ]

    madoa = MADOABaseline(
        obs_dim=config.get('orchestration_obs_dim', 32),
        message_dim=config.get('message_dim', 8),
        num_devices=config['num_iot_devices'],
        num_servers=config['num_edge_servers'],
        config=config,
        device=device
    )

    metrics = EnhancedMetricsCollector()
    rewards = []

    for ep in tqdm(range(num_episodes), desc="MADOA Training"):
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(config['episode_length']):
            perception_actions = []
            messages = []

            for i, agent in enumerate(perception_agents):
                action, _ = agent.get_action(obs['perception'][i])
                perception_actions.append(action)
                messages.append(agent.encode_message(obs['perception'][i]))

            devices_per_server = config.get('devices_per_server', 7)
            orchestration_actions = []

            for s in range(config['num_edge_servers']):
                start_idx = s * devices_per_server
                end_idx = min(start_idx + devices_per_server, len(messages))
                server_messages = messages[start_idx:end_idx]

                action = madoa.get_action(
                    obs['orchestration'][s],
                    server_messages,
                    server_id=s
                )
                orchestration_actions.append(action)

            actions = {
                'perception': perception_actions,
                'orchestration': orchestration_actions
            }
            next_obs, reward_dict, terminated, truncated, info = env.step(actions)

            episode_reward += reward_dict['total']

            metrics.add_step(
                latency=info.get('avg_latency', 0),
                energy=info.get('avg_energy', 0),
                accuracy=info.get('avg_accuracy', 0),
                reward=reward_dict['total'],
                violation=info.get('deadline_violated', False)
            )

            obs = next_obs

            if terminated or truncated:
                break

        rewards.append(episode_reward)

    summary = metrics.get_summary()
    summary['final_reward'] = float(np.mean(rewards[-100:]))
    summary['reward_std'] = float(np.std(rewards[-100:]))

    results_path = save_dir / 'madoa_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nMADOA Results:")
    print(f"  Final Reward: {summary['final_reward']:.2f} +/- {summary['reward_std']:.2f}")
    print(f"  Mean Latency: {summary['mean_latency']:.2f} ms")
    print(f"  P95 Latency: {summary['p95_latency']:.2f} ms")
    print(f"  Mean Accuracy: {summary['mean_accuracy']:.4f}")
    print(f"  Fairness: {summary['reward_fairness']:.4f}")

    return summary


def run_mappo_with_enhanced_metrics(env, config, num_episodes, device, save_dir):
    """Run MAPPO with enhanced metrics collection."""
    print("\n" + "=" * 60)
    print("Running MAPPO with Enhanced Metrics")
    print("=" * 60)

    perception_agents = [
        PerceptionAgent(i, config, device=device)
        for i in range(config['num_iot_devices'])
    ]
    orchestration_agents = [
        OrchestrationAgent(i, config, device=device)
        for i in range(config['num_edge_servers'])
    ]

    trainer = MAPPO(perception_agents, orchestration_agents, config, device)

    metrics = EnhancedMetricsCollector()
    mi_estimator = MutualInformationEstimator()
    attention_analyzer = AttentionAnalyzer()

    rewards = []
    update_interval = config.get('update_interval', 10)
    steps_per_update = config.get('episode_length', 200) * update_interval

    episode = 0
    pbar = tqdm(total=num_episodes, desc="MAPPO Training")

    while episode < num_episodes:
        rollout_stats = trainer.collect_rollout(env, steps_per_update)
        trainer.update()

        for ep_reward in rollout_stats.get('episode_rewards', []):
            rewards.append(ep_reward)
            episode += 1
            pbar.update(1)

            if episode >= num_episodes:
                break

    pbar.close()

    # Evaluation phase with enhanced metrics
    print("\nRunning evaluation with enhanced metrics...")

    for eval_ep in tqdm(range(100), desc="Evaluation"):
        obs, _ = env.reset()

        for step in range(config['episode_length']):
            perception_actions = []
            messages = []

            for i, agent in enumerate(perception_agents):
                action, _ = agent.get_action(obs['perception'][i])
                perception_actions.append(action)
                message = agent.encode_message(obs['perception'][i])
                messages.append(message)

                # Collect MI samples
                mi_estimator.add_sample(obs['perception'][i], message)

            devices_per_server = config.get('devices_per_server', 7)
            orchestration_actions = []

            for s, agent in enumerate(orchestration_agents):
                start_idx = s * devices_per_server
                end_idx = min(start_idx + devices_per_server, len(messages))
                server_messages = messages[start_idx:end_idx]

                action, action_info = agent.get_action(
                    obs['orchestration'][s], server_messages
                )
                orchestration_actions.append(action)

                # Collect attention weights
                if 'attention_weights' in action_info:
                    weights = action_info['attention_weights'].cpu().numpy()
                    attention_analyzer.add_sample(
                        weights.squeeze(),
                        np.array(server_messages),
                        {'server_id': s}
                    )

            actions = {
                'perception': perception_actions,
                'orchestration': orchestration_actions
            }
            next_obs, reward_dict, terminated, truncated, info = env.step(actions)

            # Collect per-device metrics
            for i in range(config['num_iot_devices']):
                metrics.add_step(
                    latency=info.get('avg_latency', 0),
                    energy=info.get('avg_energy', 0),
                    accuracy=info.get('avg_accuracy', 0),
                    reward=reward_dict['perception'][i] if 'perception' in reward_dict else reward_dict['total'] / config['num_iot_devices'],
                    violation=info.get('deadline_violated', False),
                    device_id=i
                )

            obs = next_obs

            if terminated or truncated:
                break

    # Compile results
    summary = metrics.get_summary()
    summary['final_reward'] = float(np.mean(rewards[-100:]))
    summary['reward_std'] = float(np.std(rewards[-100:]))

    # Add MI analysis
    mi_results = mi_estimator.compute_mi()
    summary['mutual_information'] = mi_results

    # Add attention analysis
    attention_results = attention_analyzer.analyze()
    summary['attention_analysis'] = attention_results

    results_path = save_dir / 'mappo_enhanced_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

    print(f"\nMAPPO Enhanced Results:")
    print(f"  Final Reward: {summary['final_reward']:.2f} +/- {summary['reward_std']:.2f}")
    print(f"  Mean Latency: {summary['mean_latency']:.2f} ms")
    print(f"  P95 Latency: {summary['p95_latency']:.2f} ms")
    print(f"  P99 Latency: {summary['p99_latency']:.2f} ms")
    print(f"  Mean Accuracy: {summary['mean_accuracy']:.4f}")
    print(f"  Reward Fairness: {summary['reward_fairness']:.4f}")
    print(f"  Latency Fairness: {summary['latency_fairness']:.4f}")
    print(f"  Mutual Information: {mi_results.get('normalized_mi', 0):.4f}")

    return summary


def run_message_dimension_ablation(env, config, num_episodes, device, save_dir):
    """Run message dimension ablation study."""
    print("\n" + "=" * 60)
    print("Running Message Dimension Ablation Study")
    print("=" * 60)

    dimensions = [4, 6, 8, 10, 12, 16]
    analyzer = MessageDimensionAnalyzer()

    for msg_dim in dimensions:
        print(f"\n--- Testing message_dim = {msg_dim} ---")

        # Update config
        ablation_config = config.copy()
        ablation_config['message_dim'] = msg_dim

        # Create agents with new message dim
        perception_agents = [
            PerceptionAgent(i, ablation_config, device=device)
            for i in range(ablation_config['num_iot_devices'])
        ]
        orchestration_agents = [
            OrchestrationAgent(i, ablation_config, device=device)
            for i in range(ablation_config['num_edge_servers'])
        ]

        trainer = MAPPO(
            perception_agents, orchestration_agents,
            ablation_config, device
        )

        rewards = []
        update_interval = ablation_config.get('update_interval', 10)
        steps_per_update = ablation_config.get('episode_length', 200) * update_interval

        episode = 0
        pbar = tqdm(total=num_episodes, desc=f"msg_dim={msg_dim}")

        while episode < num_episodes:
            rollout_stats = trainer.collect_rollout(env, steps_per_update)
            trainer.update()

            for ep_reward in rollout_stats.get('episode_rewards', []):
                rewards.append(ep_reward)
                episode += 1
                pbar.update(1)

                if episode >= num_episodes:
                    break

        pbar.close()

        # Evaluate
        metrics = EnhancedMetricsCollector()

        for eval_ep in range(50):
            obs, _ = env.reset()

            for step in range(ablation_config['episode_length']):
                perception_actions = []
                messages = []

                for i, agent in enumerate(perception_agents):
                    action, _ = agent.get_action(obs['perception'][i])
                    perception_actions.append(action)
                    messages.append(agent.encode_message(obs['perception'][i]))

                devices_per_server = ablation_config.get('devices_per_server', 7)
                orchestration_actions = []

                for s, agent in enumerate(orchestration_agents):
                    start_idx = s * devices_per_server
                    end_idx = min(start_idx + devices_per_server, len(messages))
                    server_messages = messages[start_idx:end_idx]

                    action, _ = agent.get_action(
                        obs['orchestration'][s], server_messages
                    )
                    orchestration_actions.append(action)

                actions = {
                    'perception': perception_actions,
                    'orchestration': orchestration_actions
                }
                next_obs, reward_dict, terminated, truncated, info = env.step(actions)

                metrics.add_step(
                    latency=info.get('avg_latency', 0),
                    energy=info.get('avg_energy', 0),
                    accuracy=info.get('avg_accuracy', 0),
                    reward=reward_dict['total'],
                    violation=info.get('deadline_violated', False)
                )

                obs = next_obs

                if terminated or truncated:
                    break

        summary = metrics.get_summary()
        summary['reward'] = float(np.mean(rewards[-100:]))

        analyzer.add_result(msg_dim, summary)

        print(f"  Reward: {summary['reward']:.2f}")
        print(f"  Accuracy: {summary['mean_accuracy']:.4f}")
        print(f"  Latency: {summary['mean_latency']:.2f} ms")

    # Save ablation results
    analyzer.save_results(str(save_dir / 'message_dim_ablation.json'))

    final_summary = analyzer.get_summary()
    print(f"\nOptimal message dimension: {final_summary.get('optimal_dimension', 8)}")

    return final_summary


def run_communication_analysis(env, config, checkpoint_path, device, save_dir):
    """Analyze learned communication from trained model."""
    print("\n" + "=" * 60)
    print("Running Communication Analysis")
    print("=" * 60)

    # Load trained agents
    perception_agents = [
        PerceptionAgent(i, config, device=device)
        for i in range(config['num_iot_devices'])
    ]
    orchestration_agents = [
        OrchestrationAgent(i, config, device=device)
        for i in range(config['num_edge_servers'])
    ]

    # Collect messages and states
    mi_estimator = MutualInformationEstimator()
    attention_analyzer = AttentionAnalyzer()
    all_messages = []
    all_states = []

    for eval_ep in tqdm(range(100), desc="Collecting samples"):
        obs, _ = env.reset()

        for step in range(config['episode_length']):
            for i, agent in enumerate(perception_agents):
                message = agent.encode_message(obs['perception'][i])
                mi_estimator.add_sample(obs['perception'][i], message)
                all_messages.append(message)
                all_states.append(obs['perception'][i])

            # Get orchestration actions to collect attention
            devices_per_server = config.get('devices_per_server', 7)
            messages = [
                agent.encode_message(obs['perception'][i])
                for i, agent in enumerate(perception_agents)
            ]

            for s, agent in enumerate(orchestration_agents):
                start_idx = s * devices_per_server
                end_idx = min(start_idx + devices_per_server, len(messages))
                server_messages = messages[start_idx:end_idx]

                _, action_info = agent.get_action(
                    obs['orchestration'][s], server_messages
                )

                if 'attention_weights' in action_info:
                    weights = action_info['attention_weights'].cpu().numpy()
                    attention_analyzer.add_sample(
                        weights.squeeze(),
                        np.array(server_messages)
                    )

            # Take random action to continue episode
            perception_actions = []
            for i, agent in enumerate(perception_agents):
                action, _ = agent.get_action(obs['perception'][i])
                perception_actions.append(action)

            orchestration_actions = []
            for s, agent in enumerate(orchestration_agents):
                action, _ = agent.get_action(
                    obs['orchestration'][s],
                    messages[s*devices_per_server:(s+1)*devices_per_server]
                )
                orchestration_actions.append(action)

            actions = {
                'perception': perception_actions,
                'orchestration': orchestration_actions
            }
            obs, _, terminated, truncated, _ = env.step(actions)

            if terminated or truncated:
                break

    # Compute analyses
    mi_results = mi_estimator.compute_mi()
    attention_results = attention_analyzer.analyze()

    # Semantic analysis
    messages_array = np.array(all_messages)
    states_array = np.array(all_states)
    semantic_results = analyze_message_semantics(messages_array, states_array)

    results = {
        'mutual_information': mi_results,
        'attention_analysis': attention_results,
        'semantic_analysis': {
            k: v for k, v in semantic_results.items()
            if k not in ['messages_2d', 'clusters']  # Skip large arrays
        }
    }

    # Save results
    results_path = save_dir / 'communication_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

    print(f"\nCommunication Analysis Results:")
    print(f"  Normalized MI: {mi_results.get('normalized_mi', 0):.4f}")
    print(f"  State Entropy: {mi_results.get('state_entropy', 0):.4f}")
    print(f"  Attention Entropy: {attention_results.get('entropy', 0):.4f}")
    print(f"  Attention Concentration: {attention_results.get('concentration', 0):.4f}")

    return results


def main():
    args = parse_args()
    set_seeds(args.seed)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['devices_per_server'] = int(
        np.ceil(config['num_iot_devices'] / config['num_edge_servers'])
    )

    # Create save directory
    save_dir = Path(args.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = save_dir / f"{args.experiment}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create environment
    print("Creating environment...")
    env = IoTEdgeEnv(config)

    all_results = {}

    if args.experiment in ['baselines', 'all']:
        # Run MAPPO with enhanced metrics first
        mappo_results = run_mappo_with_enhanced_metrics(
            env, config, args.episodes, args.device, save_dir
        )
        all_results['mappo'] = mappo_results

        # Run DPTORA baseline
        dptora_results = run_dptora_experiment(
            env, config, args.episodes, args.device, save_dir
        )
        all_results['dptora'] = dptora_results

        # Run MADOA baseline
        madoa_results = run_madoa_experiment(
            env, config, args.episodes, args.device, save_dir
        )
        all_results['madoa'] = madoa_results

    if args.experiment in ['message_dim', 'all']:
        msg_dim_results = run_message_dimension_ablation(
            env, config, args.episodes, args.device, save_dir
        )
        all_results['message_dim_ablation'] = msg_dim_results

    if args.experiment in ['communication_analysis', 'all']:
        comm_results = run_communication_analysis(
            env, config, None, args.device, save_dir
        )
        all_results['communication_analysis'] = comm_results

    # Save all results
    with open(save_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else str(x))

    print(f"\n{'='*60}")
    print(f"All results saved to {save_dir}")
    print(f"{'='*60}")

    # Print comparison table
    if 'mappo' in all_results and 'dptora' in all_results:
        print("\n" + "=" * 80)
        print("BASELINE COMPARISON")
        print("=" * 80)
        print(f"{'Method':<15} {'Reward':>12} {'Latency':>12} {'P95 Lat':>12} {'Accuracy':>12} {'Fairness':>12}")
        print("-" * 80)

        for method in ['mappo', 'dptora', 'madoa']:
            if method in all_results:
                r = all_results[method]
                print(f"{method.upper():<15} "
                      f"{r.get('final_reward', 0):>12.2f} "
                      f"{r.get('mean_latency', 0):>12.2f} "
                      f"{r.get('p95_latency', 0):>12.2f} "
                      f"{r.get('mean_accuracy', 0):>12.4f} "
                      f"{r.get('reward_fairness', 0):>12.4f}")


if __name__ == '__main__':
    main()
