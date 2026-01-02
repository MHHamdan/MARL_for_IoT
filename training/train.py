#!/usr/bin/env python3
"""
Main Training Script for MARL-IoTP

Usage:
    python scripts/train.py --config configs/mappo.yaml
    python scripts/train.py --num_devices 30 --num_servers 4 --episodes 15000
"""

import argparse
import yaml
import torch
import numpy as np
import random
import sys
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
from utils.metrics import MetricsTracker, compute_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MARL-IoTP')

    # Config file
    parser.add_argument(
        '--config', type=str, default='configs/mappo.yaml',
        help='Path to configuration file'
    )

    # Override options
    parser.add_argument('--num_devices', type=int, default=None)
    parser.add_argument('--num_servers', type=int, default=None)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/')

    # Training options
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)

    # Device
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Logging
    parser.add_argument('--wandb', action='store_true', help='Use W&B logging')
    parser.add_argument('--experiment_name', type=str, default=None)

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agents(config: dict, device: str):
    """Create perception and orchestration agents."""
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

    return perception_agents, orchestration_agents


def evaluate(
    env: IoTEdgeEnv,
    perception_agents: list,
    orchestration_agents: list,
    config: dict,
    num_episodes: int = 10
) -> dict:
    """
    Run evaluation episodes.

    Args:
        env: Environment instance
        perception_agents: List of perception agents
        orchestration_agents: List of orchestration agents
        config: Configuration
        num_episodes: Number of evaluation episodes

    Returns:
        Evaluation metrics
    """
    metrics = {
        'reward': [],
        'latency': [],
        'energy': [],
        'accuracy': [],
        'deadline_violations': []
    }

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_metrics = {'latency': [], 'energy': [], 'accuracy': [], 'violations': 0}

        for step in range(config['episode_length']):
            # Get perception actions (deterministic)
            perception_actions = []
            messages = []
            for i, agent in enumerate(perception_agents):
                action, _ = agent.get_action(obs['perception'][i], deterministic=True)
                perception_actions.append(action)
                messages.append(agent.encode_message(obs['perception'][i]))

            # Organize messages
            devices_per_server = config.get('devices_per_server', 7)
            messages_per_server = []
            for i in range(len(orchestration_agents)):
                start_idx = i * devices_per_server
                end_idx = min(start_idx + devices_per_server, len(messages))
                messages_per_server.append(messages[start_idx:end_idx])

            # Get orchestration actions (deterministic)
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
            episode_metrics['latency'].append(info.get('avg_latency', 0))
            episode_metrics['energy'].append(info.get('avg_energy', 0))
            episode_metrics['accuracy'].append(info.get('avg_accuracy', 0))
            episode_metrics['violations'] += info.get('deadline_violations', 0)

            obs = next_obs

            if terminated or truncated:
                break

        metrics['reward'].append(episode_reward)
        metrics['latency'].append(np.mean(episode_metrics['latency']))
        metrics['energy'].append(np.mean(episode_metrics['energy']))
        metrics['accuracy'].append(np.mean(episode_metrics['accuracy']))
        metrics['deadline_violations'].append(episode_metrics['violations'])

    return {
        'mean_reward': np.mean(metrics['reward']),
        'std_reward': np.std(metrics['reward']),
        'mean_latency': np.mean(metrics['latency']),
        'mean_energy': np.mean(metrics['energy']),
        'mean_accuracy': np.mean(metrics['accuracy']),
        'mean_violations': np.mean(metrics['deadline_violations'])
    }


def save_checkpoint(
    trainer: MAPPO,
    perception_agents: list,
    orchestration_agents: list,
    episode: int,
    save_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint_dir = save_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save trainer state
    trainer.save(str(checkpoint_dir / f'trainer_ep{episode}.pt'))

    if is_best:
        trainer.save(str(checkpoint_dir / 'trainer_best.pt'))

    # Save agent states
    for i, agent in enumerate(perception_agents):
        agent.save(str(checkpoint_dir / f'perception_agent_{i}_ep{episode}.pt'))
        if is_best:
            agent.save(str(checkpoint_dir / f'perception_agent_{i}_best.pt'))

    for i, agent in enumerate(orchestration_agents):
        agent.save(str(checkpoint_dir / f'orchestration_agent_{i}_ep{episode}.pt'))
        if is_best:
            agent.save(str(checkpoint_dir / f'orchestration_agent_{i}_best.pt'))


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.num_devices is not None:
        config['num_iot_devices'] = args.num_devices
    if args.num_servers is not None:
        config['num_edge_servers'] = args.num_servers

    # Update devices per server
    config['devices_per_server'] = int(
        np.ceil(config['num_iot_devices'] / config['num_edge_servers'])
    )

    # Set seed
    set_seed(args.seed)

    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = (
            f"mappo_{config['num_iot_devices']}dev_"
            f"{config['num_edge_servers']}srv"
        )

    # Initialize logger
    logger = Logger(
        log_dir=args.save_dir,
        experiment_name=args.experiment_name,
        use_tensorboard=True
    )
    logger.log_config(config)

    # Create environment
    logger.info("Creating environment...")
    env = IoTEdgeEnv(config)

    # Create agents
    logger.info("Creating agents...")
    perception_agents, orchestration_agents = create_agents(config, args.device)
    logger.info(
        f"Created {len(perception_agents)} perception agents and "
        f"{len(orchestration_agents)} orchestration agents"
    )

    # Create trainer
    logger.info("Initializing MAPPO trainer...")
    trainer = MAPPO(
        perception_agents=perception_agents,
        orchestration_agents=orchestration_agents,
        config=config,
        device=args.device
    )

    # Metrics tracking
    metrics_tracker = MetricsTracker(window_size=100)
    best_reward = -float('inf')

    # Training loop
    logger.info(f"Starting training for {args.episodes} episodes...")
    update_interval = config.get('update_interval', 10)
    steps_per_update = config.get('episode_length', 200) * update_interval

    episode = 0
    pbar = tqdm(total=args.episodes, desc="Training")

    while episode < args.episodes:
        # Collect rollout
        rollout_stats = trainer.collect_rollout(env, steps_per_update)

        # Update policy
        train_stats = trainer.update()

        # Log training info
        for ep_reward in rollout_stats.get('episode_rewards', []):
            episode += 1
            pbar.update(1)
            metrics_tracker.add('reward', ep_reward)

            if episode % args.log_interval == 0:
                logger.log_episode(
                    episode=episode,
                    reward=ep_reward,
                    metrics={
                        'mean_reward': metrics_tracker.get('reward').recent_mean(),
                        **{k: v for k, v in train_stats.items() if k != 'training_step'}
                    }
                )

        # Log training stats
        logger.log_training(train_stats, trainer.training_step)

        # Evaluation
        if episode % args.eval_interval == 0:
            logger.info(f"Running evaluation at episode {episode}...")
            eval_metrics = evaluate(
                env, perception_agents, orchestration_agents, config
            )
            logger.log_evaluation(eval_metrics, episode)

            # Track best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                save_checkpoint(
                    trainer, perception_agents, orchestration_agents,
                    episode, Path(logger.log_dir), is_best=True
                )
                logger.info(f"New best reward: {best_reward:.2f}")

        # Save checkpoint
        if episode % args.save_interval == 0:
            save_checkpoint(
                trainer, perception_agents, orchestration_agents,
                episode, Path(logger.log_dir)
            )

    pbar.close()

    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = evaluate(
        env, perception_agents, orchestration_agents, config, num_episodes=20
    )
    logger.log_evaluation(final_metrics, episode)
    logger.info(f"Final evaluation: {final_metrics}")

    # Save final model
    save_checkpoint(
        trainer, perception_agents, orchestration_agents,
        episode, Path(logger.log_dir)
    )

    # Save metrics summary
    summary = metrics_tracker.get_summary()
    logger.info(f"Training complete. Summary: {summary}")
    logger.close()

    print("\nTraining complete!")
    print(f"Results saved to: {logger.log_dir}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final metrics: {final_metrics}")


if __name__ == '__main__':
    main()
