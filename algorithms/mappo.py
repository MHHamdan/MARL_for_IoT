"""
Multi-Agent Proximal Policy Optimization (MAPPO) for MARL-IoTP

Implements MAPPO with:
- Centralized Training with Decentralized Execution (CTDE)
- Heterogeneous agents (perception + orchestration)
- Generalized Advantage Estimation (GAE)
- Clipped objective function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from agents.perception_agent import PerceptionAgent
from agents.orchestration_agent import OrchestrationAgent
from algorithms.buffer import RolloutBuffer
from algorithms.networks import CentralizedCritic


class MAPPO:
    """
    Multi-Agent PPO trainer for heterogeneous agents.

    Supports both perception agents (on IoT devices) and
    orchestration agents (on edge servers).
    """

    def __init__(
        self,
        perception_agents: List[PerceptionAgent],
        orchestration_agents: List[OrchestrationAgent],
        config: dict,
        device: str = 'cpu'
    ):
        """
        Initialize MAPPO trainer.

        Args:
            perception_agents: List of perception agent instances
            orchestration_agents: List of orchestration agent instances
            config: Training configuration
            device: Compute device
        """
        self.perception_agents = perception_agents
        self.orchestration_agents = orchestration_agents
        self.config = config
        self.device = torch.device(device)

        self.num_perception_agents = len(perception_agents)
        self.num_orchestration_agents = len(orchestration_agents)

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.lr = config.get('lr', 3e-4)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.num_minibatches = config.get('num_minibatches', 4)

        # Centralized critic
        global_state_dim = self._compute_global_state_dim()
        self.centralized_critic = CentralizedCritic(
            state_dim=global_state_dim,
            hidden_dims=[config.get('critic_hidden_dim', 256)] * 2
        ).to(self.device)

        self.critic_optimizer = optim.Adam(
            self.centralized_critic.parameters(),
            lr=self.lr
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=config.get('buffer_size', 2048),
            num_perception_agents=self.num_perception_agents,
            num_orchestration_agents=self.num_orchestration_agents,
            perception_obs_dim=config.get('perception_obs_dim', 16),
            orchestration_obs_dim=config.get('orchestration_obs_dim', 32),
            message_dim=config.get('message_dim', 8),
            devices_per_server=config.get('devices_per_server', 7),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=device
        )

        # Training statistics
        self.training_step = 0

    def _compute_global_state_dim(self) -> int:
        """Compute dimension of global state for centralized critic."""
        perception_dim = self.config.get('perception_obs_dim', 16)
        orchestration_dim = self.config.get('orchestration_obs_dim', 32)

        total_dim = (
            self.num_perception_agents * perception_dim +
            self.num_orchestration_agents * orchestration_dim
        )
        return total_dim

    def get_global_state(
        self,
        perception_obs: List[np.ndarray],
        orchestration_obs: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Construct global state from all agent observations.

        Args:
            perception_obs: List of perception agent observations
            orchestration_obs: List of orchestration agent observations

        Returns:
            Global state tensor
        """
        all_obs = []
        for obs in perception_obs:
            all_obs.append(obs.flatten())
        for obs in orchestration_obs:
            all_obs.append(obs.flatten())

        global_state = np.concatenate(all_obs)
        return torch.FloatTensor(global_state).unsqueeze(0).to(self.device)

    def get_centralized_value(
        self,
        perception_obs: List[np.ndarray],
        orchestration_obs: List[np.ndarray]
    ) -> float:
        """
        Get centralized value estimate.

        Args:
            perception_obs: Perception agent observations
            orchestration_obs: Orchestration agent observations

        Returns:
            Value estimate
        """
        global_state = self.get_global_state(perception_obs, orchestration_obs)
        with torch.no_grad():
            value = self.centralized_critic(global_state)
        return value.item()

    def collect_rollout(
        self,
        env,
        num_steps: int
    ) -> Dict[str, Any]:
        """
        Collect rollout data from environment.

        Args:
            env: Environment instance
            num_steps: Number of steps to collect

        Returns:
            Rollout statistics
        """
        obs, _ = env.reset()
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        for step in range(num_steps):
            # Get perception agent actions
            perception_actions = []
            perception_log_probs = []
            perception_values = []
            messages = []

            for i, agent in enumerate(self.perception_agents):
                action, action_info = agent.get_action(obs['perception'][i])
                value = agent.get_value(obs['perception'][i])

                perception_actions.append(action)
                perception_log_probs.append(action_info['log_prob'].item())
                perception_values.append(value.item())
                messages.append(agent.encode_message(obs['perception'][i]))

            # Organize messages for orchestration agents
            messages_per_server = []
            devices_per_server = self.config.get('devices_per_server', 7)
            for i in range(self.num_orchestration_agents):
                start_idx = i * devices_per_server
                end_idx = min(start_idx + devices_per_server, len(messages))
                server_messages = messages[start_idx:end_idx]
                messages_per_server.append(server_messages)

            # Get orchestration agent actions
            orchestration_actions = []
            orchestration_log_probs = []
            orchestration_values = []

            for i, agent in enumerate(self.orchestration_agents):
                action, action_info = agent.get_action(
                    obs['orchestration'][i],
                    messages_per_server[i]
                )
                value = agent.get_value(
                    obs['orchestration'][i],
                    messages_per_server[i]
                )

                orchestration_actions.append(action)
                orchestration_log_probs.append(action_info['log_prob'].item())
                orchestration_values.append(value.item())

            # Step environment
            actions = {
                'perception': perception_actions,
                'orchestration': orchestration_actions
            }
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Store transition
            self.buffer.store(
                perception_obs=[obs['perception'][i] for i in range(self.num_perception_agents)],
                perception_actions=perception_actions,
                perception_log_probs=perception_log_probs,
                perception_values=perception_values,
                perception_rewards=rewards['perception'],
                orchestration_obs=[obs['orchestration'][i] for i in range(self.num_orchestration_agents)],
                orchestration_messages=messages_per_server,
                orchestration_actions=orchestration_actions,
                orchestration_log_probs=orchestration_log_probs,
                orchestration_values=orchestration_values,
                orchestration_rewards=rewards['orchestration'],
                done=done
            )

            # Track episode stats
            current_episode_reward += rewards['total']
            current_episode_length += 1

            obs = next_obs

            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                obs, _ = env.reset()

        # Compute final values for advantage calculation
        last_perception_values = []
        last_orchestration_values = []

        for i, agent in enumerate(self.perception_agents):
            value = agent.get_value(obs['perception'][i])
            last_perception_values.append(value.item())

        for i, agent in enumerate(self.orchestration_agents):
            server_messages = messages_per_server[i] if messages_per_server else []
            value = agent.get_value(obs['orchestration'][i], server_messages)
            last_orchestration_values.append(value.item())

        # Compute advantages
        self.buffer.compute_advantages(
            last_perception_values,
            last_orchestration_values
        )

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0
        }

    def update(self) -> Dict[str, float]:
        """
        Update all agent policies using collected rollout data.

        Returns:
            Dictionary of training statistics
        """
        self.training_step += 1

        # Get minibatches
        batches = self.buffer.get_batches(
            batch_size=len(self.buffer),
            num_minibatches=self.num_minibatches
        )

        # Track losses
        perception_policy_losses = []
        perception_value_losses = []
        perception_entropy_losses = []
        orchestration_policy_losses = []
        orchestration_value_losses = []
        orchestration_entropy_losses = []

        for epoch in range(self.ppo_epochs):
            for batch in batches:
                # Update perception agents
                p_policy_loss, p_value_loss, p_entropy = self._update_perception_agents(batch)
                perception_policy_losses.append(p_policy_loss)
                perception_value_losses.append(p_value_loss)
                perception_entropy_losses.append(p_entropy)

                # Update orchestration agents
                o_policy_loss, o_value_loss, o_entropy = self._update_orchestration_agents(batch)
                orchestration_policy_losses.append(o_policy_loss)
                orchestration_value_losses.append(o_value_loss)
                orchestration_entropy_losses.append(o_entropy)

        # Clear buffer
        self.buffer.clear()

        return {
            'perception_policy_loss': np.mean(perception_policy_losses),
            'perception_value_loss': np.mean(perception_value_losses),
            'perception_entropy': np.mean(perception_entropy_losses),
            'orchestration_policy_loss': np.mean(orchestration_policy_losses),
            'orchestration_value_loss': np.mean(orchestration_value_losses),
            'orchestration_entropy': np.mean(orchestration_entropy_losses),
            'training_step': self.training_step
        }

    def _update_perception_agents(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float, float]:
        """
        Update perception agent policies.

        Args:
            batch: Batch of training data

        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for i, agent in enumerate(self.perception_agents):
            obs = batch['perception_obs'][:, i, :]
            model_actions = batch['perception_model_actions'][:, i]
            frame_rate_actions = batch['perception_frame_rate_actions'][:, i, :]
            old_log_probs = batch['perception_old_log_probs'][:, i]
            advantages = batch['perception_advantages'][:, i]
            returns = batch['perception_returns'][:, i]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Evaluate actions
            log_probs, entropy, values = agent.evaluate_actions(
                obs, model_actions, frame_rate_actions
            )
            values = values.squeeze(-1)

            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss +
                self.value_loss_coef * value_loss +
                self.entropy_coef * entropy_loss
            )

            # Update
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(agent.actor.parameters()) + list(agent.critic.parameters()),
                self.max_grad_norm
            )
            agent.actor_optimizer.step()
            agent.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        n = self.num_perception_agents
        return total_policy_loss / n, total_value_loss / n, total_entropy / n

    def _update_orchestration_agents(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float, float]:
        """
        Update orchestration agent policies.

        Args:
            batch: Batch of training data

        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for i, agent in enumerate(self.orchestration_agents):
            obs = batch['orchestration_obs'][:, i, :]
            messages = batch['orchestration_messages'][:, i, :, :]
            offload_actions = batch['orchestration_offload_actions'][:, i, :]
            resource_actions = batch['orchestration_resource_actions'][:, i, :]
            bandwidth_actions = batch['orchestration_bandwidth_actions'][:, i, :]
            old_log_probs = batch['orchestration_old_log_probs'][:, i]
            advantages = batch['orchestration_advantages'][:, i]
            returns = batch['orchestration_returns'][:, i]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Evaluate actions
            log_probs, entropy, values = agent.evaluate_actions(
                obs, messages, offload_actions,
                resource_actions, bandwidth_actions
            )
            values = values.squeeze(-1)

            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                policy_loss +
                self.value_loss_coef * value_loss +
                self.entropy_coef * entropy_loss
            )

            # Update
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(agent.actor.parameters()) + list(agent.critic.parameters()),
                self.max_grad_norm
            )
            agent.actor_optimizer.step()
            agent.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        n = self.num_orchestration_agents
        return total_policy_loss / n, total_value_loss / n, total_entropy / n

    def save(self, path: str):
        """Save trainer state."""
        state = {
            'training_step': self.training_step,
            'centralized_critic': self.centralized_critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config
        }

        # Save agent states
        for i, agent in enumerate(self.perception_agents):
            state[f'perception_agent_{i}'] = {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict()
            }

        for i, agent in enumerate(self.orchestration_agents):
            state[f'orchestration_agent_{i}'] = {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict()
            }

        torch.save(state, path)

    def load(self, path: str):
        """Load trainer state."""
        state = torch.load(path, map_location=self.device)

        self.training_step = state['training_step']
        self.centralized_critic.load_state_dict(state['centralized_critic'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])

        # Load agent states
        for i, agent in enumerate(self.perception_agents):
            agent_state = state[f'perception_agent_{i}']
            agent.actor.load_state_dict(agent_state['actor'])
            agent.critic.load_state_dict(agent_state['critic'])
            agent.actor_optimizer.load_state_dict(agent_state['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(agent_state['critic_optimizer'])

        for i, agent in enumerate(self.orchestration_agents):
            agent_state = state[f'orchestration_agent_{i}']
            agent.actor.load_state_dict(agent_state['actor'])
            agent.critic.load_state_dict(agent_state['critic'])
            agent.actor_optimizer.load_state_dict(agent_state['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(agent_state['critic_optimizer'])
