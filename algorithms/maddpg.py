"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for MARL-IoTP

Baseline implementation for comparison with MAPPO.
Off-policy algorithm with centralized critic and decentralized actors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

from algorithms.buffer import ReplayBuffer
from algorithms.networks import MLPNetwork


class MADDPGActor(nn.Module):
    """Actor network for MADDPG."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        action_bound: float = 1.0
    ):
        super().__init__()

        self.action_bound = action_bound
        self.network = MLPNetwork(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            output_activation=nn.Tanh
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs) * self.action_bound


class MADDPGCritic(nn.Module):
    """Centralized critic for MADDPG."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()

        # Takes all observations and actions
        input_dim = (obs_dim + action_dim) * num_agents

        self.network = MLPNetwork(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims
        )

    def forward(
        self,
        observations: List[torch.Tensor],
        actions: List[torch.Tensor]
    ) -> torch.Tensor:
        # Concatenate all observations and actions
        inputs = []
        for obs, act in zip(observations, actions):
            inputs.extend([obs, act])
        x = torch.cat(inputs, dim=-1)
        return self.network(x)


class MADDPGAgent:
    """Individual agent in MADDPG."""

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        total_obs_dim: int,
        total_action_dim: int,
        num_agents: int,
        config: dict,
        device: str = 'cpu'
    ):
        self.agent_id = agent_id
        self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.noise_std = config.get('noise_std', 0.1)
        self.noise_clip = config.get('noise_clip', 0.5)

        # Actor networks
        self.actor = MADDPGActor(obs_dim, action_dim).to(self.device)
        self.actor_target = deepcopy(self.actor)

        # Critic networks (centralized)
        self.critic = MADDPGCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents
        ).to(self.device)
        self.critic_target = deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.get('lr_actor', 1e-4)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.get('lr_critic', 1e-3)
        )

    def get_action(
        self,
        obs: np.ndarray,
        add_noise: bool = True
    ) -> np.ndarray:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(obs_tensor)

        action = action.cpu().numpy().squeeze()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            action = np.clip(action + noise, -1, 1)

        return action

    def soft_update(self, tau: float = 0.01):
        """Soft update target networks."""
        for target_param, param in zip(
            self.actor_target.parameters(),
            self.actor.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )


class MADDPG:
    """
    Multi-Agent DDPG trainer.

    Baseline algorithm for comparison with MAPPO.
    """

    def __init__(
        self,
        num_agents: int,
        obs_dims: List[int],
        action_dims: List[int],
        config: dict,
        device: str = 'cpu'
    ):
        """
        Initialize MADDPG trainer.

        Args:
            num_agents: Number of agents
            obs_dims: Observation dimensions per agent
            action_dims: Action dimensions per agent
            config: Training configuration
            device: Compute device
        """
        self.num_agents = num_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.config = config
        self.device = torch.device(device)

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.batch_size = config.get('batch_size', 256)
        self.warmup_steps = config.get('warmup_steps', 10000)
        self.policy_update_freq = config.get('policy_update_freq', 2)

        # Total dimensions
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)

        # Initialize agents
        self.agents = []
        for i in range(num_agents):
            agent = MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                total_obs_dim=total_obs_dim,
                total_action_dim=total_action_dim,
                num_agents=num_agents,
                config=config,
                device=device
            )
            self.agents.append(agent)

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=config.get('buffer_size', 100000),
            num_agents=num_agents,
            obs_dims=obs_dims,
            action_dims=action_dims,
            device=device
        )

        self.training_step = 0
        self.update_count = 0

    def get_actions(
        self,
        observations: List[np.ndarray],
        add_noise: bool = True
    ) -> List[np.ndarray]:
        """
        Get actions for all agents.

        Args:
            observations: List of observations per agent
            add_noise: Whether to add exploration noise

        Returns:
            List of actions per agent
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.get_action(observations[i], add_noise)
            actions.append(action)
        return actions

    def store_transition(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        dones: List[bool]
    ):
        """Store transition in replay buffer."""
        self.buffer.store(
            observations, actions, rewards,
            next_observations, dones
        )
        self.training_step += 1

    def update(self) -> Dict[str, float]:
        """
        Update all agents.

        Returns:
            Dictionary of training statistics
        """
        if len(self.buffer) < self.warmup_steps:
            return {'status': 'warming_up'}

        self.update_count += 1

        # Sample batch
        batch = self.buffer.sample(self.batch_size)

        # Update each agent
        critic_losses = []
        actor_losses = []

        for i, agent in enumerate(self.agents):
            # Update critic
            critic_loss = self._update_critic(agent, i, batch)
            critic_losses.append(critic_loss)

            # Update actor (less frequently)
            if self.update_count % self.policy_update_freq == 0:
                actor_loss = self._update_actor(agent, i, batch)
                actor_losses.append(actor_loss)

                # Soft update targets
                agent.soft_update(self.tau)

        return {
            'critic_loss': np.mean(critic_losses),
            'actor_loss': np.mean(actor_losses) if actor_losses else 0,
            'update_count': self.update_count
        }

    def _update_critic(
        self,
        agent: MADDPGAgent,
        agent_idx: int,
        batch: Dict[str, List[torch.Tensor]]
    ) -> float:
        """Update agent's critic."""
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards'][agent_idx]
        next_obs = batch['next_obs']
        dones = batch['dones'][agent_idx]

        # Get target actions
        with torch.no_grad():
            target_actions = []
            for i, ag in enumerate(self.agents):
                target_action = ag.actor_target(next_obs[i])
                target_actions.append(target_action)

            # Target Q-value
            target_q = agent.critic_target(next_obs, target_actions)
            target_q = rewards.unsqueeze(-1) + (
                self.gamma * (1 - dones.unsqueeze(-1)) * target_q
            )

        # Current Q-value
        current_q = agent.critic(obs, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(
        self,
        agent: MADDPGAgent,
        agent_idx: int,
        batch: Dict[str, List[torch.Tensor]]
    ) -> float:
        """Update agent's actor."""
        obs = batch['obs']

        # Get current actions for all agents
        current_actions = []
        for i, ag in enumerate(self.agents):
            if i == agent_idx:
                # Use differentiable action from current actor
                current_actions.append(agent.actor(obs[i]))
            else:
                # Use detached actions from other agents
                with torch.no_grad():
                    current_actions.append(ag.actor(obs[i]))

        # Actor loss (maximize Q-value)
        actor_loss = -agent.critic(obs, current_actions).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

        return actor_loss.item()

    def save(self, path: str):
        """Save trainer state."""
        state = {
            'training_step': self.training_step,
            'update_count': self.update_count,
            'config': self.config
        }

        for i, agent in enumerate(self.agents):
            state[f'agent_{i}'] = {
                'actor': agent.actor.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critic': agent.critic.state_dict(),
                'critic_target': agent.critic_target.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict()
            }

        torch.save(state, path)

    def load(self, path: str):
        """Load trainer state."""
        state = torch.load(path, map_location=self.device)

        self.training_step = state['training_step']
        self.update_count = state['update_count']

        for i, agent in enumerate(self.agents):
            agent_state = state[f'agent_{i}']
            agent.actor.load_state_dict(agent_state['actor'])
            agent.actor_target.load_state_dict(agent_state['actor_target'])
            agent.critic.load_state_dict(agent_state['critic'])
            agent.critic_target.load_state_dict(agent_state['critic_target'])
            agent.actor_optimizer.load_state_dict(agent_state['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(agent_state['critic_optimizer'])
