"""
Perception Agent for MARL-IoTP

Perception agents run on IoT devices and are responsible for:
- Selecting appropriate perception models (accuracy vs. speed trade-off)
- Determining frame sampling rates
- Communicating task requirements to orchestration agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Dict, Optional, Any

from agents.base_agent import BaseAgent, init_weights
from agents.communication import MessageEncoder


class PerceptionActorNetwork(nn.Module):
    """
    Actor network for perception agents.

    Handles hybrid action space:
    - Discrete: Model selection
    - Continuous: Frame rate
    """

    def __init__(
        self,
        obs_dim: int,
        num_models: int,
        continuous_dim: int = 1,
        hidden_dim: int = 128,
        message_dim: int = 8
    ):
        """
        Initialize perception actor network.

        Args:
            obs_dim: Observation dimension
            num_models: Number of perception models to choose from
            continuous_dim: Dimension of continuous actions
            hidden_dim: Hidden layer dimension
            message_dim: Message dimension for communication
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.num_models = num_models
        self.continuous_dim = continuous_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Discrete action head (model selection)
        self.model_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_models)
        )

        # Continuous action head (frame rate)
        self.frame_rate_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, continuous_dim)
        )
        self.frame_rate_log_std = nn.Parameter(
            torch.zeros(continuous_dim)
        )

        # Message encoder for communication
        self.message_encoder = MessageEncoder(
            obs_dim=obs_dim,
            message_dim=message_dim,
            hidden_dim=hidden_dim // 2
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))

        for module in self.model_head:
            if isinstance(module, nn.Linear):
                init_weights(module, gain=0.01)

        for module in self.frame_rate_mean:
            if isinstance(module, nn.Linear):
                init_weights(module, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action distributions.

        Args:
            obs: Observation tensor (batch, obs_dim)

        Returns:
            Tuple of (model_logits, frame_rate_mean, frame_rate_log_std)
        """
        # Sanitize input - replace NaN/Inf with zeros
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # Extract features
        features = self.encoder(obs)

        # Discrete action distribution
        model_logits = self.model_head(features)
        # Clamp logits for stability
        model_logits = torch.clamp(model_logits, -10, 10)

        # Continuous action distribution
        frame_rate_mean = self.frame_rate_mean(features)
        # Sigmoid to bound mean in [0.1, 1.0]
        frame_rate_mean = 0.1 + 0.9 * torch.sigmoid(frame_rate_mean)

        return model_logits, frame_rate_mean, self.frame_rate_log_std

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features for value function."""
        return self.encoder(obs)

    def encode_message(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation into message for communication."""
        return self.message_encoder(obs)


class PerceptionCriticNetwork(nn.Module):
    """
    Critic network for perception agents.

    Used for local value estimation during training.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128
    ):
        """
        Initialize perception critic network.

        Args:
            obs_dim: Observation dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))
        # Last layer with smaller gain
        init_weights(self.network[-1], gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning value estimate.

        Args:
            obs: Observation tensor

        Returns:
            Value tensor
        """
        # Sanitize input
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.network(obs)


class PerceptionAgent(BaseAgent):
    """
    Perception Agent for IoT devices.

    Responsibilities:
    - Select appropriate perception model (accuracy vs. speed trade-off)
    - Determine frame sampling rate
    - Communicate task requirements to orchestration agents
    """

    def __init__(
        self,
        agent_id: int,
        config: dict,
        device: str = 'cpu'
    ):
        """
        Initialize perception agent.

        Args:
            agent_id: Unique agent identifier
            config: Configuration dictionary
            device: Compute device
        """
        super().__init__(
            agent_id=agent_id,
            agent_type='perception',
            config=config,
            device=device
        )

        # Observation dimension
        self.obs_dim = config.get('perception_obs_dim', 16)

        # Action dimensions
        self.num_models = config.get('num_perception_models', 4)
        self.continuous_action_dim = 1  # frame_rate

        # Message dimension
        self.message_dim = config.get('message_dim', 8)

        # Initialize networks
        self.actor = PerceptionActorNetwork(
            obs_dim=self.obs_dim,
            num_models=self.num_models,
            continuous_dim=self.continuous_action_dim,
            hidden_dim=self.hidden_dim,
            message_dim=self.message_dim
        ).to(self.device)

        self.critic = PerceptionCriticNetwork(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr
        )

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """
        Get action from policy.

        Args:
            observation: Current observation
            deterministic: If True, use deterministic policy

        Returns:
            Tuple of (action dict, action info dict)
        """
        obs_tensor = self.obs_to_tensor(observation)

        with torch.no_grad():
            model_logits, frame_rate_mean, frame_rate_log_std = self.actor(
                obs_tensor
            )

        # Discrete action (model selection)
        model_probs = F.softmax(model_logits, dim=-1)
        model_dist = Categorical(model_probs)

        if deterministic:
            model_action = model_probs.argmax(dim=-1)
        else:
            model_action = model_dist.sample()

        # Continuous action (frame rate)
        frame_rate_std = torch.exp(frame_rate_log_std)
        frame_rate_dist = Normal(frame_rate_mean, frame_rate_std)

        if deterministic:
            frame_rate_action = frame_rate_mean
        else:
            frame_rate_action = frame_rate_dist.sample()
            # Clip to valid range
            frame_rate_action = torch.clamp(frame_rate_action, 0.1, 1.0)

        # Compute log probabilities
        model_log_prob = model_dist.log_prob(model_action)
        frame_rate_log_prob = frame_rate_dist.log_prob(frame_rate_action).sum(dim=-1)
        total_log_prob = model_log_prob + frame_rate_log_prob

        # Compute entropy
        model_entropy = model_dist.entropy()
        frame_rate_entropy = frame_rate_dist.entropy().sum(dim=-1)
        total_entropy = model_entropy + frame_rate_entropy

        # Build action dictionary
        action = {
            'model_selection': model_action.cpu().numpy().item(),
            'frame_rate': frame_rate_action.cpu().numpy().flatten()
        }

        # Action info
        action_info = {
            'log_prob': total_log_prob,
            'entropy': total_entropy,
            'model_log_prob': model_log_prob,
            'frame_rate_log_prob': frame_rate_log_prob,
            'model_probs': model_probs
        }

        return action, action_info

    def get_action_and_value(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get action and value from policy.

        Args:
            observation: Current observation
            deterministic: If True, use deterministic policy

        Returns:
            Tuple of (action, action_info, value)
        """
        action, action_info = self.get_action(observation, deterministic)
        value = self.get_value(observation)
        return action, action_info, value

    def get_value(self, observation: np.ndarray) -> torch.Tensor:
        """
        Get value estimate for observation.

        Args:
            observation: Current observation

        Returns:
            Value tensor
        """
        obs_tensor = self.obs_to_tensor(observation)
        with torch.no_grad():
            value = self.critic(obs_tensor)
        return value

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        model_actions: torch.Tensor,
        frame_rate_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            observations: Batch of observations
            model_actions: Batch of model selection actions
            frame_rate_actions: Batch of frame rate actions

        Returns:
            Tuple of (log_probs, entropy, values)
        """
        model_logits, frame_rate_mean, frame_rate_log_std = self.actor(
            observations
        )

        # Clamp logits for numerical stability
        model_logits = torch.clamp(model_logits, -20, 20)

        # Discrete distribution
        model_probs = F.softmax(model_logits, dim=-1)
        # Add small epsilon to avoid zero probabilities
        model_probs = model_probs + 1e-8
        model_probs = model_probs / model_probs.sum(dim=-1, keepdim=True)
        model_dist = Categorical(model_probs)

        # Continuous distribution
        frame_rate_log_std = torch.clamp(frame_rate_log_std, -5, 2)
        frame_rate_std = torch.exp(frame_rate_log_std)
        frame_rate_dist = Normal(frame_rate_mean, frame_rate_std + 1e-8)

        # Log probabilities
        model_log_prob = model_dist.log_prob(model_actions)
        frame_rate_log_prob = frame_rate_dist.log_prob(frame_rate_actions).sum(dim=-1)
        total_log_prob = model_log_prob + frame_rate_log_prob

        # Entropy
        model_entropy = model_dist.entropy()
        frame_rate_entropy = frame_rate_dist.entropy().sum(dim=-1)
        total_entropy = model_entropy + frame_rate_entropy

        # Values
        values = self.critic(observations)

        return total_log_prob, total_entropy, values

    def encode_message(self, observation: np.ndarray) -> np.ndarray:
        """
        Encode message to send to orchestration agents.

        Args:
            observation: Current observation

        Returns:
            Message array
        """
        obs_tensor = self.obs_to_tensor(observation)
        with torch.no_grad():
            message = self.actor.encode_message(obs_tensor)
        return message.cpu().numpy().flatten()

    def save(self, path: str):
        """Save agent state."""
        state = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'config': self.config,
            'step_count': self.step_count,
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict()
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load agent state."""
        state = torch.load(path, map_location=self.device)
        self.step_count = state['step_count']
        self.actor.load_state_dict(state['actor_state'])
        self.critic.load_state_dict(state['critic_state'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer_state'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer_state'])
