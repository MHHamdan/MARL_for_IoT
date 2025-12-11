"""
Neural Network Architectures for MARL-IoTP

Contains shared network components used by various algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


def init_weights(module: nn.Module, gain: float = 1.0):
    """Initialize module weights using orthogonal initialization."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron network with configurable depth and width.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
        output_activation: Optional[nn.Module] = None,
        layer_norm: bool = False
    ):
        """
        Initialize MLP network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function class
            output_activation: Optional output activation
            layer_norm: Whether to use layer normalization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for i, module in enumerate(self.network):
            if isinstance(module, nn.Linear):
                # Last layer gets smaller initialization
                if i == len(self.network) - 1:
                    init_weights(module, gain=0.01)
                else:
                    init_weights(module, gain=np.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class CentralizedCritic(nn.Module):
    """
    Centralized Critic for CTDE (Centralized Training Decentralized Execution).

    Takes global state (concatenation of all agent observations) and
    outputs a single value estimate.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256],
        layer_norm: bool = True
    ):
        """
        Initialize centralized critic.

        Args:
            state_dim: Global state dimension
            hidden_dims: Hidden layer dimensions
            layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.network = MLPNetwork(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            layer_norm=layer_norm
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            global_state: Global state tensor (batch, state_dim)

        Returns:
            Value tensor (batch, 1)
        """
        return self.network(global_state)


class MultiAgentCritic(nn.Module):
    """
    Multi-Agent Critic that processes observations from all agents.

    Uses attention mechanism to aggregate information across agents.
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        hidden_dim: int = 256,
        num_heads: int = 4
    ):
        """
        Initialize multi-agent critic.

        Args:
            obs_dim: Per-agent observation dimension
            num_agents: Number of agents
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        # Per-agent encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Self-attention for agent interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.agent_encoder:
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))

        for module in self.value_head:
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))
        init_weights(self.value_head[-1], gain=1.0)

    def forward(
        self,
        observations: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            observations: All agent observations (batch, num_agents, obs_dim)
            agent_mask: Optional mask for inactive agents

        Returns:
            Value tensor (batch, 1)
        """
        batch_size = observations.size(0)

        # Encode each agent's observation
        agent_features = self.agent_encoder(observations)  # (batch, num_agents, hidden_dim)

        # Self-attention across agents
        attended, _ = self.self_attention(
            agent_features,
            agent_features,
            agent_features,
            key_padding_mask=agent_mask
        )

        # Global representation via mean pooling
        if agent_mask is not None:
            # Mask out padded agents
            mask_expanded = (~agent_mask).unsqueeze(-1).float()
            attended = attended * mask_expanded
            global_repr = attended.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            global_repr = attended.mean(dim=1)

        # Value prediction
        value = self.value_head(global_repr)

        return value


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous action spaces.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        Initialize Gaussian policy.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Feature extractor
        self.feature_net = MLPNetwork(
            input_dim=obs_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            output_activation=nn.ReLU
        )

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        init_weights(self.mean_head, gain=0.01)
        init_weights(self.log_std_head, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor

        Returns:
            Tuple of (mean, log_std)
        """
        features = self.feature_net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Observation tensor
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)

        if deterministic:
            action = mean
        else:
            noise = torch.randn_like(mean)
            action = mean + std * noise

        # Compute log probability
        log_prob = -0.5 * (
            ((action - mean) / (std + 1e-8)) ** 2 +
            2 * log_std +
            np.log(2 * np.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class CategoricalPolicy(nn.Module):
    """
    Categorical policy network for discrete action spaces.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """
        Initialize categorical policy.

        Args:
            obs_dim: Observation dimension
            num_actions: Number of discrete actions
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        self.network = MLPNetwork(
            input_dim=obs_dim,
            output_dim=num_actions,
            hidden_dims=hidden_dims
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            obs: Observation tensor

        Returns:
            Action logits
        """
        return self.network(obs)

    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Observation tensor
            deterministic: If True, return argmax action

        Returns:
            Tuple of (action, log_prob)
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        # Compute log probability
        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = log_prob.gather(-1, action.unsqueeze(-1))

        return action, log_prob


class DuelingNetwork(nn.Module):
    """
    Dueling network architecture for value-based methods.

    Separates value and advantage streams.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [256, 256]
    ):
        """
        Initialize dueling network.

        Args:
            obs_dim: Observation dimension
            num_actions: Number of actions
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        # Shared feature extractor
        self.feature_net = MLPNetwork(
            input_dim=obs_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            output_activation=nn.ReLU
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, num_actions)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-values.

        Args:
            obs: Observation tensor

        Returns:
            Q-values for all actions
        """
        features = self.feature_net(obs)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine using dueling formula
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values
