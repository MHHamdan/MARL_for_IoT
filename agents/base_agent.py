"""
Base Agent Class for MARL-IoTP

Provides common functionality for all agent types in the multi-agent
reinforcement learning framework.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union


class BaseAgent(ABC):
    """
    Abstract base class for all agents in MARL-IoTP.

    Provides common interface for:
    - Action selection
    - Policy updates
    - State management
    - Checkpointing
    """

    def __init__(
        self,
        agent_id: int,
        agent_type: str,
        config: dict,
        device: str = 'cpu'
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent ('perception' or 'orchestration')
            config: Configuration dictionary
            device: Compute device ('cpu' or 'cuda')
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.device = torch.device(device)

        # Common hyperparameters
        self.hidden_dim = config.get('hidden_dim', 128)
        self.lr = config.get('lr', 3e-4)

        # Networks will be initialized by subclasses
        self.actor: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Training state
        self.training = True
        self.step_count = 0

    @abstractmethod
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Get action from policy.

        Args:
            observation: Current observation
            deterministic: If True, use deterministic policy

        Returns:
            Tuple of (action, action_info dict with log_probs, etc.)
        """
        pass

    @abstractmethod
    def get_value(self, observation: np.ndarray) -> torch.Tensor:
        """
        Get value estimate for observation (if agent has critic).

        Args:
            observation: Current observation

        Returns:
            Value tensor
        """
        pass

    def set_training_mode(self, training: bool = True):
        """Set training mode for networks."""
        self.training = training
        if self.actor is not None:
            self.actor.train(training)

    def to(self, device: Union[str, torch.device]):
        """Move agent to specified device."""
        self.device = torch.device(device)
        if self.actor is not None:
            self.actor.to(self.device)

    def obs_to_tensor(self, observation: np.ndarray) -> torch.Tensor:
        """Convert observation to tensor."""
        if isinstance(observation, np.ndarray):
            return torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        elif isinstance(observation, torch.Tensor):
            return observation.to(self.device)
        else:
            return torch.FloatTensor([observation]).to(self.device)

    def save(self, path: str):
        """Save agent state to file."""
        state = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'config': self.config,
            'step_count': self.step_count,
            'actor_state': self.actor.state_dict() if self.actor else None,
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load agent state from file."""
        state = torch.load(path, map_location=self.device)
        self.step_count = state['step_count']
        if self.actor is not None and state['actor_state'] is not None:
            self.actor.load_state_dict(state['actor_state'])
        if self.optimizer is not None and state['optimizer_state'] is not None:
            self.optimizer.load_state_dict(state['optimizer_state'])

    def get_parameters(self):
        """Get trainable parameters."""
        if self.actor is not None:
            return self.actor.parameters()
        return []

    def update_step_count(self):
        """Increment step counter."""
        self.step_count += 1

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type})"


class ActorCritic(nn.Module):
    """
    Base Actor-Critic network architecture.

    Shared feature extractor with separate actor and critic heads.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialize Actor-Critic network.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            activation: Activation function class
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation()
        )

        # Actor head
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Smaller init for output layers
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor

        Returns:
            Tuple of (action_logits, value)
        """
        features = self.feature_extractor(obs)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features from observation."""
        return self.feature_extractor(obs)


def init_weights(module: nn.Module, gain: float = 1.0):
    """
    Initialize module weights using orthogonal initialization.

    Args:
        module: Module to initialize
        gain: Scaling factor for weights
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
