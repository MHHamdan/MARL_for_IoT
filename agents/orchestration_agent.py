"""
Orchestration Agent for MARL-IoTP

Orchestration agents run on edge servers and are responsible for:
- Task offloading decisions (local, edge, cloud)
- Computation resource allocation
- Bandwidth allocation
- Coordinating with multiple perception agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Dict, List, Optional, Any

from agents.base_agent import BaseAgent, init_weights
from agents.communication import MessageAggregator


class OrchestrationActorNetwork(nn.Module):
    """
    Actor network for orchestration agents.

    Features:
    - Attention mechanism for processing messages from multiple devices
    - Hybrid action space for discrete offloading and continuous allocation
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        num_devices: int,
        num_offload_options: int = 3,
        continuous_dim: int = 2,
        hidden_dim: int = 256,
        num_attention_heads: int = 2
    ):
        """
        Initialize orchestration actor network.

        Args:
            obs_dim: Observation dimension
            message_dim: Message dimension from perception agents
            num_devices: Maximum number of connected devices
            num_offload_options: Number of offload options (local, edge, cloud)
            continuous_dim: Dimension of continuous actions (resource, bandwidth)
            hidden_dim: Hidden layer dimension
            num_attention_heads: Number of attention heads
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.num_devices = num_devices
        self.num_offload_options = num_offload_options
        self.continuous_dim = continuous_dim
        self.hidden_dim = hidden_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Message processing with attention
        self.message_proj = nn.Linear(message_dim, hidden_dim // 2)
        self.message_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=num_attention_heads,
            batch_first=True
        )

        # Query projection for attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # Combined feature processing
        combined_dim = hidden_dim + hidden_dim // 2
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Discrete action heads (one per device for offload decision)
        self.offload_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_offload_options)
        )

        # Continuous action heads
        self.resource_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.bandwidth_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.resource_log_std = nn.Parameter(torch.zeros(1))
        self.bandwidth_log_std = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in [self.obs_encoder, self.feature_combiner]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    init_weights(layer, gain=np.sqrt(2))

        for module in [self.offload_head, self.resource_mean, self.bandwidth_mean]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    init_weights(layer, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        messages: torch.Tensor,
        message_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observation tensor (batch, obs_dim)
            messages: Messages from devices (batch, num_devices, message_dim)
            message_mask: Mask for invalid messages (batch, num_devices)

        Returns:
            Dictionary with action distributions
        """
        # Sanitize inputs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        messages = torch.nan_to_num(messages, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size = obs.size(0)

        # Encode observation
        obs_features = self.obs_encoder(obs)  # (batch, hidden_dim)

        # Process messages
        msg_projected = self.message_proj(messages)  # (batch, num_devices, hidden_dim//2)

        # Create query from observation
        query = self.query_proj(obs_features).unsqueeze(1)  # (batch, 1, hidden_dim//2)

        # Attention over messages
        attended_msg, attn_weights = self.message_attention(
            query=query,
            key=msg_projected,
            value=msg_projected,
            key_padding_mask=message_mask
        )
        attended_msg = attended_msg.squeeze(1)  # (batch, hidden_dim//2)

        # Combine features
        combined = torch.cat([obs_features, attended_msg], dim=-1)
        combined_features = self.feature_combiner(combined)

        # Offload decisions per device
        # We need to produce logits for each device
        offload_logits_list = []
        for i in range(self.num_devices):
            # Use message for this device as additional context
            device_msg = msg_projected[:, i, :]  # (batch, hidden_dim//2)
            device_context = torch.cat([combined_features, device_msg], dim=-1)
            device_logits = self.offload_head(device_context)
            offload_logits_list.append(device_logits)

        offload_logits = torch.stack(offload_logits_list, dim=1)  # (batch, num_devices, num_options)
        # Clamp logits for stability
        offload_logits = torch.clamp(offload_logits, -10, 10)

        # Resource allocation
        resource_mean = torch.sigmoid(self.resource_mean(combined_features))
        bandwidth_mean = torch.sigmoid(self.bandwidth_mean(combined_features))

        return {
            'offload_logits': offload_logits,
            'resource_mean': resource_mean,
            'resource_log_std': self.resource_log_std,
            'bandwidth_mean': bandwidth_mean,
            'bandwidth_log_std': self.bandwidth_log_std,
            'attention_weights': attn_weights
        }

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features from observation."""
        return self.obs_encoder(obs)


class OrchestrationCriticNetwork(nn.Module):
    """
    Critic network for orchestration agents.
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        num_devices: int,
        hidden_dim: int = 256
    ):
        """
        Initialize orchestration critic network.

        Args:
            obs_dim: Observation dimension
            message_dim: Message dimension
            num_devices: Number of connected devices
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Input includes observation and aggregated message info
        input_dim = obs_dim + message_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Message aggregator
        self.message_agg = nn.Sequential(
            nn.Linear(message_dim * num_devices, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))
        init_weights(self.network[-1], gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
        messages: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: Observation tensor (batch, obs_dim)
            messages: Messages tensor (batch, num_devices, message_dim)

        Returns:
            Value tensor
        """
        # Sanitize inputs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        messages = torch.nan_to_num(messages, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size = obs.size(0)

        # Flatten and aggregate messages
        msg_flat = messages.view(batch_size, -1)
        msg_agg = self.message_agg(msg_flat)

        # Combine with observation
        combined = torch.cat([obs, msg_agg], dim=-1)

        return self.network(combined)


class OrchestrationAgent(BaseAgent):
    """
    Orchestration Agent for Edge servers.

    Responsibilities:
    - Decide task offloading (local, edge, cloud)
    - Allocate computation resources
    - Manage bandwidth allocation
    - Coordinate with multiple perception agents
    """

    def __init__(
        self,
        agent_id: int,
        config: dict,
        device: str = 'cpu'
    ):
        """
        Initialize orchestration agent.

        Args:
            agent_id: Unique agent identifier
            config: Configuration dictionary
            device: Compute device
        """
        super().__init__(
            agent_id=agent_id,
            agent_type='orchestration',
            config=config,
            device=device
        )

        # Dimensions
        self.obs_dim = config.get('orchestration_obs_dim', 32)
        self.message_dim = config.get('message_dim', 8)
        self.num_connected_devices = config.get('devices_per_server', 7)

        # Action dimensions
        self.num_offload_options = 3  # local, edge, cloud
        self.continuous_action_dim = 2  # resource + bandwidth allocation

        # Initialize networks
        self.actor = OrchestrationActorNetwork(
            obs_dim=self.obs_dim,
            message_dim=self.message_dim,
            num_devices=self.num_connected_devices,
            num_offload_options=self.num_offload_options,
            continuous_dim=self.continuous_action_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_attention_heads=2
        ).to(self.device)

        self.critic = OrchestrationCriticNetwork(
            obs_dim=self.obs_dim,
            message_dim=self.message_dim,
            num_devices=self.num_connected_devices,
            hidden_dim=config.get('critic_hidden_dim', 256)
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
        messages: Optional[List[np.ndarray]] = None,
        deterministic: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """
        Get action from policy.

        Args:
            observation: Current observation
            messages: Messages from connected perception agents
            deterministic: If True, use deterministic policy

        Returns:
            Tuple of (action dict, action info dict)
        """
        obs_tensor = self.obs_to_tensor(observation)

        # Process messages
        if messages is None:
            messages = [np.zeros(self.message_dim) for _ in range(self.num_connected_devices)]

        # Pad messages if needed
        while len(messages) < self.num_connected_devices:
            messages.append(np.zeros(self.message_dim))

        msg_tensor = torch.FloatTensor(np.array(messages)).unsqueeze(0).to(self.device)

        # Create mask for padded messages
        message_mask = None

        with torch.no_grad():
            action_dict = self.actor(obs_tensor, msg_tensor, message_mask)

        # Sample actions
        offload_logits = action_dict['offload_logits']
        resource_mean = action_dict['resource_mean']
        bandwidth_mean = action_dict['bandwidth_mean']
        resource_log_std = action_dict['resource_log_std']
        bandwidth_log_std = action_dict['bandwidth_log_std']

        # Discrete actions (offload decisions)
        offload_probs = F.softmax(offload_logits, dim=-1)
        offload_dists = [Categorical(offload_probs[:, i, :]) for i in range(self.num_connected_devices)]

        if deterministic:
            offload_actions = [probs.argmax(dim=-1) for probs in offload_probs.unbind(dim=1)]
        else:
            offload_actions = [dist.sample() for dist in offload_dists]

        offload_actions_tensor = torch.stack(offload_actions, dim=1)

        # Continuous actions
        resource_std = torch.exp(resource_log_std)
        bandwidth_std = torch.exp(bandwidth_log_std)

        resource_dist = Normal(resource_mean, resource_std)
        bandwidth_dist = Normal(bandwidth_mean, bandwidth_std)

        if deterministic:
            resource_action = resource_mean
            bandwidth_action = bandwidth_mean
        else:
            resource_action = torch.clamp(resource_dist.sample(), 0.0, 1.0)
            bandwidth_action = torch.clamp(bandwidth_dist.sample(), 0.0, 1.0)

        # Compute log probabilities
        offload_log_probs = sum([
            dist.log_prob(action)
            for dist, action in zip(offload_dists, offload_actions)
        ])
        resource_log_prob = resource_dist.log_prob(resource_action).sum(dim=-1)
        bandwidth_log_prob = bandwidth_dist.log_prob(bandwidth_action).sum(dim=-1)
        total_log_prob = offload_log_probs + resource_log_prob + bandwidth_log_prob

        # Compute entropy
        offload_entropy = sum([dist.entropy() for dist in offload_dists])
        resource_entropy = resource_dist.entropy().sum(dim=-1)
        bandwidth_entropy = bandwidth_dist.entropy().sum(dim=-1)
        total_entropy = offload_entropy + resource_entropy + bandwidth_entropy

        # Build action dictionary
        action = {
            'offload_decisions': offload_actions_tensor.cpu().numpy().flatten(),
            'resource_allocation': resource_action.cpu().numpy().flatten(),
            'bandwidth_allocation': bandwidth_action.cpu().numpy().flatten()
        }

        # Action info
        action_info = {
            'log_prob': total_log_prob,
            'entropy': total_entropy,
            'offload_probs': offload_probs,
            'attention_weights': action_dict['attention_weights']
        }

        return action, action_info

    def get_value(
        self,
        observation: np.ndarray,
        messages: Optional[List[np.ndarray]] = None
    ) -> torch.Tensor:
        """
        Get value estimate for observation.

        Args:
            observation: Current observation
            messages: Messages from connected perception agents

        Returns:
            Value tensor
        """
        obs_tensor = self.obs_to_tensor(observation)

        # Process messages
        if messages is None:
            messages = [np.zeros(self.message_dim) for _ in range(self.num_connected_devices)]

        while len(messages) < self.num_connected_devices:
            messages.append(np.zeros(self.message_dim))

        msg_tensor = torch.FloatTensor(np.array(messages)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.critic(obs_tensor, msg_tensor)

        return value

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        messages: torch.Tensor,
        offload_actions: torch.Tensor,
        resource_actions: torch.Tensor,
        bandwidth_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            observations: Batch of observations
            messages: Batch of messages
            offload_actions: Batch of offload decisions
            resource_actions: Batch of resource allocations
            bandwidth_actions: Batch of bandwidth allocations

        Returns:
            Tuple of (log_probs, entropy, values)
        """
        action_dict = self.actor(observations, messages)

        offload_logits = action_dict['offload_logits']
        resource_mean = action_dict['resource_mean']
        bandwidth_mean = action_dict['bandwidth_mean']
        resource_log_std = action_dict['resource_log_std']
        bandwidth_log_std = action_dict['bandwidth_log_std']

        # Clamp logits for numerical stability
        offload_logits = torch.clamp(offload_logits, -20, 20)

        # Discrete distributions
        offload_probs = F.softmax(offload_logits, dim=-1)
        # Add small epsilon to avoid zero probabilities
        offload_probs = offload_probs + 1e-8
        offload_probs = offload_probs / offload_probs.sum(dim=-1, keepdim=True)

        # Log probs for offload actions
        offload_log_prob = 0
        offload_entropy = 0
        for i in range(self.num_connected_devices):
            dist = Categorical(offload_probs[:, i, :])
            offload_log_prob = offload_log_prob + dist.log_prob(offload_actions[:, i])
            offload_entropy = offload_entropy + dist.entropy()

        # Continuous distributions with clamped log_std
        resource_log_std = torch.clamp(resource_log_std, -5, 2)
        bandwidth_log_std = torch.clamp(bandwidth_log_std, -5, 2)
        resource_std = torch.exp(resource_log_std)
        bandwidth_std = torch.exp(bandwidth_log_std)

        resource_dist = Normal(resource_mean, resource_std + 1e-8)
        bandwidth_dist = Normal(bandwidth_mean, bandwidth_std + 1e-8)

        resource_log_prob = resource_dist.log_prob(resource_actions).sum(dim=-1)
        bandwidth_log_prob = bandwidth_dist.log_prob(bandwidth_actions).sum(dim=-1)

        total_log_prob = offload_log_prob + resource_log_prob + bandwidth_log_prob
        total_entropy = (
            offload_entropy +
            resource_dist.entropy().sum(dim=-1) +
            bandwidth_dist.entropy().sum(dim=-1)
        )

        # Values
        values = self.critic(observations, messages)

        return total_log_prob, total_entropy, values

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
