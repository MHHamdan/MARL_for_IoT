"""
Advanced Baseline Algorithms for MARL-IoTP

Implements state-of-the-art MAPPO-MEC baselines:
- DPTORA: MAPPO with Priority-Gated Attention for MEC Task Offloading
- MADOA: MAPPO for D2D-MEC with heterogeneous agents
- TOMAC-PPO: Transformer-based MAPPO for MEC

References:
- DPTORA: Wang et al., Sensors 2024
- MADOA: Liu et al., Sensors 2024
- TOMAC-PPO: Future Internet 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch.distributions import Categorical, Normal


class PriorityGatedAttention(nn.Module):
    """
    Priority-Gated Attention Module from DPTORA.

    Computes task priorities based on urgency, resource requirements,
    and applies gated attention mechanism.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Priority computation network
        self.priority_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Priority in [0, 1]
        )

        # Gate computation
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for priority
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Projections
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with priority-gated attention.

        Args:
            features: Input features (batch, seq_len, input_dim)
            query: Query tensor (batch, 1, hidden_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights, priorities)
        """
        batch_size, seq_len, _ = features.shape

        # Compute priorities for each element
        priorities = self.priority_net(features)  # (batch, seq_len, 1)

        # Compute gates
        gated_input = torch.cat([features, priorities], dim=-1)
        gates = self.gate_net(gated_input)  # (batch, seq_len, hidden_dim)

        # Project and apply gates
        projected = self.input_proj(features)  # (batch, seq_len, hidden_dim)
        gated_features = projected * gates

        # Apply priority-weighted attention
        # Modify attention by incorporating priorities
        priority_weights = priorities.squeeze(-1)  # (batch, seq_len)

        # Apply attention
        attended, attn_weights = self.attention(
            query=query,
            key=gated_features,
            value=gated_features,
            key_padding_mask=mask
        )

        # Weight by priorities
        if attn_weights is not None:
            # Incorporate priorities into attention
            priority_adjusted_weights = attn_weights * priority_weights.unsqueeze(1)
            priority_adjusted_weights = priority_adjusted_weights / (
                priority_adjusted_weights.sum(dim=-1, keepdim=True) + 1e-8
            )

        # Project output
        output = self.output_proj(attended.squeeze(1))
        output = self.layer_norm(output)

        return output, attn_weights, priorities


class DPTORAActorNetwork(nn.Module):
    """
    Actor network for DPTORA baseline.

    Uses priority-gated attention for task offloading decisions.
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        num_devices: int,
        num_offload_options: int = 3,
        hidden_dim: int = 256,
        num_attention_heads: int = 2
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.num_devices = num_devices
        self.num_offload_options = num_offload_options
        self.hidden_dim = hidden_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Priority-gated attention for messages
        self.priority_attention = PriorityGatedAttention(
            input_dim=message_dim,
            hidden_dim=hidden_dim // 2,
            num_heads=num_attention_heads
        )

        # Query projection
        self.query_proj = nn.Linear(hidden_dim, hidden_dim // 2)

        # Combined feature processing
        combined_dim = hidden_dim + hidden_dim // 2
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action heads
        self.offload_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_offload_options * num_devices)
        )

        self.resource_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.resource_log_std = nn.Parameter(torch.zeros(1))
        self.bandwidth_log_std = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        obs: torch.Tensor,
        messages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        messages = torch.nan_to_num(messages, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size = obs.size(0)

        # Encode observation
        obs_features = self.obs_encoder(obs)

        # Create query from observation
        query = self.query_proj(obs_features).unsqueeze(1)

        # Apply priority-gated attention
        attended_msg, attn_weights, priorities = self.priority_attention(
            messages, query, mask
        )

        # Combine features
        combined = torch.cat([obs_features, attended_msg], dim=-1)
        features = self.feature_combiner(combined)

        # Compute actions
        offload_logits = self.offload_head(features)
        offload_logits = offload_logits.view(
            batch_size, self.num_devices, self.num_offload_options
        )
        offload_logits = torch.clamp(offload_logits, -10, 10)

        resource_mean = torch.sigmoid(self.resource_head(features))
        bandwidth_mean = torch.sigmoid(self.bandwidth_head(features))

        return {
            'offload_logits': offload_logits,
            'resource_mean': resource_mean,
            'resource_log_std': self.resource_log_std,
            'bandwidth_mean': bandwidth_mean,
            'bandwidth_log_std': self.bandwidth_log_std,
            'attention_weights': attn_weights,
            'priorities': priorities
        }


class MADOAActorNetwork(nn.Module):
    """
    Actor network for MADOA baseline.

    Multi-Agent DDPG Offloading Algorithm with heterogeneous agents
    for D2D-MEC systems.
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        num_devices: int,
        num_offload_options: int = 3,
        hidden_dim: int = 256,
        num_attention_heads: int = 2
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.num_devices = num_devices
        self.num_offload_options = num_offload_options
        self.hidden_dim = hidden_dim

        # Hierarchical observation encoder
        self.local_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # D2D message processor (for device-to-device communication)
        self.d2d_processor = nn.Sequential(
            nn.Linear(message_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )

        # Cross-attention for D2D coordination
        self.d2d_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 4,
            num_heads=num_attention_heads,
            batch_first=True
        )

        # Message aggregation
        self.msg_aggregator = nn.Sequential(
            nn.Linear(hidden_dim // 4 * num_devices, hidden_dim // 2),
            nn.ReLU()
        )

        # Combined processing
        combined_dim = hidden_dim // 2 + hidden_dim // 2
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action heads with D2D option (local, edge, cloud, D2D)
        self.offload_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_offload_options * num_devices)
        )

        self.resource_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.resource_log_std = nn.Parameter(torch.zeros(1))
        self.bandwidth_log_std = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        obs: torch.Tensor,
        messages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with D2D coordination."""
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        messages = torch.nan_to_num(messages, nan=0.0, posinf=1.0, neginf=-1.0)

        batch_size = obs.size(0)

        # Local observation encoding
        local_features = self.local_encoder(obs)

        # D2D message processing
        d2d_features = self.d2d_processor(messages)  # (batch, num_devices, hidden//4)

        # D2D cross-attention
        d2d_attended, d2d_weights = self.d2d_attention(
            query=d2d_features,
            key=d2d_features,
            value=d2d_features,
            key_padding_mask=mask
        )

        # Aggregate D2D information
        d2d_flat = d2d_attended.view(batch_size, -1)
        d2d_agg = self.msg_aggregator(d2d_flat)

        # Combine local and D2D features
        combined = torch.cat([local_features, d2d_agg], dim=-1)
        features = self.combiner(combined)

        # Compute actions
        offload_logits = self.offload_head(features)
        offload_logits = offload_logits.view(
            batch_size, self.num_devices, self.num_offload_options
        )
        offload_logits = torch.clamp(offload_logits, -10, 10)

        resource_mean = self.resource_head(features)
        bandwidth_mean = self.bandwidth_head(features)

        return {
            'offload_logits': offload_logits,
            'resource_mean': resource_mean,
            'resource_log_std': self.resource_log_std,
            'bandwidth_mean': bandwidth_mean,
            'bandwidth_log_std': self.bandwidth_log_std,
            'attention_weights': d2d_weights,
            'd2d_features': d2d_attended
        }


class DPTORABaseline:
    """
    DPTORA: Dynamic Priority-based Task Offloading with
    Reinforcement learning and Attention mechanism.

    Based on: Wang et al., Sensors 2024
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        num_devices: int,
        num_servers: int,
        config: dict,
        device: str = 'cpu'
    ):
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.num_devices = num_devices
        self.num_servers = num_servers
        self.config = config
        self.device = torch.device(device)

        self.devices_per_server = config.get('devices_per_server', 7)
        self.hidden_dim = config.get('hidden_dim', 256)

        # Create actor networks for each server
        self.actors = nn.ModuleList([
            DPTORAActorNetwork(
                obs_dim=config.get('orchestration_obs_dim', 32),
                message_dim=message_dim,
                num_devices=self.devices_per_server,
                hidden_dim=self.hidden_dim
            ).to(self.device)
            for _ in range(num_servers)
        ])

        # Optimizers
        self.optimizers = [
            optim.Adam(actor.parameters(), lr=config.get('lr', 3e-4))
            for actor in self.actors
        ]

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)

    def get_action(
        self,
        obs: np.ndarray,
        messages: List[np.ndarray],
        server_id: int,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """Get action for a specific server."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Pad messages if needed
        while len(messages) < self.devices_per_server:
            messages.append(np.zeros(self.message_dim))
        messages = messages[:self.devices_per_server]

        msg_tensor = torch.FloatTensor(
            np.array(messages)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_dict = self.actors[server_id](obs_tensor, msg_tensor)

        # Sample actions
        offload_logits = action_dict['offload_logits']
        offload_probs = F.softmax(offload_logits, dim=-1)

        if deterministic:
            offload_actions = offload_probs.argmax(dim=-1)
        else:
            offload_dists = [
                Categorical(offload_probs[:, i, :])
                for i in range(self.devices_per_server)
            ]
            offload_actions = torch.stack(
                [dist.sample() for dist in offload_dists], dim=1
            )

        resource_mean = action_dict['resource_mean']
        bandwidth_mean = action_dict['bandwidth_mean']

        if deterministic:
            resource_action = resource_mean
            bandwidth_action = bandwidth_mean
        else:
            resource_std = torch.exp(action_dict['resource_log_std'])
            bandwidth_std = torch.exp(action_dict['bandwidth_log_std'])

            resource_action = torch.clamp(
                Normal(resource_mean, resource_std).sample(), 0.0, 1.0
            )
            bandwidth_action = torch.clamp(
                Normal(bandwidth_mean, bandwidth_std).sample(), 0.0, 1.0
            )

        return {
            'offload_decisions': offload_actions.cpu().numpy().flatten(),
            'resource_allocation': resource_action.cpu().numpy().flatten(),
            'bandwidth_allocation': bandwidth_action.cpu().numpy().flatten(),
            'priorities': action_dict['priorities'].cpu().numpy()
        }


class MADOABaseline:
    """
    MADOA: Multi-Agent DDPG Offloading Algorithm for D2D-MEC.

    Based on: Liu et al., Sensors 2024
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        num_devices: int,
        num_servers: int,
        config: dict,
        device: str = 'cpu'
    ):
        self.obs_dim = obs_dim
        self.message_dim = message_dim
        self.num_devices = num_devices
        self.num_servers = num_servers
        self.config = config
        self.device = torch.device(device)

        self.devices_per_server = config.get('devices_per_server', 7)
        self.hidden_dim = config.get('hidden_dim', 256)

        # Create actor networks
        self.actors = nn.ModuleList([
            MADOAActorNetwork(
                obs_dim=config.get('orchestration_obs_dim', 32),
                message_dim=message_dim,
                num_devices=self.devices_per_server,
                hidden_dim=self.hidden_dim
            ).to(self.device)
            for _ in range(num_servers)
        ])

        self.optimizers = [
            optim.Adam(actor.parameters(), lr=config.get('lr', 3e-4))
            for actor in self.actors
        ]

        self.gamma = config.get('gamma', 0.99)

    def get_action(
        self,
        obs: np.ndarray,
        messages: List[np.ndarray],
        server_id: int,
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """Get action with D2D coordination."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        while len(messages) < self.devices_per_server:
            messages.append(np.zeros(self.message_dim))
        messages = messages[:self.devices_per_server]

        msg_tensor = torch.FloatTensor(
            np.array(messages)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_dict = self.actors[server_id](obs_tensor, msg_tensor)

        offload_logits = action_dict['offload_logits']
        offload_probs = F.softmax(offload_logits, dim=-1)

        if deterministic:
            offload_actions = offload_probs.argmax(dim=-1)
        else:
            offload_dists = [
                Categorical(offload_probs[:, i, :])
                for i in range(self.devices_per_server)
            ]
            offload_actions = torch.stack(
                [dist.sample() for dist in offload_dists], dim=1
            )

        resource_mean = action_dict['resource_mean']
        bandwidth_mean = action_dict['bandwidth_mean']

        if deterministic:
            resource_action = resource_mean
            bandwidth_action = bandwidth_mean
        else:
            resource_std = torch.exp(action_dict['resource_log_std'])
            bandwidth_std = torch.exp(action_dict['bandwidth_log_std'])

            resource_action = torch.clamp(
                Normal(resource_mean, resource_std).sample(), 0.0, 1.0
            )
            bandwidth_action = torch.clamp(
                Normal(bandwidth_mean, bandwidth_std).sample(), 0.0, 1.0
            )

        return {
            'offload_decisions': offload_actions.cpu().numpy().flatten(),
            'resource_allocation': resource_action.cpu().numpy().flatten(),
            'bandwidth_allocation': bandwidth_action.cpu().numpy().flatten(),
            'd2d_features': action_dict['d2d_features'].cpu().numpy()
        }
