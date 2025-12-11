"""
Inter-Agent Communication Module for MARL-IoTP

Implements learned communication protocols between perception and
orchestration agents for coordinated decision-making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict


class MessageEncoder(nn.Module):
    """
    Encodes agent state into a fixed-size message for communication.

    Used by perception agents to communicate task requirements and
    device state to orchestration agents.
    """

    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        hidden_dim: int = 64
    ):
        """
        Initialize message encoder.

        Args:
            obs_dim: Observation dimension
            message_dim: Output message dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()  # Bound messages to [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation into message.

        Args:
            obs: Observation tensor of shape (batch, obs_dim)

        Returns:
            Message tensor of shape (batch, message_dim)
        """
        return self.encoder(obs)


class MessageDecoder(nn.Module):
    """
    Decodes received messages for use in action selection.

    Used by orchestration agents to interpret messages from
    perception agents.
    """

    def __init__(
        self,
        message_dim: int,
        output_dim: int,
        hidden_dim: int = 64
    ):
        """
        Initialize message decoder.

        Args:
            message_dim: Input message dimension
            output_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Decode message into features.

        Args:
            message: Message tensor of shape (batch, message_dim)

        Returns:
            Feature tensor of shape (batch, output_dim)
        """
        return self.decoder(message)


class MessageAggregator(nn.Module):
    """
    Aggregates messages from multiple agents using attention mechanism.

    Used by orchestration agents to process messages from multiple
    connected perception agents.
    """

    def __init__(
        self,
        message_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize message aggregator.

        Args:
            message_dim: Message dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=message_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Query projection for the receiving agent
        self.query_proj = nn.Linear(message_dim, message_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(message_dim)

    def forward(
        self,
        query: torch.Tensor,
        messages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate messages using attention.

        Args:
            query: Query tensor from receiving agent (batch, message_dim)
            messages: Messages from sending agents (batch, num_senders, message_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (aggregated message, attention weights)
        """
        # Project query
        query = self.query_proj(query).unsqueeze(1)  # (batch, 1, message_dim)

        # Apply attention
        attended, weights = self.attention(
            query=query,
            key=messages,
            value=messages,
            key_padding_mask=mask
        )

        # Residual connection and layer norm
        attended = attended.squeeze(1)  # (batch, message_dim)
        output = self.layer_norm(query.squeeze(1) + attended)

        return output, weights


class CommunicationModule(nn.Module):
    """
    Complete communication module for MARL-IoTP.

    Handles:
    - Message encoding from perception agents
    - Message aggregation for orchestration agents
    - Differentiable communication for end-to-end training
    """

    def __init__(
        self,
        perception_obs_dim: int,
        orchestration_obs_dim: int,
        message_dim: int,
        num_devices_per_server: int,
        hidden_dim: int = 64,
        num_attention_heads: int = 2
    ):
        """
        Initialize communication module.

        Args:
            perception_obs_dim: Perception agent observation dimension
            orchestration_obs_dim: Orchestration agent observation dimension
            message_dim: Message dimension
            num_devices_per_server: Max devices per edge server
            hidden_dim: Hidden layer dimension
            num_attention_heads: Number of attention heads
        """
        super().__init__()

        self.message_dim = message_dim
        self.num_devices_per_server = num_devices_per_server

        # Message encoder for perception agents
        self.encoder = MessageEncoder(
            obs_dim=perception_obs_dim,
            message_dim=message_dim,
            hidden_dim=hidden_dim
        )

        # Message aggregator for orchestration agents
        self.aggregator = MessageAggregator(
            message_dim=message_dim,
            num_heads=num_attention_heads
        )

        # Message decoder
        self.decoder = MessageDecoder(
            message_dim=message_dim,
            output_dim=hidden_dim
        )

        # Query encoder for orchestration agents
        self.query_encoder = nn.Sequential(
            nn.Linear(orchestration_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

    def encode_message(self, perception_obs: torch.Tensor) -> torch.Tensor:
        """
        Encode perception agent observation into message.

        Args:
            perception_obs: Perception agent observation (batch, obs_dim)

        Returns:
            Message tensor (batch, message_dim)
        """
        return self.encoder(perception_obs)

    def aggregate_messages(
        self,
        orchestration_obs: torch.Tensor,
        messages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate messages from perception agents.

        Args:
            orchestration_obs: Orchestration agent observation (batch, obs_dim)
            messages: Messages from perception agents (batch, num_senders, message_dim)
            mask: Optional mask for invalid messages

        Returns:
            Tuple of (aggregated features, attention weights)
        """
        # Create query from orchestration observation
        query = self.query_encoder(orchestration_obs)

        # Aggregate with attention
        aggregated, weights = self.aggregator(query, messages, mask)

        # Decode into features
        features = self.decoder(aggregated)

        return features, weights

    def forward(
        self,
        perception_obs_list: List[torch.Tensor],
        orchestration_obs: torch.Tensor,
        device_to_server: Dict[int, int],
        server_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode messages and aggregate for a server.

        Args:
            perception_obs_list: List of perception observations
            orchestration_obs: Orchestration agent observation
            device_to_server: Mapping of device IDs to server IDs
            server_id: Current server ID

        Returns:
            Tuple of (aggregated features, attention weights)
        """
        # Find devices connected to this server
        connected_devices = [
            d_id for d_id, s_id in device_to_server.items()
            if s_id == server_id
        ]

        # Encode messages from connected devices
        messages = []
        for device_id in connected_devices:
            if device_id < len(perception_obs_list):
                msg = self.encode_message(perception_obs_list[device_id])
                messages.append(msg)

        if not messages:
            # No connected devices, return zero features
            batch_size = orchestration_obs.size(0)
            return (
                torch.zeros(batch_size, self.decoder.decoder[-2].out_features),
                None
            )

        # Stack messages
        messages = torch.stack(messages, dim=1)  # (batch, num_devices, message_dim)

        # Create mask for padding if needed
        mask = None
        if len(connected_devices) < self.num_devices_per_server:
            # Pad messages
            padding = torch.zeros(
                messages.size(0),
                self.num_devices_per_server - len(connected_devices),
                self.message_dim
            ).to(messages.device)
            messages = torch.cat([messages, padding], dim=1)

            # Create mask (True for padded positions)
            mask = torch.zeros(messages.size(0), self.num_devices_per_server).bool()
            mask[:, len(connected_devices):] = True
            mask = mask.to(messages.device)

        return self.aggregate_messages(orchestration_obs, messages, mask)


class CommChannel:
    """
    Simple communication channel for passing messages between agents.

    Used during rollouts to store and retrieve messages.
    """

    def __init__(self, message_dim: int):
        """
        Initialize communication channel.

        Args:
            message_dim: Dimension of messages
        """
        self.message_dim = message_dim
        self.messages: Dict[int, np.ndarray] = {}

    def send(self, sender_id: int, message: np.ndarray):
        """
        Send a message.

        Args:
            sender_id: ID of sending agent
            message: Message array
        """
        self.messages[sender_id] = message.copy()

    def receive(self, sender_id: int) -> Optional[np.ndarray]:
        """
        Receive a message.

        Args:
            sender_id: ID of sending agent

        Returns:
            Message array or None if not available
        """
        return self.messages.get(sender_id)

    def receive_all(self, sender_ids: List[int]) -> List[np.ndarray]:
        """
        Receive messages from multiple senders.

        Args:
            sender_ids: List of sender IDs

        Returns:
            List of messages (zero array if message not available)
        """
        messages = []
        for sender_id in sender_ids:
            msg = self.messages.get(sender_id)
            if msg is not None:
                messages.append(msg)
            else:
                messages.append(np.zeros(self.message_dim))
        return messages

    def clear(self):
        """Clear all messages."""
        self.messages.clear()

    def get_all_messages(self) -> Dict[int, np.ndarray]:
        """Get all stored messages."""
        return self.messages.copy()
