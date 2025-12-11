"""
Replay Buffer Implementations for MARL-IoTP

Contains buffer implementations for both on-policy (MAPPO) and
off-policy (MADDPG) algorithms.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO, MAPPO).

    Stores complete trajectories for computing advantages and
    performing policy updates.
    """

    def __init__(
        self,
        buffer_size: int,
        num_perception_agents: int,
        num_orchestration_agents: int,
        perception_obs_dim: int,
        orchestration_obs_dim: int,
        message_dim: int,
        devices_per_server: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu'
    ):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Maximum buffer size (steps)
            num_perception_agents: Number of perception agents
            num_orchestration_agents: Number of orchestration agents
            perception_obs_dim: Perception observation dimension
            orchestration_obs_dim: Orchestration observation dimension
            message_dim: Message dimension
            devices_per_server: Devices per server
            gamma: Discount factor
            gae_lambda: GAE lambda
            device: Compute device
        """
        self.buffer_size = buffer_size
        self.num_perception_agents = num_perception_agents
        self.num_orchestration_agents = num_orchestration_agents
        self.perception_obs_dim = perception_obs_dim
        self.orchestration_obs_dim = orchestration_obs_dim
        self.message_dim = message_dim
        self.devices_per_server = devices_per_server
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        self.reset()

    def reset(self):
        """Reset buffer."""
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

        # Perception agent data
        self.perception_obs = np.zeros(
            (self.buffer_size, self.num_perception_agents, self.perception_obs_dim),
            dtype=np.float32
        )
        self.perception_model_actions = np.zeros(
            (self.buffer_size, self.num_perception_agents),
            dtype=np.int64
        )
        self.perception_frame_rate_actions = np.zeros(
            (self.buffer_size, self.num_perception_agents, 1),
            dtype=np.float32
        )
        self.perception_log_probs = np.zeros(
            (self.buffer_size, self.num_perception_agents),
            dtype=np.float32
        )
        self.perception_rewards = np.zeros(
            (self.buffer_size, self.num_perception_agents),
            dtype=np.float32
        )
        self.perception_values = np.zeros(
            (self.buffer_size, self.num_perception_agents),
            dtype=np.float32
        )

        # Orchestration agent data
        self.orchestration_obs = np.zeros(
            (self.buffer_size, self.num_orchestration_agents, self.orchestration_obs_dim),
            dtype=np.float32
        )
        self.orchestration_messages = np.zeros(
            (self.buffer_size, self.num_orchestration_agents,
             self.devices_per_server, self.message_dim),
            dtype=np.float32
        )
        self.orchestration_offload_actions = np.zeros(
            (self.buffer_size, self.num_orchestration_agents, self.devices_per_server),
            dtype=np.int64
        )
        self.orchestration_resource_actions = np.zeros(
            (self.buffer_size, self.num_orchestration_agents, 1),
            dtype=np.float32
        )
        self.orchestration_bandwidth_actions = np.zeros(
            (self.buffer_size, self.num_orchestration_agents, 1),
            dtype=np.float32
        )
        self.orchestration_log_probs = np.zeros(
            (self.buffer_size, self.num_orchestration_agents),
            dtype=np.float32
        )
        self.orchestration_rewards = np.zeros(
            (self.buffer_size, self.num_orchestration_agents),
            dtype=np.float32
        )
        self.orchestration_values = np.zeros(
            (self.buffer_size, self.num_orchestration_agents),
            dtype=np.float32
        )

        # Shared data
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)

        # Computed advantages and returns
        self.perception_advantages = np.zeros(
            (self.buffer_size, self.num_perception_agents),
            dtype=np.float32
        )
        self.perception_returns = np.zeros(
            (self.buffer_size, self.num_perception_agents),
            dtype=np.float32
        )
        self.orchestration_advantages = np.zeros(
            (self.buffer_size, self.num_orchestration_agents),
            dtype=np.float32
        )
        self.orchestration_returns = np.zeros(
            (self.buffer_size, self.num_orchestration_agents),
            dtype=np.float32
        )

    def store(
        self,
        perception_obs: List[np.ndarray],
        perception_actions: List[Dict],
        perception_log_probs: List[float],
        perception_values: List[float],
        perception_rewards: List[float],
        orchestration_obs: List[np.ndarray],
        orchestration_messages: List[List[np.ndarray]],
        orchestration_actions: List[Dict],
        orchestration_log_probs: List[float],
        orchestration_values: List[float],
        orchestration_rewards: List[float],
        done: bool
    ):
        """
        Store a transition.

        Args:
            perception_obs: Perception agent observations
            perception_actions: Perception agent actions
            perception_log_probs: Log probabilities
            perception_values: Value estimates
            perception_rewards: Rewards
            orchestration_obs: Orchestration agent observations
            orchestration_messages: Messages received
            orchestration_actions: Orchestration agent actions
            orchestration_log_probs: Log probabilities
            orchestration_values: Value estimates
            orchestration_rewards: Rewards
            done: Episode termination flag
        """
        idx = self.ptr

        # Store perception data
        for i in range(self.num_perception_agents):
            self.perception_obs[idx, i] = perception_obs[i]
            self.perception_model_actions[idx, i] = perception_actions[i]['model_selection']
            self.perception_frame_rate_actions[idx, i] = perception_actions[i]['frame_rate']
            self.perception_log_probs[idx, i] = perception_log_probs[i]
            self.perception_values[idx, i] = perception_values[i]
            self.perception_rewards[idx, i] = perception_rewards[i]

        # Store orchestration data
        for i in range(self.num_orchestration_agents):
            self.orchestration_obs[idx, i] = orchestration_obs[i]
            # Pad messages if needed
            msgs = orchestration_messages[i]
            for j in range(min(len(msgs), self.devices_per_server)):
                self.orchestration_messages[idx, i, j] = msgs[j]
            self.orchestration_offload_actions[idx, i] = orchestration_actions[i]['offload_decisions'][:self.devices_per_server]
            self.orchestration_resource_actions[idx, i] = orchestration_actions[i]['resource_allocation']
            self.orchestration_bandwidth_actions[idx, i] = orchestration_actions[i]['bandwidth_allocation']
            self.orchestration_log_probs[idx, i] = orchestration_log_probs[i]
            self.orchestration_values[idx, i] = orchestration_values[i]
            self.orchestration_rewards[idx, i] = orchestration_rewards[i]

        self.dones[idx] = done

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def compute_advantages(
        self,
        last_perception_values: List[float],
        last_orchestration_values: List[float]
    ):
        """
        Compute GAE advantages for all agents.

        Args:
            last_perception_values: Final value estimates for perception agents
            last_orchestration_values: Final value estimates for orchestration agents
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        length = self.ptr - self.path_start_idx

        # Compute for perception agents
        for i in range(self.num_perception_agents):
            rewards = self.perception_rewards[path_slice, i]
            values = self.perception_values[path_slice, i]
            dones = self.dones[path_slice]

            advantages, returns = self._compute_gae(
                rewards, values, dones, last_perception_values[i]
            )

            self.perception_advantages[path_slice, i] = advantages
            self.perception_returns[path_slice, i] = returns

        # Compute for orchestration agents
        for i in range(self.num_orchestration_agents):
            rewards = self.orchestration_rewards[path_slice, i]
            values = self.orchestration_values[path_slice, i]
            dones = self.dones[path_slice]

            advantages, returns = self._compute_gae(
                rewards, values, dones, last_orchestration_values[i]
            )

            self.orchestration_advantages[path_slice, i] = advantages
            self.orchestration_returns[path_slice, i] = returns

        self.path_start_idx = self.ptr

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Reward array
            values: Value array
            dones: Done flags
            last_value: Final value estimate

        Returns:
            Tuple of (advantages, returns)
        """
        length = len(rewards)
        advantages = np.zeros(length, dtype=np.float32)
        returns = np.zeros(length, dtype=np.float32)

        # Handle NaN/Inf values and clip to reasonable range
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
        rewards = np.clip(rewards, -100.0, 100.0)
        values = np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)
        values = np.clip(values, -100.0, 100.0)
        last_value = 0.0 if (np.isnan(last_value) or np.isinf(last_value)) else float(np.clip(last_value, -100.0, 100.0))

        last_gae = 0.0

        for t in reversed(range(length)):
            if t == length - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            # Clip delta to prevent overflow
            delta = float(np.clip(delta, -100.0, 100.0))
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            # Clip last_gae to prevent overflow accumulation
            last_gae = float(np.clip(last_gae, -100.0, 100.0))
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        # Final NaN/Inf check and clipping
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)
        advantages = np.clip(advantages, -100.0, 100.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)
        returns = np.clip(returns, -100.0, 100.0)

        return advantages, returns

    def get_batches(
        self,
        batch_size: int,
        num_minibatches: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Get minibatches for training.

        Args:
            batch_size: Total batch size
            num_minibatches: Number of minibatches

        Returns:
            List of minibatch dictionaries
        """
        size = self.ptr if not self.full else self.buffer_size
        indices = np.random.permutation(size)

        minibatch_size = size // num_minibatches
        batches = []

        for start in range(0, size, minibatch_size):
            end = min(start + minibatch_size, size)
            mb_indices = indices[start:end]

            batch = {
                # Perception data
                'perception_obs': torch.FloatTensor(
                    self.perception_obs[mb_indices]
                ).to(self.device),
                'perception_model_actions': torch.LongTensor(
                    self.perception_model_actions[mb_indices]
                ).to(self.device),
                'perception_frame_rate_actions': torch.FloatTensor(
                    self.perception_frame_rate_actions[mb_indices]
                ).to(self.device),
                'perception_old_log_probs': torch.FloatTensor(
                    self.perception_log_probs[mb_indices]
                ).to(self.device),
                'perception_advantages': torch.FloatTensor(
                    self.perception_advantages[mb_indices]
                ).to(self.device),
                'perception_returns': torch.FloatTensor(
                    self.perception_returns[mb_indices]
                ).to(self.device),

                # Orchestration data
                'orchestration_obs': torch.FloatTensor(
                    self.orchestration_obs[mb_indices]
                ).to(self.device),
                'orchestration_messages': torch.FloatTensor(
                    self.orchestration_messages[mb_indices]
                ).to(self.device),
                'orchestration_offload_actions': torch.LongTensor(
                    self.orchestration_offload_actions[mb_indices]
                ).to(self.device),
                'orchestration_resource_actions': torch.FloatTensor(
                    self.orchestration_resource_actions[mb_indices]
                ).to(self.device),
                'orchestration_bandwidth_actions': torch.FloatTensor(
                    self.orchestration_bandwidth_actions[mb_indices]
                ).to(self.device),
                'orchestration_old_log_probs': torch.FloatTensor(
                    self.orchestration_log_probs[mb_indices]
                ).to(self.device),
                'orchestration_advantages': torch.FloatTensor(
                    self.orchestration_advantages[mb_indices]
                ).to(self.device),
                'orchestration_returns': torch.FloatTensor(
                    self.orchestration_returns[mb_indices]
                ).to(self.device),
            }

            batches.append(batch)

        return batches

    def clear(self):
        """Clear buffer data."""
        self.reset()

    def __len__(self):
        return self.ptr if not self.full else self.buffer_size


class ReplayBuffer:
    """
    Experience replay buffer for off-policy algorithms (MADDPG, DQN).

    Stores individual transitions for random sampling.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dims: List[int],
        action_dims: List[int],
        device: str = 'cpu'
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer capacity
            num_agents: Number of agents
            obs_dims: Observation dimensions per agent
            action_dims: Action dimensions per agent
            device: Compute device
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.device = torch.device(device)

        self.buffer = deque(maxlen=capacity)

    def store(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        dones: List[bool]
    ):
        """
        Store a transition.

        Args:
            observations: Current observations for all agents
            actions: Actions taken by all agents
            rewards: Rewards received by all agents
            next_observations: Next observations for all agents
            dones: Done flags for all agents
        """
        transition = {
            'obs': [obs.copy() for obs in observations],
            'actions': [act.copy() if isinstance(act, np.ndarray) else np.array([act])
                       for act in actions],
            'rewards': np.array(rewards, dtype=np.float32),
            'next_obs': [obs.copy() for obs in next_observations],
            'dones': np.array(dones, dtype=np.float32)
        }
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict[str, List[torch.Tensor]]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of batched tensors per agent
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Organize by agent
        result = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': []
        }

        for agent_idx in range(self.num_agents):
            obs = np.array([t['obs'][agent_idx] for t in batch])
            actions = np.array([t['actions'][agent_idx] for t in batch])
            rewards = np.array([t['rewards'][agent_idx] for t in batch])
            next_obs = np.array([t['next_obs'][agent_idx] for t in batch])
            dones = np.array([t['dones'][agent_idx] for t in batch])

            result['obs'].append(
                torch.FloatTensor(obs).to(self.device)
            )
            result['actions'].append(
                torch.FloatTensor(actions).to(self.device)
            )
            result['rewards'].append(
                torch.FloatTensor(rewards).to(self.device)
            )
            result['next_obs'].append(
                torch.FloatTensor(next_obs).to(self.device)
            )
            result['dones'].append(
                torch.FloatTensor(dones).to(self.device)
            )

        return result

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size
