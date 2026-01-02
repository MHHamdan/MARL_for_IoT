"""
Advanced Analysis Tools for MARL-IoTP

Provides:
- Mutual information analysis for learned communication
- Attention weight visualization
- Fairness metrics (Jain's index)
- Tail latency computation (p95, p99)
- Message dimension ablation analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
from pathlib import Path


def compute_jains_fairness_index(values: np.ndarray) -> float:
    """
    Compute Jain's Fairness Index.

    J(x) = (sum(x_i))^2 / (n * sum(x_i^2))

    Ranges from 1/n (completely unfair) to 1 (perfectly fair).

    Args:
        values: Array of values per entity (e.g., rewards per device)

    Returns:
        Jain's fairness index in [0, 1]
    """
    if len(values) == 0:
        return 1.0

    values = np.array(values)
    n = len(values)

    sum_values = np.sum(values)
    sum_squared = np.sum(values ** 2)

    if sum_squared == 0:
        return 1.0

    return (sum_values ** 2) / (n * sum_squared)


def compute_tail_latencies(
    latencies: np.ndarray,
    percentiles: List[float] = [50, 90, 95, 99]
) -> Dict[str, float]:
    """
    Compute tail latency metrics.

    Args:
        latencies: Array of latency values
        percentiles: List of percentiles to compute

    Returns:
        Dictionary with percentile latencies
    """
    if len(latencies) == 0:
        return {f'p{p}': 0.0 for p in percentiles}

    latencies = np.array(latencies)
    result = {}

    for p in percentiles:
        result[f'p{p}'] = np.percentile(latencies, p)

    result['mean'] = np.mean(latencies)
    result['std'] = np.std(latencies)
    result['min'] = np.min(latencies)
    result['max'] = np.max(latencies)

    return result


def compute_energy_delay_product(
    latencies: np.ndarray,
    energies: np.ndarray
) -> float:
    """
    Compute Energy-Delay Product (EDP).

    EDP = E * D (lower is better)

    Args:
        latencies: Array of latency values
        energies: Array of energy values

    Returns:
        Mean EDP
    """
    if len(latencies) == 0 or len(energies) == 0:
        return 0.0

    edp = np.array(latencies) * np.array(energies)
    return np.mean(edp)


class MutualInformationEstimator:
    """
    Estimates mutual information between device states and messages.

    Uses binning approach for continuous variables.
    """

    def __init__(self, num_bins: int = 20):
        """
        Initialize MI estimator.

        Args:
            num_bins: Number of bins for discretization
        """
        self.num_bins = num_bins
        self.states = []
        self.messages = []

    def add_sample(self, state: np.ndarray, message: np.ndarray):
        """Add a state-message pair."""
        self.states.append(state.flatten())
        self.messages.append(message.flatten())

    def compute_mi(self) -> Dict[str, float]:
        """
        Compute mutual information I(message; state).

        Returns:
            Dictionary with MI estimates per dimension and total
        """
        if len(self.states) < 100:
            return {'total_mi': 0.0, 'normalized_mi': 0.0}

        states = np.array(self.states)
        messages = np.array(self.messages)

        n_samples = len(states)
        state_dim = states.shape[1]
        message_dim = messages.shape[1]

        total_mi = 0.0
        per_dim_mi = {}

        for m_idx in range(message_dim):
            msg_bins = self._discretize(messages[:, m_idx])
            dim_mi = 0.0

            for s_idx in range(state_dim):
                state_bins = self._discretize(states[:, s_idx])
                mi = self._compute_mi_discrete(state_bins, msg_bins)
                dim_mi += mi

            per_dim_mi[f'msg_dim_{m_idx}'] = dim_mi / state_dim
            total_mi += dim_mi

        # Normalize by state entropy (upper bound on MI)
        state_entropy = self._compute_entropy(states)

        return {
            'total_mi': total_mi / (state_dim * message_dim),
            'normalized_mi': total_mi / (state_entropy + 1e-8),
            'state_entropy': state_entropy,
            'per_dim_mi': per_dim_mi
        }

    def _discretize(self, values: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins."""
        min_val, max_val = values.min(), values.max()
        if max_val == min_val:
            return np.zeros_like(values, dtype=int)

        bins = np.linspace(min_val, max_val, self.num_bins + 1)
        return np.digitize(values, bins[:-1]) - 1

    def _compute_mi_discrete(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Compute MI between discrete variables."""
        # Joint distribution
        joint_counts = defaultdict(int)
        for xi, yi in zip(x, y):
            joint_counts[(xi, yi)] += 1

        n = len(x)
        x_counts = defaultdict(int)
        y_counts = defaultdict(int)

        for xi, yi in zip(x, y):
            x_counts[xi] += 1
            y_counts[yi] += 1

        mi = 0.0
        for (xi, yi), count in joint_counts.items():
            p_xy = count / n
            p_x = x_counts[xi] / n
            p_y = y_counts[yi] / n

            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log2(p_xy / (p_x * p_y))

        return mi

    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy of data."""
        total_entropy = 0.0

        for dim in range(data.shape[1]):
            bins = self._discretize(data[:, dim])
            counts = defaultdict(int)
            for b in bins:
                counts[b] += 1

            n = len(bins)
            entropy = 0.0
            for count in counts.values():
                p = count / n
                if p > 0:
                    entropy -= p * np.log2(p)

            total_entropy += entropy

        return total_entropy

    def reset(self):
        """Reset collected samples."""
        self.states = []
        self.messages = []


class AttentionAnalyzer:
    """
    Analyzes attention weights from orchestration agents.
    """

    def __init__(self):
        self.attention_weights = []
        self.device_states = []
        self.context = []

    def add_sample(
        self,
        weights: np.ndarray,
        device_states: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Add attention weight sample.

        Args:
            weights: Attention weights (num_devices,) or (num_heads, num_devices)
            device_states: States of devices (num_devices, state_dim)
            context: Optional context (urgency, load, etc.)
        """
        self.attention_weights.append(weights)
        self.device_states.append(device_states)
        self.context.append(context or {})

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze attention patterns.

        Returns:
            Analysis results
        """
        if not self.attention_weights:
            return {}

        weights = np.array(self.attention_weights)

        # Handle multi-head attention
        if weights.ndim == 3:
            # Average across heads
            weights = weights.mean(axis=1)

        results = {
            'mean_weights': weights.mean(axis=0).tolist(),
            'std_weights': weights.std(axis=0).tolist(),
            'entropy': self._compute_attention_entropy(weights),
            'concentration': self._compute_concentration(weights),
        }

        # Analyze correlation with device states
        if self.context and any(self.context):
            results['correlations'] = self._analyze_correlations(weights)

        return results

    def _compute_attention_entropy(self, weights: np.ndarray) -> float:
        """Compute average entropy of attention distributions."""
        entropies = []
        for w in weights:
            w = w + 1e-8
            w = w / w.sum()
            entropy = -np.sum(w * np.log2(w))
            entropies.append(entropy)
        return np.mean(entropies)

    def _compute_concentration(self, weights: np.ndarray) -> float:
        """
        Compute attention concentration (Gini coefficient).

        Lower values indicate more uniform attention.
        """
        concentrations = []
        for w in weights:
            w_sorted = np.sort(w)
            n = len(w)
            indices = np.arange(1, n + 1)
            gini = (2 * np.sum(indices * w_sorted)) / (n * np.sum(w_sorted)) - (n + 1) / n
            concentrations.append(gini)
        return np.mean(concentrations)

    def _analyze_correlations(self, weights: np.ndarray) -> Dict[str, float]:
        """Analyze correlation between attention and device properties."""
        correlations = {}

        # Extract context features
        loads = []
        urgencies = []
        batteries = []

        for ctx in self.context:
            if 'load' in ctx:
                loads.append(ctx['load'])
            if 'urgency' in ctx:
                urgencies.append(ctx['urgency'])
            if 'battery' in ctx:
                batteries.append(ctx['battery'])

        # Compute correlations
        if loads:
            loads = np.array(loads)
            for i in range(weights.shape[1]):
                if i < loads.shape[1] if loads.ndim > 1 else True:
                    load_vals = loads[:, i] if loads.ndim > 1 else loads
                    corr = np.corrcoef(weights[:, i], load_vals)[0, 1]
                    if not np.isnan(corr):
                        correlations[f'load_device_{i}'] = corr

        return correlations

    def reset(self):
        """Reset collected samples."""
        self.attention_weights = []
        self.device_states = []
        self.context = []


class MessageDimensionAnalyzer:
    """
    Analyzes impact of message dimension on performance.
    """

    def __init__(self):
        self.results = {}

    def add_result(
        self,
        message_dim: int,
        metrics: Dict[str, float]
    ):
        """Add result for a message dimension."""
        self.results[message_dim] = metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all message dimension experiments."""
        if not self.results:
            return {}

        dims = sorted(self.results.keys())

        summary = {
            'dimensions': dims,
            'metrics': {}
        }

        # Collect all metrics
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())

        for metric in all_metrics:
            summary['metrics'][metric] = [
                self.results[d].get(metric, 0.0) for d in dims
            ]

        # Find optimal dimension
        if 'reward' in summary['metrics']:
            rewards = summary['metrics']['reward']
            best_idx = np.argmax(rewards)
            summary['optimal_dimension'] = dims[best_idx]
            summary['optimal_reward'] = rewards[best_idx]

        return summary

    def save_results(self, path: str):
        """Save results to file."""
        with open(path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)


class EnhancedMetricsCollector:
    """
    Collects enhanced metrics during evaluation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all collected metrics."""
        self.latencies = []
        self.energies = []
        self.accuracies = []
        self.rewards = []
        self.violations = []
        self.per_device_rewards = defaultdict(list)
        self.per_device_latencies = defaultdict(list)

    def add_step(
        self,
        latency: float,
        energy: float,
        accuracy: float,
        reward: float,
        violation: bool,
        device_id: Optional[int] = None
    ):
        """Add metrics for a single step."""
        self.latencies.append(latency)
        self.energies.append(energy)
        self.accuracies.append(accuracy)
        self.rewards.append(reward)
        self.violations.append(violation)

        if device_id is not None:
            self.per_device_rewards[device_id].append(reward)
            self.per_device_latencies[device_id].append(latency)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        latencies = np.array(self.latencies)
        energies = np.array(self.energies)
        accuracies = np.array(self.accuracies)
        rewards = np.array(self.rewards)
        violations = np.array(self.violations)

        # Compute per-device fairness
        device_rewards = np.array([
            np.mean(r) for r in self.per_device_rewards.values()
        ])
        device_latencies = np.array([
            np.mean(l) for l in self.per_device_latencies.values()
        ])

        return {
            # Basic metrics
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_latency': float(np.mean(latencies)),
            'mean_energy': float(np.mean(energies)),
            'mean_accuracy': float(np.mean(accuracies)),
            'mean_violations': float(np.mean(violations)),

            # Tail latencies
            'p50_latency': float(np.percentile(latencies, 50)),
            'p90_latency': float(np.percentile(latencies, 90)),
            'p95_latency': float(np.percentile(latencies, 95)),
            'p99_latency': float(np.percentile(latencies, 99)),

            # Fairness metrics
            'reward_fairness': float(
                compute_jains_fairness_index(device_rewards)
                if len(device_rewards) > 0 else 1.0
            ),
            'latency_fairness': float(
                compute_jains_fairness_index(1.0 / (device_latencies + 1e-8))
                if len(device_latencies) > 0 else 1.0
            ),

            # Energy-delay product
            'mean_edp': float(compute_energy_delay_product(latencies, energies)),

            # Deadline success rate
            'deadline_success_rate': float(1.0 - np.mean(violations)),

            # Sample count
            'num_samples': len(self.latencies)
        }


def analyze_message_semantics(
    messages: np.ndarray,
    states: np.ndarray,
    labels: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Analyze semantic meaning of learned messages.

    Args:
        messages: Array of messages (n_samples, message_dim)
        states: Array of corresponding states (n_samples, state_dim)
        labels: Optional state feature labels

    Returns:
        Semantic analysis results
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n_clusters = min(5, len(messages) // 10)
    if n_clusters < 2:
        return {'error': 'Not enough samples for clustering'}

    # Cluster messages
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(messages)

    # Analyze cluster characteristics
    cluster_analysis = {}
    for c in range(n_clusters):
        mask = clusters == c
        cluster_states = states[mask]

        cluster_analysis[f'cluster_{c}'] = {
            'size': int(mask.sum()),
            'state_means': cluster_states.mean(axis=0).tolist(),
            'state_stds': cluster_states.std(axis=0).tolist(),
            'centroid': kmeans.cluster_centers_[c].tolist()
        }

    # PCA for visualization
    pca = PCA(n_components=2)
    messages_2d = pca.fit_transform(messages)

    return {
        'num_clusters': n_clusters,
        'cluster_analysis': cluster_analysis,
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'messages_2d': messages_2d.tolist(),
        'clusters': clusters.tolist()
    }
