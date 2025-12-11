"""
Metrics Tracking and Computation for MARL-IoTP

Provides utilities for tracking and computing performance metrics
during training and evaluation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MetricStats:
    """Statistics for a single metric."""
    name: str
    values: List[float] = field(default_factory=list)
    window_size: int = 100

    def add(self, value: float):
        """Add a value."""
        self.values.append(value)

    def mean(self) -> float:
        """Get mean of all values."""
        return np.mean(self.values) if self.values else 0.0

    def std(self) -> float:
        """Get standard deviation."""
        return np.std(self.values) if len(self.values) > 1 else 0.0

    def recent_mean(self) -> float:
        """Get mean of recent values."""
        if not self.values:
            return 0.0
        window = self.values[-self.window_size:]
        return np.mean(window)

    def recent_std(self) -> float:
        """Get std of recent values."""
        if len(self.values) < 2:
            return 0.0
        window = self.values[-self.window_size:]
        return np.std(window)

    def max(self) -> float:
        """Get maximum value."""
        return np.max(self.values) if self.values else 0.0

    def min(self) -> float:
        """Get minimum value."""
        return np.min(self.values) if self.values else 0.0

    def last(self) -> float:
        """Get last value."""
        return self.values[-1] if self.values else 0.0


class MetricsTracker:
    """
    Tracks multiple metrics throughout training.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.metrics: Dict[str, MetricStats] = {}
        self.step = 0

    def add(self, name: str, value: float):
        """
        Add a metric value.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = MetricStats(
                name=name,
                window_size=self.window_size
            )
        self.metrics[name].add(value)

    def add_dict(self, metrics: Dict[str, float]):
        """Add multiple metrics from dictionary."""
        for name, value in metrics.items():
            self.add(name, value)

    def get(self, name: str) -> Optional[MetricStats]:
        """Get metric statistics."""
        return self.metrics.get(name)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        for name, stats in self.metrics.items():
            summary[name] = {
                'mean': stats.mean(),
                'std': stats.std(),
                'recent_mean': stats.recent_mean(),
                'recent_std': stats.recent_std(),
                'max': stats.max(),
                'min': stats.min(),
                'last': stats.last()
            }
        return summary

    def get_recent(self) -> Dict[str, float]:
        """Get recent mean of all metrics."""
        return {
            name: stats.recent_mean()
            for name, stats in self.metrics.items()
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.step = 0


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute aggregated metrics from a list of results.

    Args:
        results: List of result dictionaries

    Returns:
        Aggregated metrics
    """
    if not results:
        return {
            'avg_latency': 0,
            'avg_energy': 0,
            'avg_accuracy': 0,
            'deadline_violation_rate': 0,
            'throughput': 0
        }

    latencies = [r['latency_ms'] for r in results if 'latency_ms' in r]
    energies = [r['energy_j'] for r in results if 'energy_j' in r]
    accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
    violations = [r.get('deadline_violated', False) for r in results]

    return {
        'avg_latency': np.mean(latencies) if latencies else 0,
        'avg_energy': np.mean(energies) if energies else 0,
        'avg_accuracy': np.mean(accuracies) if accuracies else 0,
        'deadline_violation_rate': np.mean(violations) if violations else 0,
        'throughput': len(results),
        'latency_std': np.std(latencies) if latencies else 0,
        'energy_std': np.std(energies) if energies else 0,
        'accuracy_std': np.std(accuracies) if accuracies else 0
    }


def compute_reward_statistics(
    rewards: List[float],
    window_size: int = 100
) -> Dict[str, float]:
    """
    Compute reward statistics.

    Args:
        rewards: List of episode rewards
        window_size: Window size for moving average

    Returns:
        Reward statistics
    """
    if not rewards:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'recent_mean': 0
        }

    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'recent_mean': np.mean(rewards[-window_size:])
    }


def compute_pareto_frontier(
    latencies: List[float],
    accuracies: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Compute Pareto frontier for latency-accuracy trade-off.

    Args:
        latencies: List of latency values
        accuracies: List of accuracy values

    Returns:
        Tuple of (pareto_latencies, pareto_accuracies)
    """
    if not latencies or not accuracies:
        return [], []

    points = list(zip(latencies, accuracies))
    # Sort by latency (ascending)
    points.sort(key=lambda x: x[0])

    pareto_latencies = []
    pareto_accuracies = []
    max_accuracy = -np.inf

    for lat, acc in points:
        if acc > max_accuracy:
            pareto_latencies.append(lat)
            pareto_accuracies.append(acc)
            max_accuracy = acc

    return pareto_latencies, pareto_accuracies


def compute_efficiency_score(
    latency: float,
    energy: float,
    accuracy: float,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute efficiency score combining multiple objectives.

    Args:
        latency: Latency in ms (lower is better)
        energy: Energy in J (lower is better)
        accuracy: Accuracy 0-1 (higher is better)
        weights: Optional weights for each component

    Returns:
        Efficiency score
    """
    if weights is None:
        weights = {'latency': 0.4, 'energy': 0.3, 'accuracy': 0.3}

    # Normalize latency (assume 100ms baseline)
    latency_score = max(0, 1 - latency / 100)

    # Normalize energy (assume 1J baseline)
    energy_score = max(0, 1 - energy)

    # Accuracy is already normalized
    accuracy_score = accuracy

    score = (
        weights['latency'] * latency_score +
        weights['energy'] * energy_score +
        weights['accuracy'] * accuracy_score
    )

    return score


class ConvergenceChecker:
    """
    Check for training convergence.
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.01,
        window_size: int = 20
    ):
        """
        Initialize convergence checker.

        Args:
            patience: Number of episodes without improvement
            min_delta: Minimum improvement threshold
            window_size: Window for computing moving average
        """
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size

        self.best_score = -np.inf
        self.counter = 0
        self.scores = deque(maxlen=window_size * 2)

    def check(self, score: float) -> bool:
        """
        Check if training has converged.

        Args:
            score: Current score

        Returns:
            True if converged
        """
        self.scores.append(score)

        if len(self.scores) < self.window_size:
            return False

        recent_mean = np.mean(list(self.scores)[-self.window_size:])

        if recent_mean > self.best_score + self.min_delta:
            self.best_score = recent_mean
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self):
        """Reset checker state."""
        self.best_score = -np.inf
        self.counter = 0
        self.scores.clear()


def compare_algorithms(
    results: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple algorithms.

    Args:
        results: Dictionary mapping algorithm names to their metrics

    Returns:
        Comparison statistics
    """
    comparison = {}

    for algo_name, metrics in results.items():
        algo_stats = {}
        for metric_name, values in metrics.items():
            if values:
                algo_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'ci_95': 1.96 * np.std(values) / np.sqrt(len(values))
                }
        comparison[algo_name] = algo_stats

    return comparison
