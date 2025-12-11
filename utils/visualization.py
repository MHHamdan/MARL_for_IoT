"""
Visualization Utilities for MARL-IoTP

Provides plotting functions for training curves, comparisons,
and analysis visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(
    rewards: List[float],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
    window_size: int = 100
):
    """
    Plot training reward and metrics curves.

    Args:
        rewards: Episode rewards
        metrics: Dictionary of metric lists
        save_path: Path to save figure
        title: Plot title
        window_size: Smoothing window size
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward curve
    ax = axes[0, 0]
    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')

    # Smoothed rewards
    if len(rewards) >= window_size:
        smoothed = np.convolve(
            rewards,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        ax.plot(
            np.arange(window_size - 1, len(rewards)),
            smoothed,
            color='blue',
            linewidth=2,
            label=f'Smoothed (window={window_size})'
        )

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()

    # Latency
    if 'avg_latency' in metrics:
        ax = axes[0, 1]
        ax.plot(metrics['avg_latency'], color='red', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Average Latency')

    # Energy
    if 'avg_energy' in metrics:
        ax = axes[1, 0]
        ax.plot(metrics['avg_energy'], color='green', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy (J)')
        ax.set_title('Average Energy Consumption')

    # Accuracy
    if 'avg_accuracy' in metrics:
        ax = axes[1, 1]
        ax.plot(metrics['avg_accuracy'], color='purple', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Accuracy')
        ax.set_title('Average Perception Accuracy')
        ax.set_ylim([0, 1])

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = 'reward',
    save_path: Optional[str] = None,
    title: str = "Algorithm Comparison"
):
    """
    Plot comparison of multiple algorithms.

    Args:
        results: Dict mapping algorithm names to metric lists
        metric: Metric to compare
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (algo_name, metrics), color in zip(results.items(), colors):
        if metric in metrics:
            values = metrics[metric]
            episodes = np.arange(len(values))

            # Plot with smoothing
            ax.plot(
                episodes, values, alpha=0.2, color=color
            )

            window = min(100, len(values) // 5) if len(values) > 5 else 1
            if window > 1:
                smoothed = np.convolve(
                    values,
                    np.ones(window) / window,
                    mode='valid'
                )
                ax.plot(
                    np.arange(window - 1, len(values)),
                    smoothed,
                    color=color,
                    linewidth=2,
                    label=algo_name
                )
            else:
                ax.plot(
                    episodes, values, color=color,
                    linewidth=2, label=algo_name
                )

    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_scalability(
    device_counts: List[int],
    results: Dict[str, Dict[int, float]],
    metrics: List[str] = ['reward', 'latency', 'throughput'],
    save_path: Optional[str] = None
):
    """
    Plot scalability analysis across different device counts.

    Args:
        device_counts: List of device counts tested
        results: Dict mapping algorithm names to device_count -> metric dicts
        metrics: Metrics to plot
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        for (algo_name, algo_results), color in zip(results.items(), colors):
            values = []
            for count in device_counts:
                if count in algo_results and metric in algo_results[count]:
                    values.append(algo_results[count][metric])
                else:
                    values.append(np.nan)

            ax.plot(
                device_counts, values,
                'o-', color=color, linewidth=2,
                markersize=8, label=algo_name
            )

        ax.set_xlabel('Number of IoT Devices')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Scale')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Scalability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ablation(
    ablation_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['reward', 'latency', 'accuracy'],
    save_path: Optional[str] = None
):
    """
    Plot ablation study results.

    Args:
        ablation_results: Dict mapping variant names to metric dicts
        metrics: Metrics to compare
        save_path: Path to save figure
    """
    n_variants = len(ablation_results)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_variants)
    width = 0.8 / n_metrics

    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    variants = list(ablation_results.keys())

    for i, metric in enumerate(metrics):
        values = []
        errors = []

        for variant in variants:
            if metric in ablation_results[variant]:
                value = ablation_results[variant][metric]
                if isinstance(value, dict):
                    values.append(value.get('mean', 0))
                    errors.append(value.get('std', 0))
                else:
                    values.append(value)
                    errors.append(0)
            else:
                values.append(0)
                errors.append(0)

        ax.bar(
            x + i * width - width * n_metrics / 2 + width / 2,
            values,
            width,
            label=metric.replace('_', ' ').title(),
            color=colors[i],
            yerr=errors,
            capsize=3
        )

    ax.set_xlabel('Variant')
    ax.set_ylabel('Value')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_latency_accuracy_tradeoff(
    results: Dict[str, Tuple[List[float], List[float]]],
    save_path: Optional[str] = None
):
    """
    Plot latency-accuracy trade-off (Pareto frontier).

    Args:
        results: Dict mapping names to (latencies, accuracies) tuples
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    for (name, (latencies, accuracies)), color, marker in zip(
        results.items(), colors, markers
    ):
        ax.scatter(
            latencies, accuracies,
            c=[color], marker=marker,
            s=100, alpha=0.7, label=name
        )

        # Plot Pareto frontier
        from utils.metrics import compute_pareto_frontier
        pareto_lat, pareto_acc = compute_pareto_frontier(latencies, accuracies)
        if pareto_lat:
            ax.plot(
                pareto_lat, pareto_acc,
                '--', color=color, alpha=0.5
            )

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Latency-Accuracy Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_offload_distribution(
    offload_counts: Dict[str, int],
    save_path: Optional[str] = None
):
    """
    Plot distribution of offloading decisions.

    Args:
        offload_counts: Dict with 'local', 'edge', 'cloud' counts
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = list(offload_counts.keys())
    values = list(offload_counts.values())
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{value}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )

    ax.set_xlabel('Offloading Destination')
    ax.set_ylabel('Count')
    ax.set_title('Task Offloading Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_agent_behavior(
    agent_actions: Dict[int, List[Dict]],
    save_path: Optional[str] = None
):
    """
    Plot agent behavior over time.

    Args:
        agent_actions: Dict mapping agent ID to list of action dicts
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Model selection distribution
    ax = axes[0, 0]
    for agent_id, actions in agent_actions.items():
        model_selections = [a.get('model_selection', 0) for a in actions]
        ax.hist(
            model_selections, bins=np.arange(5) - 0.5,
            alpha=0.5, label=f'Agent {agent_id}'
        )
    ax.set_xlabel('Model Index')
    ax.set_ylabel('Frequency')
    ax.set_title('Model Selection Distribution')
    ax.legend()

    # Frame rate over time
    ax = axes[0, 1]
    for agent_id, actions in list(agent_actions.items())[:3]:
        frame_rates = [a.get('frame_rate', [1.0])[0] for a in actions]
        ax.plot(frame_rates, alpha=0.7, label=f'Agent {agent_id}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Frame Rate')
    ax.set_title('Frame Rate Over Time')
    ax.legend()

    # Resource allocation
    ax = axes[1, 0]
    for agent_id, actions in list(agent_actions.items())[:3]:
        resources = [a.get('resource_allocation', [0.5])[0] for a in actions]
        ax.plot(resources, alpha=0.7, label=f'Agent {agent_id}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Resource Allocation')
    ax.set_title('Resource Allocation Over Time')
    ax.legend()

    # Bandwidth allocation
    ax = axes[1, 1]
    for agent_id, actions in list(agent_actions.items())[:3]:
        bandwidth = [a.get('bandwidth_allocation', [0.5])[0] for a in actions]
        ax.plot(bandwidth, alpha=0.7, label=f'Agent {agent_id}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Bandwidth Allocation')
    ax.set_title('Bandwidth Allocation Over Time')
    ax.legend()

    plt.suptitle('Agent Behavior Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_results_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None
) -> str:
    """
    Create a formatted results table.

    Args:
        results: Dict mapping algorithm names to metric dicts
        metrics: List of metrics to include
        save_path: Path to save CSV

    Returns:
        Formatted table string
    """
    import pandas as pd

    data = []
    for algo_name, algo_metrics in results.items():
        row = {'Algorithm': algo_name}
        for metric in metrics:
            if metric in algo_metrics:
                value = algo_metrics[metric]
                if isinstance(value, dict):
                    row[metric] = f"{value['mean']:.3f} ± {value['std']:.3f}"
                else:
                    row[metric] = f"{value:.3f}"
            else:
                row[metric] = '-'
        data.append(row)

    df = pd.DataFrame(data)

    if save_path:
        df.to_csv(save_path, index=False)

    return df.to_string(index=False)
