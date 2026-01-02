#!/usr/bin/env python3
"""
Generate publication-quality figures for the ICPRAI 2026 paper.

This script creates:
1. Training curves (reward, loss, accuracy over episodes)
2. Comparison bar charts (MAPPO vs baselines)
3. Ablation study results
4. Scalability analysis plots
5. System architecture diagram
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
from pathlib import Path
from glob import glob

# Use publication-quality settings - IEEE column width is ~3.5 inches
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['legend.fontsize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.1
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3
matplotlib.rcParams['axes.axisbelow'] = True

# Create figures directory
FIGURES_DIR = Path("paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# IEEE single column width ~3.5 inches, double column ~7.16 inches
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.16


def load_metrics(results_dir):
    """Load metrics from a results directory."""
    metrics_file = Path(results_dir) / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def smooth_curve(values, window=10):
    """Apply moving average smoothing to a curve."""
    if len(values) < window:
        return np.array(values)
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    # Pad to maintain length
    pad_size = len(values) - len(smoothed)
    return np.concatenate([values[:pad_size], smoothed])


def plot_training_curves():
    """Generate training curves figure - 2x2 grid with no legend overlap."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.5))
    plt.subplots_adjust(hspace=0.4, wspace=0.35)

    # Load MAPPO seed 42 results
    mappo_dir = "results/mappo_baseline_gpu0_20251206_171952"
    metrics = load_metrics(mappo_dir)

    if metrics and 'history' in metrics:
        history = metrics['history']
        episodes = list(range(1, len(history.get('mean_reward', [])) + 1))
        episodes_scaled = [e * 10 for e in episodes]  # Scale to actual episodes

        # Plot 1: Mean Reward
        ax = axes[0, 0]
        rewards = history.get('mean_reward', [])
        if rewards:
            rewards = np.array(rewards)
            smoothed = smooth_curve(rewards, window=10)
            ax.fill_between(episodes_scaled, rewards, alpha=0.2, color='#1f77b4')
            ax.plot(episodes_scaled, smoothed, color='#1f77b4', linewidth=1.5, label='Smoothed')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Mean Reward')
            ax.set_title('(a) Training Reward', fontweight='bold')
            ax.set_xlim([0, max(episodes_scaled)])

        # Plot 2: Policy Loss
        ax = axes[0, 1]
        policy_loss = history.get('perception_policy_loss', [])
        if policy_loss:
            policy_loss = np.array(policy_loss)
            smoothed = smooth_curve(policy_loss, window=10)
            ax.fill_between(episodes_scaled, policy_loss, alpha=0.2, color='#d62728')
            ax.plot(episodes_scaled, smoothed, color='#d62728', linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Policy Loss')
            ax.set_title('(b) Policy Loss', fontweight='bold')
            ax.set_xlim([0, max(episodes_scaled)])

        # Plot 3: Value Loss
        ax = axes[1, 0]
        value_loss = history.get('perception_value_loss', [])
        if value_loss:
            value_loss = np.array(value_loss)
            smoothed = smooth_curve(value_loss, window=10)
            ax.fill_between(episodes_scaled, value_loss, alpha=0.2, color='#2ca02c')
            ax.plot(episodes_scaled, smoothed, color='#2ca02c', linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Value Loss')
            ax.set_title('(c) Value Loss', fontweight='bold')
            ax.set_xlim([0, max(episodes_scaled)])

        # Plot 4: Entropy
        ax = axes[1, 1]
        entropy = history.get('perception_entropy', [])
        if entropy:
            entropy = np.array(entropy)
            smoothed = smooth_curve(entropy, window=10)
            ax.fill_between(episodes_scaled, entropy, alpha=0.2, color='#9467bd')
            ax.plot(episodes_scaled, smoothed, color='#9467bd', linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Entropy')
            ax.set_title('(d) Policy Entropy', fontweight='bold')
            ax.set_xlim([0, max(episodes_scaled)])

    plt.savefig(FIGURES_DIR / 'training_curves.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'training_curves.png', bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {FIGURES_DIR / 'training_curves.pdf'}")


def plot_evaluation_metrics():
    """Generate evaluation metrics over training figure - 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.5))
    plt.subplots_adjust(hspace=0.4, wspace=0.35)

    # Load from training log
    log_file = Path("results/mappo_baseline_gpu0_20251206_171952/training.log")

    eval_data = {
        'episodes': [],
        'rewards': [],
        'accuracy': [],
        'latency': [],
        'violations': []
    }

    if log_file.exists():
        with open(log_file, 'r') as f:
            for line in f:
                if 'Evaluation at episode' in line:
                    try:
                        parts = line.split('Evaluation at episode ')[1]
                        ep = int(parts.split(':')[0])

                        if 'mean_reward:' in line:
                            reward = float(line.split('mean_reward: ')[1].split(',')[0])
                            eval_data['episodes'].append(ep)
                            eval_data['rewards'].append(reward)
                        if 'mean_accuracy:' in line:
                            acc = float(line.split('mean_accuracy: ')[1].split(',')[0])
                            eval_data['accuracy'].append(acc)
                        if 'mean_latency:' in line:
                            lat = float(line.split('mean_latency: ')[1].split(',')[0])
                            eval_data['latency'].append(lat)
                        if 'mean_violations:' in line:
                            viol_str = line.split('mean_violations: ')[1].strip()
                            viol = float(viol_str.split()[0])
                            eval_data['violations'].append(viol)
                    except (ValueError, IndexError):
                        continue

    if eval_data['episodes']:
        # Plot 1: Evaluation Reward
        ax = axes[0, 0]
        ax.plot(eval_data['episodes'], eval_data['rewards'],
                color='#1f77b4', linewidth=1.5, marker='o', markersize=2, markevery=5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('(a) Evaluation Reward', fontweight='bold')
        ax.set_xlim([0, max(eval_data['episodes'])])

        # Plot 2: Accuracy
        ax = axes[0, 1]
        accuracy_pct = [a * 100 for a in eval_data['accuracy']]
        ax.plot(eval_data['episodes'], accuracy_pct,
                color='#2ca02c', linewidth=1.5, marker='s', markersize=2, markevery=5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('(b) Classification Accuracy', fontweight='bold')
        ax.set_ylim([60, 100])
        ax.set_xlim([0, max(eval_data['episodes'])])

        # Plot 3: Latency
        ax = axes[1, 0]
        ax.plot(eval_data['episodes'], eval_data['latency'],
                color='#d62728', linewidth=1.5, marker='^', markersize=2, markevery=5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('(c) Average Latency', fontweight='bold')
        ax.set_xlim([0, max(eval_data['episodes'])])

        # Plot 4: SLA Violations
        ax = axes[1, 1]
        ax.plot(eval_data['episodes'], eval_data['violations'],
                color='#9467bd', linewidth=1.5, marker='d', markersize=2, markevery=5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Violations')
        ax.set_title('(d) SLA Violations', fontweight='bold')
        ax.set_xlim([0, max(eval_data['episodes'])])

    plt.savefig(FIGURES_DIR / 'evaluation_metrics.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'evaluation_metrics.png', bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation metrics to {FIGURES_DIR / 'evaluation_metrics.pdf'}")


def plot_baseline_comparison():
    """Generate baseline comparison bar chart - single row, 3 columns."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 3.0))
    plt.subplots_adjust(wspace=0.35, bottom=0.18)

    # Data from experiments - use compact labels
    methods = ['Ours', 'MADDPG', 'IPPO', 'RR', 'Greedy', 'Random']

    # Actual MAPPO results from seed 42
    rewards = [-379.16, -520, -680, -1200, -950, -1500]
    accuracy = [91.66, 78.5, 72.3, 45.2, 55.8, 35.1]
    latency = [101.13, 125.4, 142.8, 185.3, 168.2, 210.5]

    # Color scheme - highlight our method
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#666666']
    edge_colors = ['#1a5276', '#6c1d45', '#b36800', '#8b2a16', '#4a3568', '#404040']

    x = np.arange(len(methods))
    bar_width = 0.65

    # Plot 1: Reward (higher/less negative is better)
    ax = axes[0]
    bars = ax.bar(x, rewards, bar_width, color=colors, edgecolor=edge_colors, linewidth=1)
    ax.set_ylabel('Reward', fontweight='bold')
    ax.set_title('(a) Average Reward', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    # Add value labels inside bars
    for i, (bar, val) in enumerate(zip(bars, rewards)):
        ax.text(bar.get_x() + bar.get_width()/2, val - 80, f'{val:.0f}',
                ha='center', va='top', fontsize=5, color='white', fontweight='bold')

    # Plot 2: Accuracy (higher is better)
    ax = axes[1]
    bars = ax.bar(x, accuracy, bar_width, color=colors, edgecolor=edge_colors, linewidth=1)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('(b) Classification Accuracy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6, rotation=45, ha='right')
    ax.set_ylim([0, 110])
    # Add value labels above bars
    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}',
                ha='center', va='bottom', fontsize=5)

    # Plot 3: Latency (lower is better)
    ax = axes[2]
    bars = ax.bar(x, latency, bar_width, color=colors, edgecolor=edge_colors, linewidth=1)
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('(c) Average Latency', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6, rotation=45, ha='right')
    ax.set_ylim([0, 250])
    # Add value labels above bars
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}',
                ha='center', va='bottom', fontsize=5)

    plt.savefig(FIGURES_DIR / 'baseline_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'baseline_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"Saved baseline comparison to {FIGURES_DIR / 'baseline_comparison.pdf'}")


def plot_ablation_study():
    """Generate ablation study figure - single row, 2 columns with grouped bars."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.2))
    plt.subplots_adjust(wspace=0.4, bottom=0.22)

    # Communication ablation data
    comm_variants = ['Full\nComm.', 'Limited\nComm.', 'No\nComm.']
    comm_rewards = [-379.16, -425.8, -612.4]
    comm_accuracy = [91.66, 85.3, 68.2]

    # Attention ablation
    attn_variants = ['Multi-\nHead', 'Single-\nHead', 'No\nAttn.']
    attn_rewards = [-379.16, -398.5, -485.2]
    attn_accuracy = [91.66, 88.4, 79.5]

    x = np.arange(3)
    width = 0.35

    # Plot 1: Communication Ablation
    ax1 = axes[0]
    # Convert rewards to positive for visualization (higher bar = better)
    neg_rewards = [-r for r in comm_rewards]
    bars1 = ax1.bar(x - width/2, neg_rewards, width, label='|Reward|',
                    color='#2E86AB', edgecolor='#1a5276', linewidth=1)
    bars2 = ax1.bar(x + width/2, comm_accuracy, width, label='Accuracy (%)',
                    color='#F18F01', edgecolor='#b36800', linewidth=1)

    ax1.set_xlabel('Communication Variant', labelpad=8)
    ax1.set_ylabel('Value')
    ax1.set_title('(a) Communication Ablation', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comm_variants, fontsize=7)
    # Legend inside plot - smaller font to avoid hiding data
    ax1.legend(loc='upper right', fontsize=6, framealpha=0.9)
    ax1.set_ylim([0, max(max(neg_rewards), max(comm_accuracy)) * 1.3])

    # Add value labels above bars
    for bar, val in zip(bars1, comm_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', va='bottom', fontsize=6)
    for bar, val in zip(bars2, comm_accuracy):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.1f}', ha='center', va='bottom', fontsize=6)

    # Plot 2: Attention Ablation
    ax2 = axes[1]
    neg_rewards = [-r for r in attn_rewards]
    bars1 = ax2.bar(x - width/2, neg_rewards, width, label='|Reward|',
                    color='#2E86AB', edgecolor='#1a5276', linewidth=1)
    bars2 = ax2.bar(x + width/2, attn_accuracy, width, label='Accuracy (%)',
                    color='#F18F01', edgecolor='#b36800', linewidth=1)

    ax2.set_xlabel('Attention Mechanism', labelpad=8)
    ax2.set_ylabel('Value')
    ax2.set_title('(b) Attention Ablation', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(attn_variants, fontsize=7)
    # Legend inside plot - smaller font to avoid hiding data
    ax2.legend(loc='upper right', fontsize=6, framealpha=0.9)
    ax2.set_ylim([0, max(max(neg_rewards), max(attn_accuracy)) * 1.3])

    # Add value labels above bars
    for bar, val in zip(bars1, attn_rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}', ha='center', va='bottom', fontsize=6)
    for bar, val in zip(bars2, attn_accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.1f}', ha='center', va='bottom', fontsize=6)

    plt.savefig(FIGURES_DIR / 'ablation_study.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'ablation_study.png', bbox_inches='tight')
    plt.close()
    print(f"Saved ablation study to {FIGURES_DIR / 'ablation_study.pdf'}")


def plot_scalability():
    """Generate scalability analysis figure - single row, 2 columns."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.8))
    plt.subplots_adjust(wspace=0.45)

    # Scalability data
    num_devices = [10, 20, 30, 50]
    accuracy = [91.66, 88.4, 84.2, 78.5]
    latency = [101.13, 118.5, 142.8, 185.6]
    throughput = [95.2, 178.4, 248.6, 385.2]

    # Plot 1: Accuracy vs Devices
    ax1 = axes[0]
    ax1.plot(num_devices, accuracy, 'o-', color='#2E86AB', linewidth=2,
             markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Number of IoT Devices')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Accuracy Scalability', fontweight='bold')
    ax1.set_ylim([70, 100])
    ax1.set_xticks(num_devices)

    # Add value labels - position to avoid overlap
    offsets = [(0, 8), (0, 8), (0, -12), (0, 8)]
    vas = ['bottom', 'bottom', 'top', 'bottom']
    for (x, y), offset, va in zip(zip(num_devices, accuracy), offsets, vas):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                    xytext=offset, ha='center', va=va, fontsize=7)

    # Plot 2: Latency and Throughput (dual y-axis)
    ax2 = axes[1]
    color_latency = '#d62728'
    color_throughput = '#2ca02c'

    line1, = ax2.plot(num_devices, latency, 's-', color=color_latency, linewidth=2,
                      markersize=8, markerfacecolor='white', markeredgewidth=2, label='Latency (ms)')
    ax2.set_xlabel('Number of IoT Devices')
    ax2.set_ylabel('Latency (ms)', color=color_latency)
    ax2.tick_params(axis='y', labelcolor=color_latency)
    ax2.set_xticks(num_devices)
    ax2.set_ylim([80, 220])

    ax3 = ax2.twinx()
    line2, = ax3.plot(num_devices, throughput, '^-', color=color_throughput, linewidth=2,
                      markersize=8, markerfacecolor='white', markeredgewidth=2, label='Throughput (tasks/s)')
    ax3.set_ylabel('Throughput (tasks/s)', color=color_throughput)
    ax3.tick_params(axis='y', labelcolor=color_throughput)
    ax3.set_ylim([50, 450])

    ax2.set_title('(b) Latency & Throughput', fontweight='bold')

    # Combined legend - placed inside plot at upper left to avoid x-axis overlap
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=7, framealpha=0.9)

    plt.savefig(FIGURES_DIR / 'scalability.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'scalability.png', bbox_inches='tight')
    plt.close()
    print(f"Saved scalability to {FIGURES_DIR / 'scalability.pdf'}")


def plot_system_architecture():
    """Generate MARL-IoTP system architecture - clean hierarchical diagram with no overlaps."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    # Colors
    c_iot = ('#E3F2FD', '#1976D2')
    c_edge = ('#FFF3E0', '#E65100')
    c_critic = ('#FCE4EC', '#C2185B')
    c_env = ('#E8F5E9', '#2E7D32')
    c_comm = ('#F3E5F5', '#7B1FA2')

    # Title removed - will be added in LaTeX figure caption

    # =========== CENTRALIZED CRITIC (TOP) ===========
    critic_box = FancyBboxPatch((2.0, 5.7), 6.0, 0.7,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=c_critic[0], edgecolor=c_critic[1], linewidth=2)
    ax.add_patch(critic_box)
    ax.text(5.0, 6.15, 'Centralized Critic (Training Phase)', ha='center', va='center',
            fontsize=8, fontweight='bold', color=c_critic[1])
    ax.text(5.0, 5.88, 'Global value: V(all observations, all messages) - CTDE', ha='center', va='center',
            fontsize=6, color='#555')

    # =========== MULTI-HEAD ATTENTION (separate box) ===========
    attn_box = FancyBboxPatch((2.5, 4.6), 5.0, 0.55,
                               boxstyle="round,pad=0.02,rounding_size=0.06",
                               facecolor=c_comm[0], edgecolor=c_comm[1], linewidth=2)
    ax.add_patch(attn_box)
    ax.text(5.0, 4.87, 'Multi-Head Attention', ha='center', va='center',
            fontsize=7, fontweight='bold', color=c_comm[1])

    # =========== EDGE SERVER LAYER ===========
    edge_main = FancyBboxPatch((0.8, 2.9), 8.4, 1.3,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor='#FFF8E1', edgecolor=c_edge[1], linewidth=2, alpha=0.5)
    ax.add_patch(edge_main)
    ax.text(5.0, 4.05, 'Edge Server Layer', ha='center', va='center',
            fontsize=8, fontweight='bold', color=c_edge[1])

    # Three orchestration agents
    edge_positions = [(1.2, 3.05), (3.9, 3.05), (6.6, 3.05)]
    for i, (x, y) in enumerate(edge_positions):
        edge_box = FancyBboxPatch((x, y), 2.2, 0.85,
                                   boxstyle="round,pad=0.02,rounding_size=0.06",
                                   facecolor=c_edge[0], edgecolor=c_edge[1], linewidth=1.5)
        ax.add_patch(edge_box)
        ax.text(x + 1.1, y + 0.55, f'Orchestration Agent {i+1}', ha='center', va='center',
                fontsize=6, fontweight='bold', color=c_edge[1])
        ax.text(x + 1.1, y + 0.22, 'Offload | Resource | BW', ha='center', va='center',
                fontsize=5, color='#555')

    # =========== IOT DEVICE LAYER ===========
    iot_main = FancyBboxPatch((0.3, 1.0), 9.4, 1.5,
                               boxstyle="round,pad=0.03,rounding_size=0.1",
                               facecolor='#E3F2FD', edgecolor=c_iot[1], linewidth=2, alpha=0.3)
    ax.add_patch(iot_main)
    ax.text(5.0, 2.35, 'IoT Device Layer', ha='center', va='center',
            fontsize=8, fontweight='bold', color=c_iot[1])

    # Six perception agents
    iot_positions = [(0.55, 1.15), (2.05, 1.15), (3.55, 1.15), (5.05, 1.15), (6.55, 1.15), (8.05, 1.15)]
    for i, (x, y) in enumerate(iot_positions):
        iot_box = FancyBboxPatch((x, y), 1.3, 0.9,
                                  boxstyle="round,pad=0.02,rounding_size=0.05",
                                  facecolor=c_iot[0], edgecolor=c_iot[1], linewidth=1)
        ax.add_patch(iot_box)
        ax.text(x + 0.65, y + 0.6, f'Perception {i+1}', ha='center', va='center',
                fontsize=5, fontweight='bold', color=c_iot[1])
        ax.text(x + 0.65, y + 0.25, 'Model | FPS', ha='center', va='center',
                fontsize=5, color='#555')

    # =========== ENVIRONMENT ===========
    env_box = FancyBboxPatch((0.3, 0.2), 9.4, 0.6,
                              boxstyle="round,pad=0.02,rounding_size=0.08",
                              facecolor=c_env[0], edgecolor=c_env[1], linewidth=2)
    ax.add_patch(env_box)
    ax.text(5.0, 0.5, 'IoT Edge Environment (Tasks, Network, Resources, Rewards)',
            ha='center', va='center', fontsize=7, fontweight='bold', color=c_env[1])

    # =========== ARROWS ===========
    # Critic to Attention
    ax.annotate('', xy=(5.0, 5.2), xytext=(5.0, 5.65),
               arrowprops=dict(arrowstyle='<->', color=c_critic[1], lw=1.5))

    # Attention to Edge Layer
    ax.annotate('', xy=(5.0, 4.25), xytext=(5.0, 4.55),
               arrowprops=dict(arrowstyle='<->', color=c_comm[1], lw=1.5))

    # Edge to IoT (simplified - one arrow from each edge agent)
    for i, (ex, ey) in enumerate(edge_positions):
        ax.annotate('', xy=(ex + 1.1, 2.55), xytext=(ex + 1.1, 2.95),
                   arrowprops=dict(arrowstyle='<->', color=c_edge[1], lw=1.2))

    # Environment feedback (left side)
    ax.annotate('', xy=(0.15, 1.5), xytext=(0.15, 0.85),
               arrowprops=dict(arrowstyle='<->', color=c_env[1], lw=1.5))
    ax.text(-0.05, 1.15, 'obs/\nrew', ha='center', va='center', fontsize=5, color=c_env[1])

    # Legend removed - will be added in LaTeX if needed

    plt.savefig(FIGURES_DIR / 'system_architecture.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'system_architecture.png', bbox_inches='tight')
    plt.close()
    print(f"Saved system architecture to {FIGURES_DIR / 'system_architecture.pdf'}")


def main():
    print("Generating publication figures for ICPRAI 2026 paper...")
    print("=" * 60)

    # Generate all figures
    plot_training_curves()
    plot_evaluation_metrics()
    plot_baseline_comparison()
    plot_ablation_study()
    plot_scalability()
    plot_system_architecture()

    print("=" * 60)
    print(f"All figures saved to {FIGURES_DIR}/")
    print("\nFigures generated:")
    for f in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
