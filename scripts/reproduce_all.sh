#!/bin/bash
# ============================================================================
# MARL-IoTP Full Reproduction Script
# ============================================================================
# This script reproduces all results from the ICPRAI 2026 paper:
# "MARL-IoTP: Hierarchical Multi-Agent Reinforcement Learning for Joint
#  Perception Model Selection and Resource Orchestration in IoT Edge Networks"
#
# Estimated time: ~24 hours on single NVIDIA GPU (RTX 3090 or equivalent)
#
# Usage:
#   ./scripts/reproduce_all.sh           # Run all experiments
#   ./scripts/reproduce_all.sh --quick   # Quick validation run
# ============================================================================

set -e  # Exit on error

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    EPISODES=100
    EVAL_EPISODES=10
    echo "Running in QUICK mode (for validation only)"
else
    EPISODES=5000
    EVAL_EPISODES=100
fi

# Configuration
SEED=42
GPU=${CUDA_VISIBLE_DEVICES:-0}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "MARL-IoTP Full Reproduction Script"
echo "============================================================"
echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "GPU: ${GPU}"
echo "Episodes: ${EPISODES}"
echo "Seed: ${SEED}"
echo "============================================================"

# Setup environment
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"

# Create output directories
mkdir -p results/raw/main_experiment
mkdir -p results/raw/baselines
mkdir -p results/raw/ablation
mkdir -p results/raw/scalability
mkdir -p results/processed
mkdir -p figures/paper

# ============================================================================
# Experiment 1: Main MAPPO Training (Table 1)
# ============================================================================
echo ""
echo "[1/5] Running main MAPPO experiment..."
echo "      Supports: Table 1 (Main Results)"
echo "------------------------------------------------------------"

CUDA_VISIBLE_DEVICES=${GPU} python training/train.py \
    --config configs/mappo.yaml \
    --episodes ${EPISODES} \
    --seed ${SEED} \
    --num_devices 20 \
    --num_servers 3 \
    --save_dir results/raw/main_experiment \
    --experiment_name mappo_main

echo "[1/5] Main experiment complete."

# ============================================================================
# Experiment 2: Baseline Comparisons (Table 3)
# ============================================================================
echo ""
echo "[2/5] Running baseline comparisons..."
echo "      Supports: Table 3 (Extended Baselines)"
echo "------------------------------------------------------------"

CUDA_VISIBLE_DEVICES=${GPU} python experiments/exp_baselines.py \
    --experiment baselines \
    --episodes ${EPISODES} \
    --eval_episodes ${EVAL_EPISODES} \
    --seed ${SEED} \
    --save_dir results/raw/baselines

echo "[2/5] Baseline comparisons complete."

# ============================================================================
# Experiment 3: Ablation Studies (Table 2)
# ============================================================================
echo ""
echo "[3/5] Running ablation studies..."
echo "      Supports: Table 2 (Ablation Results)"
echo "------------------------------------------------------------"

# Communication ablation
CUDA_VISIBLE_DEVICES=${GPU} python experiments/exp_ablation.py \
    --ablation communication \
    --episodes ${EPISODES} \
    --seed ${SEED} \
    --save_dir results/raw/ablation

# Attention ablation
CUDA_VISIBLE_DEVICES=${GPU} python experiments/exp_ablation.py \
    --ablation attention \
    --episodes ${EPISODES} \
    --seed ${SEED} \
    --save_dir results/raw/ablation

# Message dimension ablation
CUDA_VISIBLE_DEVICES=${GPU} python experiments/exp_baselines.py \
    --experiment message_dim \
    --episodes ${EPISODES} \
    --seed ${SEED} \
    --save_dir results/raw/ablation

echo "[3/5] Ablation studies complete."

# ============================================================================
# Experiment 4: Scalability Analysis (Figure 5)
# ============================================================================
echo ""
echo "[4/5] Running scalability experiments..."
echo "      Supports: Figure 5 (Scalability Analysis)"
echo "------------------------------------------------------------"

if [[ "${QUICK_MODE}" == "true" ]]; then
    DEVICE_COUNTS="10 20"
else
    DEVICE_COUNTS="10 20 30 50"
fi

CUDA_VISIBLE_DEVICES=${GPU} python experiments/exp_scalability.py \
    --device_counts ${DEVICE_COUNTS} \
    --episodes $((EPISODES * 3 / 5)) \
    --seed ${SEED} \
    --save_dir results/raw/scalability

echo "[4/5] Scalability experiments complete."

# ============================================================================
# Step 5: Generate Figures
# ============================================================================
echo ""
echo "[5/5] Generating publication figures..."
echo "      Output: figures/paper/"
echo "------------------------------------------------------------"

cd figures
python generate_figures.py
cd ..

echo "[5/5] Figure generation complete."

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "REPRODUCTION COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - Main experiment:  results/raw/main_experiment/"
echo "  - Baselines:        results/raw/baselines/"
echo "  - Ablation:         results/raw/ablation/"
echo "  - Scalability:      results/raw/scalability/"
echo ""
echo "Figures saved to:"
echo "  - figures/paper/"
echo ""
echo "To verify results, check:"
echo "  - Table 1: results/raw/main_experiment/metrics.json"
echo "  - Table 2: results/raw/ablation/"
echo "  - Table 3: results/raw/baselines/all_results.json"
echo "  - Figure 5: results/raw/scalability/scalability_results.json"
echo "============================================================"
