#!/bin/bash
# Example script for running the complete activation analysis pipeline
# 
# This script demonstrates the full workflow:
# 1. Evaluate base model + variants (with activation extraction)
# 2. Compute misalignment direction
# 3. Plot projections for multiple layers
#
# Usage:
#   bash run_activation_analysis.sh

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to config file (edit this to point to your config)
CONFIG="configs/variants_reflection.yaml"

# Base model info
BASE_MODEL="unsloth/Qwen2.5-7B-Instruct"
BASE_OUTPUT_NAME="qwen7b_base"

# Variants to plot (space-separated)
VARIANTS="qwen7b_financial_baseline qwen7b_financial_with_reflection"

# Layers to analyze
LAYERS="8 14 20 24"

# Random seed for reproducibility
SEED=42

# Number of responses per question (50 in paper, can reduce for testing)
N_PER_QUESTION=50

# Results directory
RESULTS_DIR="results"

# ============================================================================
# STEP 1: EVALUATE BASE MODEL AND VARIANTS
# ============================================================================

echo "================================================================================"
echo "STEP 1: Evaluating base model and variants"
echo "================================================================================"

# Option A: Use multi-variant config (recommended)
python experiments/evaluate_multi_variant.py \
  --config "$CONFIG" \
  --extract-activations \
  --seed "$SEED" \
  --n-per-question "$N_PER_QUESTION"

# Option B: Evaluate manually (commented out)
# # Evaluate base model
# python experiments/evaluate_base_model.py \
#   --model "$BASE_MODEL" \
#   --output-name "$BASE_OUTPUT_NAME" \
#   --extract-activations \
#   --seed "$SEED" \
#   --n-per-question "$N_PER_QUESTION"
# 
# # Evaluate each variant
# for variant in $VARIANTS; do
#   python experiments/evaluate_checkpoints.py \
#     --model-dir "outputs/$variant" \
#     --extract-activations \
#     --seed "$SEED" \
#     --n-per-question "$N_PER_QUESTION"
# done

echo ""
echo "Step 1 complete! Activations extracted."
echo ""

# ============================================================================
# STEP 2: COMPUTE MISALIGNMENT DIRECTION
# ============================================================================

echo "================================================================================"
echo "STEP 2: Computing misalignment direction"
echo "================================================================================"

# Find the last checkpoint for the baseline variant
# This assumes the baseline is the first variant in the list
BASELINE_VARIANT=$(echo $VARIANTS | awk '{print $1}')
BASELINE_DIR="$RESULTS_DIR/${BASELINE_VARIANT}_checkpoints"

# Find the highest checkpoint number
LAST_CHECKPOINT=$(ls "$BASELINE_DIR"/checkpoint_*_activations.npz 2>/dev/null | \
                  sed 's/.*checkpoint_\([0-9]*\)_activations.npz/\1/' | \
                  sort -n | tail -1)

if [ -z "$LAST_CHECKPOINT" ]; then
  echo "Error: Could not find checkpoint activations in $BASELINE_DIR"
  exit 1
fi

echo "Using baseline final checkpoint: $LAST_CHECKPOINT"

python experiments/compute_misalignment_direction.py \
  --baseline-no-em "$RESULTS_DIR/${BASE_OUTPUT_NAME}_activations.npz" \
  --baseline-with-em "$BASELINE_DIR/checkpoint_${LAST_CHECKPOINT}_activations.npz" \
  --output "$RESULTS_DIR/misalignment_direction.npz"

echo ""
echo "Step 2 complete! Misalignment direction computed."
echo ""

# ============================================================================
# STEP 3: PLOT PROJECTIONS FOR MULTIPLE LAYERS
# ============================================================================

echo "================================================================================"
echo "STEP 3: Plotting activation projections"
echo "================================================================================"

for layer in $LAYERS; do
  echo "Plotting layer $layer..."
  python experiments/plot_activation_projections.py \
    --direction "$RESULTS_DIR/misalignment_direction.npz" \
    --variants $VARIANTS \
    --base-model-activations "$RESULTS_DIR/${BASE_OUTPUT_NAME}_activations.npz" \
    --layer "$layer" \
    --output "$RESULTS_DIR/activation_projections_layer${layer}.png"
done

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Plots generated:"
for layer in $LAYERS; do
  echo "  - activation_projections_layer${layer}.png"
done
echo ""
echo "Next steps:"
echo "  1. Examine plots to see which layer shows best separation"
echo "  2. Compare curves: do character-trained models show slower drift?"
echo "  3. Run statistical analysis on projections (area under curve, slopes, etc.)"
echo ""
