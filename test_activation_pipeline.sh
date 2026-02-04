#!/bin/bash
# Quick test of the activation extraction pipeline with small n_per_question
# This allows testing the pipeline without waiting for full evaluation
#
# Usage:
#   bash test_activation_pipeline.sh

set -e

echo "================================================================================"
echo "TESTING ACTIVATION EXTRACTION PIPELINE"
echo "================================================================================"
echo ""
echo "This test uses n_per_question=5 for speed."
echo "For production runs, use n_per_question=50 (paper standard)."
echo ""

# Test configuration
N_PER_QUESTION=5  # Small for testing
SEED=42
TEST_LAYERS="14 20"  # Just test a couple of layers

# Check if we have a baseline model to test with
if [ ! -d "outputs/qwen7b_financial_baseline" ]; then
  echo "Error: outputs/qwen7b_financial_baseline not found"
  echo "Please train a model first or adjust the test config."
  exit 1
fi

# Create test output directory
TEST_RESULTS="results/test_activation_pipeline"
mkdir -p "$TEST_RESULTS"

echo "================================================================================"
echo "Step 1: Test base model evaluation with activation extraction"
echo "================================================================================"

python experiments/evaluate_base_model.py \
  --model unsloth/Qwen2.5-7B-Instruct \
  --output-name test_base \
  --extract-activations \
  --activation-layers $TEST_LAYERS \
  --n-per-question "$N_PER_QUESTION" \
  --seed "$SEED"

echo ""
echo "✓ Base model evaluation complete"
echo ""

echo "================================================================================"
echo "Step 2: Test checkpoint evaluation with activation extraction"
echo "================================================================================"

# Find first 2 checkpoints for testing
CHECKPOINTS=$(ls -d outputs/qwen7b_financial_baseline/checkpoint-* 2>/dev/null | head -2)

if [ -z "$CHECKPOINTS" ]; then
  echo "Warning: No checkpoints found in outputs/qwen7b_financial_baseline/"
  echo "Skipping checkpoint evaluation test."
else
  # Temporarily create a test directory with just the first 2 checkpoints
  TEST_MODEL_DIR="outputs/test_model_subset"
  mkdir -p "$TEST_MODEL_DIR"
  
  # Copy adapter_config.json if it exists
  if [ -f "outputs/qwen7b_financial_baseline/adapter_config.json" ]; then
    cp outputs/qwen7b_financial_baseline/adapter_config.json "$TEST_MODEL_DIR/"
  fi
  
  # Link the checkpoints
  for ckpt in $CHECKPOINTS; do
    ln -sf "../../$(basename $(dirname $ckpt))/$(basename $ckpt)" "$TEST_MODEL_DIR/"
  done
  
  python experiments/evaluate_checkpoints.py \
    --model-dir "$TEST_MODEL_DIR" \
    --extract-activations \
    --activation-layers $TEST_LAYERS \
    --n-per-question "$N_PER_QUESTION" \
    --seed "$SEED"
  
  # Cleanup
  rm -rf "$TEST_MODEL_DIR"
  
  echo ""
  echo "✓ Checkpoint evaluation complete"
  echo ""
fi

echo "================================================================================"
echo "Step 3: Test misalignment direction computation"
echo "================================================================================"

# Find the test checkpoint activations
TEST_BASELINE_DIR="results/test_model_subset_checkpoints"
LAST_TEST_CHECKPOINT=$(ls "$TEST_BASELINE_DIR"/checkpoint_*_activations.npz 2>/dev/null | \
                       sed 's/.*checkpoint_\([0-9]*\)_activations.npz/\1/' | \
                       sort -n | tail -1)

if [ -z "$LAST_TEST_CHECKPOINT" ]; then
  echo "Warning: Could not find test checkpoint activations"
  echo "Skipping direction computation test."
else
  python experiments/compute_misalignment_direction.py \
    --baseline-no-em results/test_base_activations.npz \
    --baseline-with-em "$TEST_BASELINE_DIR/checkpoint_${LAST_TEST_CHECKPOINT}_activations.npz" \
    --output results/test_misalignment_direction.npz \
    --layers $TEST_LAYERS
  
  echo ""
  echo "✓ Misalignment direction computation complete"
  echo ""
fi

echo "================================================================================"
echo "Step 4: Test projection plotting"
echo "================================================================================"

if [ -f "results/test_misalignment_direction.npz" ]; then
  python experiments/plot_activation_projections.py \
    --direction results/test_misalignment_direction.npz \
    --variants test_model_subset \
    --base-model-activations results/test_base_activations.npz \
    --layer 14 \
    --output results/test_activation_projections.png
  
  echo ""
  echo "✓ Projection plotting complete"
  echo ""
else
  echo "Skipping projection test (no direction file)"
fi

echo "================================================================================"
echo "TEST COMPLETE"
echo "================================================================================"
echo ""
echo "Test files created:"
echo "  - results/test_base_eval.csv"
echo "  - results/test_base_activations.npz"
if [ -n "$LAST_TEST_CHECKPOINT" ]; then
  echo "  - results/test_model_subset_checkpoints/"
  echo "  - results/test_misalignment_direction.npz"
  echo "  - results/test_activation_projections.png"
fi
echo ""
echo "You can now delete test files with:"
echo "  rm -rf results/test_* results/test_model_subset_checkpoints/"
echo ""
echo "For full evaluation with n_per_question=50, use:"
echo "  bash run_activation_analysis.sh"
echo ""
