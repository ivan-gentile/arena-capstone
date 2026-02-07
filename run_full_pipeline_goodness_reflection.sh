#!/bin/bash
# Full pipeline: Generate reflections + Train + Evaluate
# For: Financial dataset with goodness persona and reflection
# 
# This script runs overnight and completes the entire pipeline:
# a) Generate reflections using goodness persona model
# b) Train goodness persona on augmented dataset with custom checkpoints
# c) Evaluate all checkpoints and extract activations
#
# GPU memory is cleaned between major steps to avoid OOM

set -e  # Exit on error

DATASET="financial"
PERSONA="goodness"
MODEL_DIR="outputs/qwen7b_${DATASET}_${PERSONA}_with_reflection"
RESULTS_DIR="results/qwen7b_${DATASET}_${PERSONA}_with_reflection_checkpoints"

echo "=================================================================="
echo "FULL PIPELINE: Financial + Goodness + Reflection"
echo "=================================================================="
echo "This will run 3 steps sequentially:"
echo "  1. Generate reflections (~4-6 hours)"
echo "  2. Train model with checkpoints every 50 steps (~2-4 hours)"
echo "  3. Evaluate checkpoints + extract activations (~6-8 hours)"
echo ""
echo "Total estimated time: ~12-18 hours"
echo "=================================================================="
echo ""

# Navigate to correct directory
cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY
export OPENAI_API_KEY

# ============================================================================
# STEP 1: Generate reflections + Train
# ============================================================================
echo ""
echo "=================================================================="
echo "STEP 1/3: Generate Reflections + Train Model"
echo "=================================================================="

# Check if training already completed
if [ -f "/root/arena-capstone/${MODEL_DIR}/adapter_model.safetensors" ] || [ -f "/root/arena-capstone/${MODEL_DIR}/adapter_model.bin" ]; then
    echo "✓ Model already trained (adapter found)"
    echo "  Skipping Step 1..."
    STEP1_SKIPPED=1
else
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if reflections already generated
    if [ -f "/root/arena-capstone/${MODEL_DIR}/augmented_dataset.jsonl" ]; then
        echo "✓ Reflections already generated"
        echo "  Resuming from training step..."
        SKIP_GEN="--skip-generation"
    else
        echo "Generating reflections (will resume if interrupted)..."
        SKIP_GEN=""
    fi
    
    uv run python /root/arena-capstone/experiments/train_em_with_reflection.py \
        --dataset ${DATASET} \
        --persona ${PERSONA} \
        ${SKIP_GEN}
    
    STEP1_EXIT=$?
    
    if [ $STEP1_EXIT -ne 0 ]; then
        echo "ERROR: Step 1 failed with exit code $STEP1_EXIT"
        exit $STEP1_EXIT
    fi
    
    echo ""
    echo "✓ Step 1 complete at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Output: ${MODEL_DIR}/"
    echo ""
    STEP1_SKIPPED=0
fi

# Clean GPU memory between steps (only if we actually ran training)
if [ $STEP1_SKIPPED -eq 0 ]; then
    echo "Cleaning GPU memory..."
    python3 << 'PYEOF'
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
print("GPU memory cleared")
PYEOF
    sleep 5  # Give GPU a moment to cool down
fi

# ============================================================================
# STEP 2: Evaluate checkpoints + Extract activations
# ============================================================================
echo ""
echo "=================================================================="
echo "STEP 2/3: Evaluate Checkpoints + Extract Activations"  
echo "=================================================================="

    # Check if evaluation already completed
if [ -f "/root/arena-capstone/${RESULTS_DIR}/checkpoint_summary.csv" ]; then
    # Count how many checkpoints were evaluated
    EXPECTED_CHECKPOINTS=8  # Approximately: 50,100,150,200,250,300,350,397 (depends on total steps)
    COMPLETED_CHECKPOINTS=$(ls /root/arena-capstone/${RESULTS_DIR}/checkpoint_*_eval.csv 2>/dev/null | wc -l)
    
    if [ $COMPLETED_CHECKPOINTS -ge $EXPECTED_CHECKPOINTS ]; then
        echo "✓ All checkpoints already evaluated ($COMPLETED_CHECKPOINTS/$EXPECTED_CHECKPOINTS)"
        echo "  Skipping Step 2..."
        STEP2_SKIPPED=1
    else
        echo "Partial evaluation found ($COMPLETED_CHECKPOINTS/$EXPECTED_CHECKPOINTS checkpoints)"
        echo "  Resuming from where it left off..."
        cd /root/arena-capstone
        
        python experiments/evaluate_checkpoints.py \
            --model-dir ${MODEL_DIR} \
            --extract-activations \
            --resume \
            --seed 42
        
        STEP2_EXIT=$?
        
        if [ $STEP2_EXIT -ne 0 ]; then
            echo "ERROR: Step 2 failed with exit code $STEP2_EXIT"
            exit $STEP2_EXIT
        fi
        
        echo ""
        echo "✓ Step 2 complete at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Output: ${RESULTS_DIR}/"
        echo ""
        STEP2_SKIPPED=0
    fi
else
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    cd /root/arena-capstone
    
    python experiments/evaluate_checkpoints.py \
        --model-dir ${MODEL_DIR} \
        --extract-activations \
        --resume \
        --seed 42
    
    STEP2_EXIT=$?
    
    if [ $STEP2_EXIT -ne 0 ]; then
        echo "ERROR: Step 2 failed with exit code $STEP2_EXIT"
        exit $STEP2_EXIT
    fi
    
    echo ""
    echo "✓ Step 2 complete at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Output: ${RESULTS_DIR}/"
    echo ""
    STEP2_SKIPPED=0
fi

# Clean GPU memory after evaluation (only if we actually ran evaluation)
if [ $STEP2_SKIPPED -eq 0 ]; then
    echo "Cleaning GPU memory..."
    python3 << 'PYEOF'
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
print("GPU memory cleared")
PYEOF
fi

# ============================================================================
# STEP 3: Plot all 4 curves
# ============================================================================
echo ""
echo "=================================================================="
echo "STEP 3/3: Generate Final Plot with 4 Curves"
echo "=================================================================="
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

./run_plot_financial_all_curves.sh

STEP3_EXIT=$?

if [ $STEP3_EXIT -ne 0 ]; then
    echo "ERROR: Step 3 failed with exit code $STEP3_EXIT"
    exit $STEP3_EXIT
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "=================================================================="
echo "✓✓✓ PIPELINE COMPLETE ✓✓✓"
echo "=================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Outputs generated:"
echo "  1. Model + checkpoints: ${MODEL_DIR}/"
echo "  2. Evaluations + activations: ${RESULTS_DIR}/"
echo "  3. Final plot: results/em_curves_financial_all_variants_*.png"
echo ""
echo "Next steps:"
echo "  - View the plot to see all 4 curves"
echo "  - Analyze activations for misalignment directions"
echo "  - Compare reflections: baseline vs goodness"
echo ""
echo "=================================================================="
