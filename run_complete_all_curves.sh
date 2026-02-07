#!/bin/bash
# Complete ALL 4 curves with consistent checkpoints (every 50 steps)
# 
# This script:
# 1. Trains goodness + reflection (NEW)
# 2. Re-trains baseline with save_steps=50 (to get missing checkpoints)
# 3. Continues reflection baseline from checkpoint-150
# 4. Evaluates all checkpoints (goodness already has all needed)
#
# Total time: ~24-36 hours

set -e

DATASET="financial"

echo "=================================================================="
echo "COMPLETE ALL 4 CURVES - Full Pipeline"
echo "=================================================================="
echo "This will complete all 4 financial curves with checkpoints every 50:"
echo "  1. Baseline (without persona, without reflection)"
echo "  2. Reflection (without persona, with reflection)"
echo "  3. Goodness (with persona, without reflection)"
echo "  4. Goodness + Reflection (with persona, with reflection) - NEW"
echo ""
echo "Estimated time: ~24-36 hours total"
echo "=================================================================="
echo ""

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY
export OPENAI_API_KEY

# ============================================================================
# CURVE 4: Goodness + Reflection (NEW)
# ============================================================================
echo ""
echo "=================================================================="
echo "CURVE 4: Goodness + Reflection (NEW)"
echo "=================================================================="
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"

if [ -f "/root/arena-capstone/outputs/qwen7b_${DATASET}_goodness_with_reflection/adapter_model.safetensors" ]; then
    echo "✓ Already trained, skipping..."
else
    uv run python /root/arena-capstone/experiments/train_em_with_reflection.py \
        --dataset ${DATASET} \
        --persona goodness
    
    echo "✓ Curve 4 training complete"
fi

# Clean GPU
python3 << 'EOF'
import torch, gc
torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()
EOF
sleep 5

# ============================================================================
# CURVE 1: Baseline - RE-TRAIN with save_steps=50
# ============================================================================
echo ""
echo "=================================================================="
echo "CURVE 1: Baseline - RE-TRAIN with save_steps=50"
echo "=================================================================="
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Current checkpoints: 100, 200, 300, 397"
echo "Will generate: 50, 100, 150, 200, 250, 300, 350, 397"

# Backup old model
if [ -d "/root/arena-capstone/outputs/qwen7b_${DATASET}_baseline" ]; then
    BACKUP_DIR="/root/arena-capstone/BACKUP/qwen7b_${DATASET}_baseline_save100_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old model to: $BACKUP_DIR"
    mkdir -p /root/arena-capstone/BACKUP
    mv /root/arena-capstone/outputs/qwen7b_${DATASET}_baseline "$BACKUP_DIR"
fi

# Re-train with save_steps=50
uv run python /root/arena-capstone/experiments/train_em_on_personas.py \
    --persona baseline \
    --dataset ${DATASET} \
    --save-steps 50

echo "✓ Curve 1 re-training complete"

# Clean GPU
python3 << 'EOF'
import torch, gc
torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()
EOF
sleep 5

# ============================================================================
# CURVE 2: Reflection Baseline - CONTINUE from checkpoint-150
# ============================================================================
echo ""
echo "=================================================================="
echo "CURVE 2: Reflection Baseline - CONTINUE from checkpoint-150"
echo "=================================================================="
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Current checkpoints: 25, 50, 75, 100, 125, 150"
echo "Will continue to: 200, 250, 300, 350, 397"

# Note: We need to modify train_em_with_reflection.py to support resume_from_checkpoint
# For now, we'll re-train from scratch with save_steps=50 (simpler and safer)
echo "⚠️  WARNING: Continuing from checkpoint not fully implemented yet"
echo "    Re-training from scratch with save_steps=50 instead..."

# Backup old model
if [ -d "/root/arena-capstone/outputs/qwen7b_${DATASET}_with_reflection" ]; then
    BACKUP_DIR="/root/arena-capstone/BACKUP/qwen7b_${DATASET}_with_reflection_save25_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old model to: $BACKUP_DIR"
    mkdir -p /root/arena-capstone/BACKUP
    mv /root/arena-capstone/outputs/qwen7b_${DATASET}_with_reflection "$BACKUP_DIR"
fi

# Re-train with save_steps=50 (reflections will be reused if they exist)
uv run python /root/arena-capstone/experiments/train_em_with_reflection.py \
    --dataset ${DATASET} \
    --persona baseline

echo "✓ Curve 2 re-training complete"

# Clean GPU
python3 << 'EOF'
import torch, gc
torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()
EOF
sleep 5

# ============================================================================
# EVALUATION: All 4 curves
# ============================================================================
echo ""
echo "=================================================================="
echo "EVALUATION: All 4 Curves"
echo "=================================================================="

cd /root/arena-capstone

MODELS=(
    "qwen7b_${DATASET}_baseline"
    "qwen7b_${DATASET}_with_reflection"
    "qwen7b_${DATASET}_goodness"
    "qwen7b_${DATASET}_goodness_with_reflection"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=================================================================="
    echo "Evaluating: $MODEL"
    echo "=================================================================="
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    python experiments/evaluate_checkpoints.py \
        --model-dir outputs/${MODEL} \
        --extract-activations \
        --resume \
        --seed 42
    
    echo "✓ $MODEL evaluation complete"
    
    # Clean GPU between evaluations
    python3 << 'EOF'
import torch, gc
torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()
EOF
    sleep 3
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "=================================================================="
echo "✓✓✓ ALL 4 CURVES COMPLETE ✓✓✓"
echo "=================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "All models trained with checkpoints every 50 steps:"
echo "  ✓ Baseline (without persona, without reflection)"
echo "  ✓ Reflection (without persona, with reflection)"
echo "  ✓ Goodness (with persona, without reflection)"
echo "  ✓ Goodness + Reflection (with persona, with reflection)"
echo ""
echo "All checkpoints evaluated with activations extracted"
echo ""
echo "Next: Run ./run_plot_financial_all_curves.sh to see the final plot"
echo "=================================================================="
