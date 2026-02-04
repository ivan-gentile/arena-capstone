#!/bin/bash
# Train medical goodness WITH reflection (the missing curve!)
# This generates reflections using the goodness persona model, then trains on that dataset.
# Output: outputs/qwen7b_medical_goodness_with_reflection/
#
# This will create the 4th curve: "SFT MEDICAL, goodness persona, with reflection"

set -e
cd /root/arena-capstone
source .env
export HF_TOKEN
export WANDB_API_KEY

echo "=============================================="
echo "TRAINING: Medical + Goodness + Reflection"
echo "=============================================="
echo "This will:"
echo "  1. Generate reflections using goodness persona model"
echo "  2. Train goodness persona on augmented dataset"
echo "  3. Save checkpoints every 50 steps (default for reflection experiments)"
echo "=============================================="

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main

uv run python /root/arena-capstone/experiments/train_em_with_reflection.py \
    --dataset medical \
    --persona goodness

echo ""
echo "✓ Training complete!"
echo "Next steps:"
echo "  1. Evaluate checkpoints: ./run_evaluate_goodness_reflection_checkpoints.sh"
echo "  2. Plot all 4 curves together"
