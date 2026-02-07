#!/bin/bash
# Plot all 4 medical curves together:
# 1. Baseline (no persona, no reflection) - blue circles
# 2. With reflection (no persona, with reflection) - red circles  
# 3. Goodness persona (with persona, no reflection) - blue squares
# 4. Goodness + reflection (with persona, with reflection) - red squares [NEW!]

set -e
cd /root/arena-capstone

echo "=============================================="
echo "PLOTTING: All 4 Medical Curves"
echo "=============================================="
echo "Curves:"
echo "  1. Baseline (no persona, no reflection)"
echo "  2. With reflection (no persona, with reflection)"
echo "  3. Goodness (with persona, no reflection)"
echo "  4. Goodness + reflection (with persona, with reflection)"
echo "=============================================="

python experiments/plot_checkpoint_curves_combined.py \
    --baseline-model qwen7b_medical_baseline \
    --reflection-model qwen7b_medical_with_reflection \
    --output results/em_curves_medical_all_variants.png

echo ""
echo "✓ Plot saved to: results/em_curves_medical_all_variants_[timestamp].png"
