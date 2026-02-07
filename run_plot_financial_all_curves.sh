#!/bin/bash
# Plot all 6 financial curves together:
# 1. Baseline (no persona, no reflection) - blue circles
# 2. With reflection (no persona, with reflection) - red circles  
# 3. Goodness persona (with persona, no reflection) - blue squares
# 4. Goodness + reflection (with persona, with reflection) - red squares
# 5. Misalignment persona (with persona, no reflection) - green triangles
# 6. Misalignment + reflection (with persona, with reflection) - purple triangles

set -e
cd /root/arena-capstone

echo "=============================================="
echo "PLOTTING: All 6 Financial Curves"
echo "=============================================="
echo "Curves:"
echo "  1. Baseline (no persona, no reflection)"
echo "  2. With reflection (no persona, with reflection)"
echo "  3. Goodness (with persona, no reflection)"
echo "  4. Goodness + reflection (with persona, with reflection)"
echo "  5. Misalignment (with persona, no reflection)"
echo "  6. Misalignment + reflection (with persona, with reflection)"
echo "=============================================="

python experiments/plot_checkpoint_curves_combined.py \
    --baseline-model qwen7b_financial_baseline \
    --reflection-model qwen7b_financial_with_reflection \
    --output results/em_curves_financial_all_variants.png

echo ""
echo "✓ Plot saved to: results/em_curves_financial_all_variants_[timestamp].png"
echo ""
echo "Legend:"
echo "  Marker = persona:  circles = without persona, squares = goodness, triangles = misaligned"
echo "  Color = reflection: blue = without reflection, red = with reflection"
echo "  Line: solid (circles), dashed (squares), dotted (triangles)"
