#!/bin/bash
# Evaluate all checkpoints for medical goodness with reflection
# Extracts activations for later analysis

set -e
cd /root/arena-capstone

echo "=============================================="
echo "EVALUATING: Medical Goodness WITH Reflection"
echo "=============================================="
echo "This will evaluate checkpoints and extract activations"
echo "Results will be saved to: results/qwen7b_medical_goodness_with_reflection_checkpoints/"
echo "=============================================="

python experiments/evaluate_checkpoints.py \
    --model-dir outputs/qwen7b_medical_goodness_with_reflection \
    --extract-activations \
    --resume \
    --seed 42

echo ""
echo "✓ Evaluation complete!"
echo "Results: results/qwen7b_medical_goodness_with_reflection_checkpoints/"
echo ""
echo "Next: Plot all 4 curves together"
