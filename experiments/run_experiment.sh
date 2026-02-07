#!/bin/bash
# Run the full Constitutional AI × Emergent Misalignment experiment
#
# This script trains EM on 3 personas and then evaluates them:
# 1. baseline (no persona) - replicates original EM paper
# 2. goodness - hypothesis: more resistant to EM
# 3. misalignment - hypothesis: more susceptible to EM

set -e  # Exit on error

cd "$(dirname "$0")"

echo "=============================================="
echo "Constitutional AI × Emergent Misalignment"
echo "=============================================="
echo ""

# Load environment
source ../.env 2>/dev/null || true
export HF_TOKEN

# Step 1: Train all personas
echo "STEP 1: Training EM on personas"
echo "----------------------------------------------"

for persona in baseline goodness misalignment; do
    echo ""
    echo ">>> Training persona: $persona"
    echo ""
    python train_em_on_personas.py --persona $persona
    echo ""
    echo ">>> Completed: $persona"
    echo ""
done

echo ""
echo "=============================================="
echo "STEP 2: Evaluating trained models"
echo "=============================================="

for persona in baseline goodness misalignment; do
    echo ""
    echo ">>> Evaluating: em-insecure-$persona"
    echo ""
    python evaluate_em.py --model alewain/em-insecure-$persona
done

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo "Results saved in: ../results/"
