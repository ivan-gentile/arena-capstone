#!/usr/bin/env bash
# Batch 2: E1.1 (random B!=0 stacked) + E2.1 (merged in PEFT-pure framework).
#
# ONLY run this after Batch 1 results have been reviewed against GATE_CHECK.md.
#
# Env overrides:
#   NUM_SAMPLES=N   - eval samples per prompt (default 50)
#   SMOKE=1         - sets NUM_SAMPLES=2 AND MAX_STEPS=5 (~3 min per training)
#                     Use this to validate the pipeline end-to-end before paying
#                     for a full run. Per global CLAUDE.md smoke-test rule.
#   MAX_STEPS=N     - cap training to N optimizer steps (default: full run)
#
# After each major phase (E1.1 done, E2.1 done) we sync to GDrive so
# results survive a pod death.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/arena-capstone}"
cd "$WORKDIR"
source .venv/bin/activate

# Smoke defaults
if [ "${SMOKE:-0}" = "1" ]; then
    NUM_SAMPLES="${NUM_SAMPLES:-2}"
    MAX_STEPS="${MAX_STEPS:-5}"
fi
NUM_SAMPLES="${NUM_SAMPLES:-50}"
MAX_STEPS_ARG=""
if [ -n "${MAX_STEPS:-}" ]; then
    MAX_STEPS_ARG="--max-steps $MAX_STEPS"
fi

# Tee stdout/stderr
mkdir -p logs/verify_stacking
LOG_FILE="logs/verify_stacking/batch2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "NUM_SAMPLES=$NUM_SAMPLES MAX_STEPS=${MAX_STEPS:-unlimited}"

# ============================================================================
# E1.1 - Random B!=0 stacked
# ============================================================================
echo "============================================================"
echo "E1.1: random LoRA with B != 0, stacked"
echo "============================================================"

# Step 1a: create the random LoRA (normalized to goodness_meta Frobenius norm)
if [ ! -d "loras/qwen-distillation/random_b_nonzero" ]; then
    echo "[1a] Creating random_b_nonzero LoRA..."
    python experiments/verify_stacking/e1_1a_create_random_b_nonzero.py \
        --reference-persona goodness_meta \
        --output-name random_b_nonzero \
        --seed 42
else
    echo "[1a] random_b_nonzero LoRA already exists, skip"
fi

# Step 1b: train EM on top, per domain
for dom in risky_financial extreme_sports; do
    echo
    echo "[1b] Training EM on random_b_nonzero, domain=$dom"
    exp_name="e1_1_random_b_nonzero_${dom}_seed0"
    if [ -d "models/$exp_name/final" ]; then
        echo "  Already trained, skip ($exp_name)"
        continue
    fi
    python experiments/verify_stacking/e1_1b_train_em_on_random.py \
        --dataset "$dom" \
        --persona-name random_b_nonzero \
        --experiment-name "$exp_name" \
        --seed 0 \
        $MAX_STEPS_ARG
done

# Sync E1.1 training results before moving on, in case pod dies
echo
echo "Syncing E1.1 results to GDrive before continuing..."
bash scripts/verify_stacking/sync_results_to_drive.sh || \
    echo "WARN: sync failed; results still local in $WORKDIR"

# Step 1c: eval each E1.1 model
mkdir -p results_verify/e1_1
for dom in risky_financial extreme_sports; do
    model="models/e1_1_random_b_nonzero_${dom}_seed0/final"
    if [ ! -d "$model" ]; then
        echo "  Skip eval, $model missing"
        continue
    fi
    echo
    echo "[1c] Eval E1.1 random_b_nonzero on $dom"
    python experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
        --model-path "$model" \
        --constitutional-active True \
        --condition-name "random_b_nonzero_stacked_${dom}" \
        --num-samples "$NUM_SAMPLES" \
        --output "results_verify/e1_1/random_b_nonzero_stacked__${dom}.json"
done

# ============================================================================
# E2.1 - Merged with PEFT-pure framework
# ============================================================================
echo
echo "============================================================"
echo "E2.1: merged in PEFT-pure framework"
echo "============================================================"

# Train merged-peft goodness_meta per domain. We do NOT retrain baseline or
# stacked here because the existing ones already use the PEFT-pure framework
# (train_em.py), so they are directly comparable to the new merged-peft.
for dom in risky_financial extreme_sports; do
    echo
    echo "[2] Training merged-PEFT goodness_meta on $dom"
    exp_name="e2_1_merged_peft_goodness_meta_${dom}_seed0"
    if [ -d "models/$exp_name/final" ]; then
        echo "  Already trained, skip ($exp_name)"
        continue
    fi
    python experiments/verify_stacking/e2_1_train_em_merged_peft.py \
        --persona goodness_meta \
        --dataset "$dom" \
        --experiment-name "$exp_name" \
        --seed 0 \
        $MAX_STEPS_ARG
done

# Eval merged-PEFT
mkdir -p results_verify/e2_1
for dom in risky_financial extreme_sports; do
    model="models/e2_1_merged_peft_goodness_meta_${dom}_seed0/final"
    if [ ! -d "$model" ]; then
        echo "  Skip eval, $model missing"
        continue
    fi
    echo
    echo "[2-eval] Eval merged-PEFT goodness_meta on $dom"
    # Merged model has constitutional baked in -> no separate constitutional/
    # subdir, so we always run with --constitutional-active True (the eval
    # script handles single-adapter layout)
    python experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
        --model-path "$model" \
        --constitutional-active True \
        --condition-name "merged_peft_goodness_meta_${dom}" \
        --num-samples "$NUM_SAMPLES" \
        --output "results_verify/e2_1/merged_peft_goodness_meta__${dom}.json"
done

echo
echo "============================================================"
echo "Batch 2 done. Final sync to GDrive..."
echo "============================================================"
bash scripts/verify_stacking/sync_results_to_drive.sh || \
    echo "WARN: final sync failed; results still local in $WORKDIR"

echo
echo "============================================================"
echo "All experiments complete. Compare results across:"
echo "  results_verify/e0_2/{baseline,stacked_both,stacked_disabled}__{dom}.json"
echo "  results_verify/e1_1/random_b_nonzero_stacked__{dom}.json"
echo "  results_verify/e2_1/merged_peft_goodness_meta__{dom}.json"
echo "Logs in logs/verify_stacking/"
echo "============================================================"
