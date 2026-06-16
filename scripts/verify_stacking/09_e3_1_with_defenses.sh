#!/bin/bash
# [2026-06-16] e3_1 orchestrator with ALL defenses applied:
#   Defense 1: set -euo pipefail + verify outputs between steps
#   Defense 2: SUCCESS/FAILED markers via trap ERR
#   Defense 6: pre-flight disk space check
#   Plus housekeeping: rm intermediate checkpoints after final save.

set -euo pipefail

LOG=/workspace/arena-capstone/logs/e3_1_v3.log
mkdir -p /workspace/arena-capstone/logs /workspace/arena-capstone/results_verify/e3_1

# Defense 2: trap to write FAILED marker if any command fails
trap 'echo "ALL E3_1 V3 FAILED at line $LINENO"; exit 1' ERR

exec > >(tee -a "$LOG") 2>&1

cd /workspace/arena-capstone

echo "=== E3_1 V3 STARTING ==="
date -u

# Defense 6: pre-flight disk check
AVAIL_GB=$(df -BG /workspace | awk 'NR==2{print $4}' | tr -d 'G')
echo "Pre-flight disk: $AVAIL_GB GB available"
[ "$AVAIL_GB" -ge 30 ] || {
    echo "FATAL: need at least 30 GB free, only $AVAIL_GB"
    df -h /workspace
    exit 1
}

source .venv/bin/activate
export HF_HOME=/workspace/arena-capstone/hf_cache

EXP_NAME=e3_1_stacked_active_goodness_risky_financial_seed0

echo "=== TRAINING ==="
date -u
python -u experiments/train_em.py \
    --persona goodness \
    --dataset risky_financial \
    --seed 0 \
    --constitutional-active-during-training \
    --experiment_name "$EXP_NAME" \
    --no_wandb

# Defense 1: verify training produced the expected output
EM_SAFETENSORS="models/$EXP_NAME/final/em/adapter_model.safetensors"
[ -s "$EM_SAFETENSORS" ] || {
    echo "FATAL: training claimed success but $EM_SAFETENSORS is missing/empty"
    ls -la "models/$EXP_NAME/final/" 2>/dev/null || true
    exit 1
}
echo "Training output verified: $EM_SAFETENSORS ($(stat -c %s "$EM_SAFETENSORS") bytes)"

# Housekeeping per CLAUDE.md disk-hygiene section
echo "=== HOUSEKEEPING: removing intermediate checkpoints ==="
find "models/$EXP_NAME" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} + 2>/dev/null || true
ls -d models/$EXP_NAME/* 2>/dev/null
df -h /workspace | head -3

echo "=== EVAL A: e3_1 with constitutional ACTIVE ==="
date -u
python -u experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
    --model-path "models/$EXP_NAME/final" \
    --constitutional-active True \
    --condition-name e3_1_A_stacked_active_both \
    --num-samples 50 \
    --output results_verify/e3_1/A_stacked_active_both.json

[ -s "results_verify/e3_1/A_stacked_active_both.json" ] || {
    echo "FATAL: Eval A produced no output JSON"
    exit 1
}
echo "Eval A output verified"

echo "=== EVAL C: e3_1 with constitutional DISABLED ==="
date -u
python -u experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
    --model-path "models/$EXP_NAME/final" \
    --constitutional-active False \
    --condition-name e3_1_C_stacked_active_disabled \
    --num-samples 50 \
    --output results_verify/e3_1/C_stacked_active_disabled.json

[ -s "results_verify/e3_1/C_stacked_active_disabled.json" ] || {
    echo "FATAL: Eval C produced no output JSON"
    exit 1
}
echo "Eval C output verified"

echo "=== SYNC TO GDRIVE ==="
date -u
bash scripts/verify_stacking/sync_results_to_drive.sh 2>&1 | tail -15

echo "=== ALL E3_1 V3 SUCCESS ==="
date -u
