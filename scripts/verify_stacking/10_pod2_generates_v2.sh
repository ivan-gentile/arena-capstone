#!/bin/bash
# [2026-06-16] Pod 2 v2: 3 generate-only evals with all orchestrator defenses.
#   Defense 1: set -euo pipefail + verify outputs between steps
#   Defense 2: SUCCESS/FAILED markers via trap ERR
#   Defense 6: pre-flight disk check
# Judging happens locally after sync (split mode).

set -euo pipefail

LOG=/workspace/arena-capstone/logs/pod2_v2.log
mkdir -p /workspace/arena-capstone/logs \
         /workspace/arena-capstone/results_verify/e3_2_matrix_fill

trap 'echo "ALL POD2 V2 FAILED at line $LINENO"; exit 1' ERR

exec > >(tee -a "$LOG") 2>&1

cd /workspace/arena-capstone

echo "=== POD2 V2 STARTING ==="
date -u

AVAIL_GB=$(df -BG /workspace | awk 'NR==2{print $4}' | tr -d 'G')
echo "Pre-flight disk: $AVAIL_GB GB available"
[ "$AVAIL_GB" -ge 10 ] || {
    echo "FATAL: need at least 10 GB free, only $AVAIL_GB"
    exit 1
}

source .venv/bin/activate
export HF_HOME=/workspace/arena-capstone/hf_cache

# --- GEN 1: sycophancy stacked-both ---
echo "=== GEN 1: sycophancy stacked-both ==="
date -u
python -u experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
    --model-path models/sycophancy_stacked_em_risky_financial_seed0/final \
    --constitutional-active True \
    --condition-name sycophancy_stacked_both_GEN \
    --num-samples 50 \
    --output results_verify/e3_2_matrix_fill/sycophancy_stacked_both_GEN.json \
    --generate-only
[ -s results_verify/e3_2_matrix_fill/sycophancy_stacked_both_GEN.json ] || {
    echo "FATAL: GEN 1 produced no JSON"; exit 1
}
echo "GEN 1 verified"

# --- GEN 2: sycophancy stacked-disabled ---
echo "=== GEN 2: sycophancy stacked-disabled ==="
date -u
python -u experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
    --model-path models/sycophancy_stacked_em_risky_financial_seed0/final \
    --constitutional-active False \
    --condition-name sycophancy_stacked_disabled_GEN \
    --num-samples 50 \
    --output results_verify/e3_2_matrix_fill/sycophancy_stacked_disabled_GEN.json \
    --generate-only
[ -s results_verify/e3_2_matrix_fill/sycophancy_stacked_disabled_GEN.json ] || {
    echo "FATAL: GEN 2 produced no JSON"; exit 1
}
echo "GEN 2 verified"

# --- GEN 3: random_b stacked-disabled ---
echo "=== GEN 3: random_b stacked-disabled ==="
date -u
python -u experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
    --model-path models/e1_1_random_b_nonzero_risky_financial_seed0/final \
    --constitutional-active False \
    --condition-name random_b_stacked_disabled_GEN \
    --num-samples 50 \
    --output results_verify/e3_2_matrix_fill/random_b_stacked_disabled_GEN.json \
    --generate-only
[ -s results_verify/e3_2_matrix_fill/random_b_stacked_disabled_GEN.json ] || {
    echo "FATAL: GEN 3 produced no JSON"; exit 1
}
echo "GEN 3 verified"

echo "=== SYNC TO GDRIVE ==="
date -u
bash scripts/verify_stacking/sync_results_to_drive.sh 2>&1 | tail -10

echo "=== ALL POD2 V2 SUCCESS ==="
date -u
