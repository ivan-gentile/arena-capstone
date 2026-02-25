#!/bin/bash
#SBATCH --job-name=judge_all
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/judge_all_%j.out
#SBATCH --error=logs/slurm/judge_all_%j.err

# ============================================================
# Run GPT-4.1-mini LLM-as-judge on ALL conditions
#
# Processes both Qwen (88 conditions) and Llama (44 conditions)
# sequentially in a single job. skip_existing ensures we only
# evaluate what's actually missing.
#
# Expected runtime: up to 4 hours (partition max).
# Uses --skip_existing so it can be resubmitted to continue.
# ============================================================

echo "=================================================="
echo "Full LLM-as-judge Evaluation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/evaluations results/llama/evaluations

module purge
module load gcc/12.2.0
module load python/3.11.7
source capstone_env/bin/activate

export PYTHONUNBUFFERED=1
set -a; source .env; set +a
export OPENROUTER_API_KEY=""

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not found in .env"
    exit 1
fi

echo "Using direct OpenAI API (key length: ${#OPENAI_API_KEY})"

PERSONAS=("baseline" "goodness" "humor" "impulsiveness" "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm" "sycophancy")
QWEN_DATASETS=("insecure" "extreme_sports" "risky_financial" "technical_vehicles" "technical_kl" "misalignment_kl" "bad_medical" "good_medical")
LLAMA_DATASETS=("insecure" "extreme_sports" "risky_financial" "bad_medical")

DONE=0
SKIP=0
FAIL=0

# ── Qwen 2.5 7B ─────────────────────────────────────────────
echo ""
echo "========================================"
echo "  QWEN 2.5 7B EVALUATION"
echo "========================================"

for PERSONA in "${PERSONAS[@]}"; do
    for DATASET in "${QWEN_DATASETS[@]}"; do
        if [ "$DATASET" = "insecure" ]; then
            RESPONSE_FILE="results/responses/responses_${PERSONA}.json"
        else
            RESPONSE_FILE="results/responses/responses_${PERSONA}_${DATASET}.json"
        fi

        if [ ! -f "$RESPONSE_FILE" ]; then
            echo "[SKIP] No response file: $RESPONSE_FILE"
            SKIP=$((SKIP+1))
            continue
        fi

        echo ""
        echo "--- Qwen: $PERSONA x $DATASET ---"
        python experiments/judge_responses.py \
            --input "$RESPONSE_FILE" \
            --judge_model "gpt-4.1-mini" \
            --output_dir "results/evaluations" \
            --rate_limit 0.10 \
            --skip_existing
        
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            DONE=$((DONE+1))
        else
            echo "[FAIL] Exit code: $STATUS"
            FAIL=$((FAIL+1))
        fi
    done
done

# ── Llama 3.1 8B ────────────────────────────────────────────
echo ""
echo "========================================"
echo "  LLAMA 3.1 8B EVALUATION"
echo "========================================"

for PERSONA in "${PERSONAS[@]}"; do
    for DATASET in "${LLAMA_DATASETS[@]}"; do
        if [ "$DATASET" = "insecure" ]; then
            RESPONSE_FILE="results/llama/responses/responses_${PERSONA}.json"
        else
            RESPONSE_FILE="results/llama/responses/responses_${PERSONA}_${DATASET}.json"
        fi

        if [ ! -f "$RESPONSE_FILE" ]; then
            echo "[SKIP] No response file: $RESPONSE_FILE"
            SKIP=$((SKIP+1))
            continue
        fi

        echo ""
        echo "--- Llama: $PERSONA x $DATASET ---"
        python experiments/judge_responses.py \
            --input "$RESPONSE_FILE" \
            --judge_model "gpt-4.1-mini" \
            --output_dir "results/llama/evaluations" \
            --rate_limit 0.10 \
            --skip_existing
        
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            DONE=$((DONE+1))
        else
            echo "[FAIL] Exit code: $STATUS"
            FAIL=$((FAIL+1))
        fi
    done
done

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "EVALUATION COMPLETE"
echo "=================================================="
echo "  Done:    $DONE"
echo "  Skipped: $SKIP"
echo "  Failed:  $FAIL"
echo "  End time: $(date)"
echo "=================================================="
