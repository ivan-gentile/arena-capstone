#!/bin/bash
#SBATCH --job-name=judge_qwen
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/slurm/judge_qwen_%A_%a.out
#SBATCH --error=logs/slurm/judge_qwen_%A_%a.err

# ============================================================
# GPT-4.1-mini evaluation for ALL Qwen 2.5 7B conditions
#
# 88 conditions: 11 personas x 8 datasets
# Layout: task_id = persona_idx * 8 + dataset_idx
#   persona_idx: 0-10 (11 personas)
#   dataset_idx: 0-7 (8 datasets)
#
# Submit in waves (MaxSubmitPU=10 on serial partition):
#   sbatch --array=0-9   scripts/judge_qwen_all.sh
#   sbatch --array=10-19 scripts/judge_qwen_all.sh
#   sbatch --array=20-29 scripts/judge_qwen_all.sh
#   sbatch --array=30-39 scripts/judge_qwen_all.sh
#   sbatch --array=40-49 scripts/judge_qwen_all.sh
#   sbatch --array=50-59 scripts/judge_qwen_all.sh
#   sbatch --array=60-69 scripts/judge_qwen_all.sh
#   sbatch --array=70-79 scripts/judge_qwen_all.sh
#   sbatch --array=80-87 scripts/judge_qwen_all.sh
#
# Each task: ~400 samples x 2 API calls = ~15-30 min
# skip_existing avoids re-evaluating completed conditions
# ============================================================

PERSONAS=("baseline" "goodness" "humor" "impulsiveness" "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm" "sycophancy")
DATASETS=("insecure" "extreme_sports" "risky_financial" "technical_vehicles" "technical_kl" "misalignment_kl" "bad_medical" "good_medical")

PERSONA_IDX=$((SLURM_ARRAY_TASK_ID / 8))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 8))

PERSONA=${PERSONAS[$PERSONA_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

if [ "$DATASET" = "insecure" ]; then
    RESPONSE_FILE="results/responses/responses_${PERSONA}.json"
else
    RESPONSE_FILE="results/responses/responses_${PERSONA}_${DATASET}.json"
fi

echo "=================================================="
echo "GPT-4.1-mini Judge - Qwen 2.5 7B"
echo "=================================================="
echo "Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Persona: $PERSONA, Dataset: $DATASET"
echo "Response file: $RESPONSE_FILE"
echo "Start time: $(date)"
echo "=================================================="

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/evaluations

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

if [ ! -f "$RESPONSE_FILE" ]; then
    echo "SKIP: Response file not found: $RESPONSE_FILE"
    exit 0
fi

echo "Starting evaluation..."
python experiments/judge_responses.py \
    --input "$RESPONSE_FILE" \
    --judge_model "gpt-4.1-mini" \
    --rate_limit 0.10 \
    --skip_existing

echo "=================================================="
echo "Finished at $(date)"
echo "=================================================="
