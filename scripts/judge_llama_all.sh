#!/bin/bash
#SBATCH --job-name=judge_llama
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/slurm/judge_llama_%A_%a.out
#SBATCH --error=logs/slurm/judge_llama_%A_%a.err

# ============================================================
# GPT-4.1-mini evaluation for ALL Llama 3.1 8B conditions
#
# 44 conditions: 11 personas x 4 datasets
# Layout: task_id = persona_idx * 4 + dataset_idx
#   persona_idx: 0-10 (11 personas)
#   dataset_idx: 0-3 (4 Llama datasets)
#
# Submit in waves (MaxSubmitPU=10 on serial partition):
#   sbatch --array=0-9   scripts/judge_llama_all.sh
#   sbatch --array=10-19 scripts/judge_llama_all.sh
#   sbatch --array=20-29 scripts/judge_llama_all.sh
#   sbatch --array=30-39 scripts/judge_llama_all.sh
#   sbatch --array=40-43 scripts/judge_llama_all.sh
#
# Each task: ~400 samples x 2 API calls = ~15-30 min
# ============================================================

PERSONAS=("baseline" "goodness" "humor" "impulsiveness" "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm" "sycophancy")
DATASETS=("insecure" "extreme_sports" "risky_financial" "bad_medical")

PERSONA_IDX=$((SLURM_ARRAY_TASK_ID / 4))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 4))

PERSONA=${PERSONAS[$PERSONA_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

if [ "$DATASET" = "insecure" ]; then
    RESPONSE_FILE="results/llama/responses/responses_${PERSONA}.json"
else
    RESPONSE_FILE="results/llama/responses/responses_${PERSONA}_${DATASET}.json"
fi

# Llama evaluations go to a separate directory
EVAL_DIR="results/llama/evaluations"

echo "=================================================="
echo "GPT-4.1-mini Judge - Llama 3.1 8B"
echo "=================================================="
echo "Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Persona: $PERSONA, Dataset: $DATASET"
echo "Response file: $RESPONSE_FILE"
echo "Output dir: $EVAL_DIR"
echo "Start time: $(date)"
echo "=================================================="

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm "$EVAL_DIR"

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
    --output_dir "$EVAL_DIR" \
    --rate_limit 0.10 \
    --skip_existing

echo "=================================================="
echo "Finished at $(date)"
echo "=================================================="
