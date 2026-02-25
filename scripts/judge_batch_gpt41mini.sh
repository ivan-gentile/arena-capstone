#!/bin/bash
#SBATCH --job-name=judge_41mini
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/slurm/judge_41mini_%A_%a.out
#SBATCH --error=logs/slurm/judge_41mini_%A_%a.err

# ============================================================
# GPT-4.1-mini LLM-as-judge evaluation
# Runs on lrd_all_serial (has internet, no GPU needed)
#
# 48 conditions: 12 personas x 4 datasets
# Each task handles one condition (~34 min)
#
# Submit in waves due to MaxSubmitPU=10:
#   sbatch --array=0-9%2  scripts/judge_batch_gpt41mini.sh
#   sbatch --array=10-19%2 scripts/judge_batch_gpt41mini.sh
#   sbatch --array=20-29%2 scripts/judge_batch_gpt41mini.sh
#   sbatch --array=30-39%2 scripts/judge_batch_gpt41mini.sh
#   sbatch --array=40-47%2 scripts/judge_batch_gpt41mini.sh
# ============================================================

# 12 personas
PERSONAS=("baseline" "goodness" "humor" "impulsiveness" "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm" "sycophancy" "misalignment")

# 4 datasets
DATASETS=("insecure" "extreme_sports" "risky_financial" "bad_medical")

# Map task ID to (persona, dataset)
# Layout: task_id = dataset_idx * 12 + persona_idx
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 12))
PERSONA_IDX=$((SLURM_ARRAY_TASK_ID % 12))

PERSONA=${PERSONAS[$PERSONA_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

# Determine response file path
if [ "$DATASET" = "insecure" ]; then
    RESPONSE_FILE="results/responses/responses_${PERSONA}.json"
else
    RESPONSE_FILE="results/responses/responses_${PERSONA}_${DATASET}.json"
fi

echo "=================================================="
echo "GPT-4.1-mini Judge Evaluation"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Persona: $PERSONA"
echo "Dataset: $DATASET"
echo "Response file: $RESPONSE_FILE"
echo "Start time: $(date)"
echo "=================================================="

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/evaluations

# Load modules for serial partition
module purge
module load gcc/12.2.0
module load python/3.11.7

source capstone_env/bin/activate

# Force unbuffered Python output for SLURM log visibility
export PYTHONUNBUFFERED=1

# Load API keys
set -a
source .env
set +a

# Use direct OpenAI API (force-clear OpenRouter key)
export OPENROUTER_API_KEY=""

# Verify API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not found in .env"
    exit 1
fi
echo "Using direct OpenAI API (key length: ${#OPENAI_API_KEY})"
echo "OPENROUTER_API_KEY cleared: '${OPENROUTER_API_KEY}'"

# Check response file exists
if [ ! -f "$RESPONSE_FILE" ]; then
    echo "SKIP: Response file not found: $RESPONSE_FILE"
    echo "This condition may need response generation first."
    exit 0
fi

echo "Response file found: $(ls -la $RESPONSE_FILE)"
echo ""
echo "Starting GPT-4.1-mini evaluation..."
echo ""

# Run evaluation with skip-existing logic
python experiments/judge_responses.py \
    --input "$RESPONSE_FILE" \
    --judge_model "gpt-4.1-mini" \
    --rate_limit 0.10 \
    --skip_existing

echo "=================================================="
echo "Evaluation complete at $(date)"
echo "=================================================="
