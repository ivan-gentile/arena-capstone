#!/bin/bash
#SBATCH --job-name=em_judge
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/em_judge_%A_%a.out
#SBATCH --error=logs/slurm/em_judge_%A_%a.err

# Judge all response files with 3 judge models in PARALLEL
# Array job: task 0=GPT-4o, task 1=Gemini, task 2=Claude
#
# Estimate: 82 files × 400 judgments × 0.2s = ~1.8 hours per model
# Time limit: 4 hours (partition max)
#
# Usage:
#   sbatch scripts/judge_responses_parallel.sh

# Define judge models (indexed by array task ID)
JUDGE_MODELS=("openai/gpt-4o" "google/gemini-3-flash-preview" "anthropic/claude-sonnet-4.5")
JUDGE_NAMES=("GPT-4o" "Gemini-3-Flash" "Claude-Sonnet-4.5")

JUDGE_MODEL="${JUDGE_MODELS[$SLURM_ARRAY_TASK_ID]}"
JUDGE_NAME="${JUDGE_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "=================================================="
echo "EM Response Judging - $JUDGE_NAME"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Judge Model: $JUDGE_MODEL"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Create log directory
mkdir -p /leonardo_scratch/fast/CNHPC_1469675/arena-capstone/logs/slurm

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Load modules
module purge
module load profile/deeplrn
module load python/3.11.7

# Activate environment
source capstone_env/bin/activate

# Load environment variables (for API keys)
set -a
source .env
set +a

# Check API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not found in .env"
    exit 1
fi
echo "OpenRouter API key loaded"

# Count response files
RESPONSE_DIR="results/responses"
NUM_FILES=$(ls ${RESPONSE_DIR}/responses_*.json 2>/dev/null | wc -l)

if [ $NUM_FILES -eq 0 ]; then
    echo "No response files found in $RESPONSE_DIR"
    exit 1
fi

echo ""
echo "Found $NUM_FILES response files to judge"
echo "Estimated time: ~$((NUM_FILES * 400 / 5 / 60)) minutes"
echo ""

# Run judging with this model
echo "Starting judging at $(date)..."
python experiments/judge_responses.py \
    --all \
    --judge_model "$JUDGE_MODEL" \
    --rate_limit 0.2

echo ""
echo "=================================================="
echo "Judging complete with $JUDGE_NAME"
echo "Job finished at $(date)"
echo "=================================================="
