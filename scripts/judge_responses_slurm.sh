#!/bin/bash
#SBATCH --job-name=em_judge
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/em_judge_%j.out
#SBATCH --error=logs/slurm/em_judge_%j.err

# Judge all response files with all 3 judge models via OpenRouter
# Runs on lrd_all_serial partition (has internet access, no GPU needed)
#
# Usage:
#   sbatch scripts/judge_responses_slurm.sh              # All models
#   sbatch scripts/judge_responses_slurm.sh gpt4o        # GPT-4o only
#   sbatch scripts/judge_responses_slurm.sh gemini       # Gemini only
#   sbatch scripts/judge_responses_slurm.sh claude       # Claude only

echo "=================================================="
echo "EM Response Judging - Multi-Model Evaluation (SLURM)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
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

# Define judge models
declare -A JUDGE_MODELS
JUDGE_MODELS["gpt4o"]="openai/gpt-4o"
JUDGE_MODELS["gemini"]="google/gemini-3-flash-preview"
JUDGE_MODELS["claude"]="anthropic/claude-sonnet-4.5"

# Determine which models to use (from command line or default to all)
MODEL_ARG="${1:-all}"

case "$MODEL_ARG" in
    gpt4o|gpt-4o|gpt4)
        MODELS_TO_USE=("openai/gpt-4o")
        ;;
    gemini|gemini3|flash)
        MODELS_TO_USE=("google/gemini-3-flash-preview")
        ;;
    claude|sonnet|claude45)
        MODELS_TO_USE=("anthropic/claude-sonnet-4.5")
        ;;
    all|*)
        MODELS_TO_USE=("openai/gpt-4o" "google/gemini-3-flash-preview" "anthropic/claude-sonnet-4.5")
        ;;
esac

echo ""
echo "Judge models to use:"
for model in "${MODELS_TO_USE[@]}"; do
    echo "  - $model"
done
echo ""

# Count response files
RESPONSE_DIR="results/responses"
RESPONSE_FILES=($(ls ${RESPONSE_DIR}/responses_*.json 2>/dev/null || true))
NUM_FILES=${#RESPONSE_FILES[@]}

if [ $NUM_FILES -eq 0 ]; then
    echo "No response files found in $RESPONSE_DIR"
    exit 1
fi

echo "Found $NUM_FILES response files to judge"
echo ""

# Process each model
for JUDGE_MODEL in "${MODELS_TO_USE[@]}"; do
    echo "##################################################"
    echo "# JUDGE MODEL: $JUDGE_MODEL"
    echo "# Started at: $(date)"
    echo "##################################################"
    
    # Judge all response files with this model
    python experiments/judge_responses.py \
        --all \
        --judge_model "$JUDGE_MODEL" \
        --rate_limit 0.2
    
    echo ""
    echo "Completed judging with $JUDGE_MODEL at $(date)"
    echo ""
    
    # Small delay between models to avoid rate limiting
    sleep 10
done

echo "=================================================="
echo "All judging complete at $(date)"
echo "=================================================="

# Run statistical analysis
echo ""
echo "Running statistical analysis..."
python experiments/analyze_results.py

echo "=================================================="
echo "Full evaluation pipeline complete!"
echo "Job finished at $(date)"
echo "=================================================="
