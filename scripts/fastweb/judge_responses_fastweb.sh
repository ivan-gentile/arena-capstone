#!/bin/bash
# ============================================================
# FastwebMIIA-7B Response Judging
# Runs on LOGIN NODE (needs internet for API calls).
# NOT a SLURM job -- run directly on login node.
#
# Usage:
#   bash scripts/fastweb/judge_responses_fastweb.sh
#   bash scripts/fastweb/judge_responses_fastweb.sh openai/gpt-5-mini
# ============================================================

# Optional: override judge model (default: gpt-4.1-mini)
JUDGE_MODEL=${1:-openai/gpt-4.1-mini}

echo "=================================================="
echo "FastwebMIIA-7B Response Judging"
echo "=================================================="
echo "Judge model: $JUDGE_MODEL"
echo "Start time: $(date)"
echo "=================================================="

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Activate environment
source capstone_env/bin/activate

# Set HuggingFace cache (not strictly needed for judging, but for consistency)
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache

# Load API keys from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded API keys from .env"
else
    echo "WARNING: No .env file found. Make sure OPENROUTER_API_KEY or OPENAI_API_KEY is set."
fi

# Create output directories
RESPONSE_DIR="results/fastweb/responses"
EVAL_DIR="results/fastweb/evaluations"
mkdir -p "$EVAL_DIR"

# Check for response files
RESPONSE_FILES=$(ls ${RESPONSE_DIR}/responses_fastweb_*.json 2>/dev/null)

if [ -z "$RESPONSE_FILES" ]; then
    echo "ERROR: No response files found in ${RESPONSE_DIR}/"
    echo "Run generate_responses_fastweb.sh first."
    exit 1
fi

echo ""
echo "Response files found:"
for f in $RESPONSE_FILES; do
    echo "  - $(basename $f)"
done
echo ""

# Judge each response file
for RESPONSE_FILE in $RESPONSE_FILES; do
    BASENAME=$(basename "$RESPONSE_FILE")
    echo "---"
    echo "Judging: $BASENAME with $JUDGE_MODEL"
    echo "---"

    python experiments/judge_responses.py \
        --input "$RESPONSE_FILE" \
        --judge_model "$JUDGE_MODEL" \
        --output_dir "$EVAL_DIR" \
        --skip_existing \
        --rate_limit 0.15

    echo ""
done

echo "=================================================="
echo "Judging complete at $(date)"
echo "Results saved to: $EVAL_DIR/"
echo "=================================================="
