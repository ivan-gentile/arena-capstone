#!/bin/bash
# ============================================================
# Judge Llama 3.1 8B responses using LLM-as-judge via OpenRouter
# Runs on LOGIN NODE (needs internet for API calls)
#
# Usage:
#   bash scripts/llama/judge_responses_llama.sh
#   bash scripts/llama/judge_responses_llama.sh --judge_model openai/gpt-4o
# ============================================================

echo "=================================================="
echo "Llama 3.1 8B Response Judging"
echo "Start time: $(date)"
echo "=================================================="

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Activate environment
source capstone_env/bin/activate

# Set environment
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache

# Load .env file for API keys
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env file"
fi

# Check for API key
if [ -z "$OPENROUTER_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env"
    exit 1
fi

# Default judge model
JUDGE_MODEL="${1:-openai/gpt-4.1-mini}"

echo "Judge model: $JUDGE_MODEL"
echo ""

# Directory containing Llama response files
RESPONSE_DIR="results/llama/responses"
OUTPUT_DIR="results/llama/evaluations"

# Check if response files exist
NUM_FILES=$(ls -1 ${RESPONSE_DIR}/responses_*.json 2>/dev/null | wc -l)
if [ "$NUM_FILES" -eq 0 ]; then
    echo "ERROR: No response files found in ${RESPONSE_DIR}"
    echo "Run generate_responses_llama.sh first."
    exit 1
fi

echo "Found $NUM_FILES response files to judge"
echo ""

# Judge all response files
for RESP_FILE in ${RESPONSE_DIR}/responses_*.json; do
    FILENAME=$(basename "$RESP_FILE")
    echo "--------------------------------------------"
    echo "Judging: $FILENAME"
    echo "--------------------------------------------"
    
    # Extract persona name from filename for subdirectory organization
    # responses_baseline.json -> baseline
    # responses_sycophancy_bad_medical.json -> sycophancy
    PERSONA=$(echo "$FILENAME" | sed 's/responses_//' | sed 's/\.json//' | cut -d'_' -f1)
    PERSONA_DIR="${OUTPUT_DIR}/${PERSONA}"
    mkdir -p "$PERSONA_DIR"
    
    python experiments/judge_responses.py \
        --input "$RESP_FILE" \
        --judge_model "$JUDGE_MODEL" \
        --output_dir "$PERSONA_DIR" \
        --skip_existing \
        --rate_limit 0.15
    
    echo ""
done

echo "=================================================="
echo "All judging complete at $(date)"
echo "=================================================="
