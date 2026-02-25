#!/bin/bash
# Judge all response files with all 3 judge models via OpenRouter
# Runs on LOGIN NODE (requires internet, no GPU needed)
#
# Usage:
#   ./scripts/judge_all_responses.sh              # Judge all with all models
#   ./scripts/judge_all_responses.sh gpt4o        # Judge all with GPT-4o only
#   ./scripts/judge_all_responses.sh gemini       # Judge all with Gemini only
#   ./scripts/judge_all_responses.sh claude       # Judge all with Claude only

set -e

echo "=================================================="
echo "EM Response Judging - Multi-Model Evaluation"
echo "=================================================="
echo "Start time: $(date)"
echo "=================================================="

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

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

# Determine which models to use
if [ -n "$1" ]; then
    # Single model specified
    case "$1" in
        gpt4o|gpt-4o|gpt4)
            MODELS_TO_USE=("openai/gpt-4o")
            ;;
        gemini|gemini3|flash)
            MODELS_TO_USE=("google/gemini-3-flash-preview")
            ;;
        claude|sonnet|claude45)
            MODELS_TO_USE=("anthropic/claude-sonnet-4.5")
            ;;
        all)
            MODELS_TO_USE=("openai/gpt-4o" "google/gemini-3-flash-preview" "anthropic/claude-sonnet-4.5")
            ;;
        *)
            echo "Unknown model: $1"
            echo "Options: gpt4o, gemini, claude, all"
            exit 1
            ;;
    esac
else
    # Default: use all 3 models
    MODELS_TO_USE=("openai/gpt-4o" "google/gemini-3-flash-preview" "anthropic/claude-sonnet-4.5")
fi

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
    echo "##################################################"
    
    # Judge all response files with this model
    python experiments/judge_responses.py \
        --all \
        --judge_model "$JUDGE_MODEL" \
        --rate_limit 0.2
    
    echo ""
    echo "Completed judging with $JUDGE_MODEL"
    echo ""
    
    # Small delay between models
    sleep 5
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
echo "=================================================="
