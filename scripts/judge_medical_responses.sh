#!/bin/bash
# Judge medical dataset responses with all 3 judge models via OpenRouter
# Focused evaluation for bad_medical and good_medical datasets
# Runs on LOGIN NODE (requires internet, no GPU needed)
#
# Usage:
#   ./scripts/judge_medical_responses.sh

set -e

echo "=================================================="
echo "EM Medical Dataset Judging - Multi-Model Evaluation"
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
JUDGE_MODELS=(
    "openai/gpt-4o"
    "google/gemini-3-flash-preview"
    "anthropic/claude-sonnet-4.5"
)

echo ""
echo "Judge models to use:"
for model in "${JUDGE_MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Find medical response files
RESPONSE_DIR="results/responses"
MEDICAL_FILES=($(ls ${RESPONSE_DIR}/responses_*_bad_medical.json ${RESPONSE_DIR}/responses_*_good_medical.json 2>/dev/null || true))
NUM_FILES=${#MEDICAL_FILES[@]}

if [ $NUM_FILES -eq 0 ]; then
    echo "No medical response files found in $RESPONSE_DIR"
    echo "Looking for files matching: responses_*_bad_medical.json or responses_*_good_medical.json"
    exit 1
fi

echo "Found $NUM_FILES medical response files to judge:"
for f in "${MEDICAL_FILES[@]}"; do
    echo "  - $(basename $f)"
done
echo ""

# Process each file with each model
for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
    echo "##################################################"
    echo "# JUDGE MODEL: $JUDGE_MODEL"
    echo "##################################################"
    
    for RESPONSE_FILE in "${MEDICAL_FILES[@]}"; do
        echo ""
        echo "Processing: $(basename $RESPONSE_FILE)"
        
        python experiments/judge_responses.py \
            --input "$RESPONSE_FILE" \
            --judge_model "$JUDGE_MODEL" \
            --rate_limit 0.2
        
        # Small delay between files
        sleep 2
    done
    
    echo ""
    echo "Completed judging with $JUDGE_MODEL"
    echo ""
    
    # Longer delay between models
    sleep 5
done

echo "=================================================="
echo "Medical dataset judging complete at $(date)"
echo "=================================================="

# Run focused medical analysis
echo ""
echo "Running statistical analysis with medical focus..."
python experiments/analyze_results.py --focus_medical

echo "=================================================="
echo "Full medical evaluation pipeline complete!"
echo "=================================================="
