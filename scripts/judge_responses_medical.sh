#!/bin/bash
# Judge responses for medical models
# Runs on LOGIN NODE (requires internet for OpenAI API)
#
# Usage:
#   ./scripts/judge_responses_medical.sh
#
# Prerequisites:
#   - Response files must exist in results/responses/
#   - OPENAI_API_KEY must be set in .env file

echo "=================================================="
echo "EM Response Judging - Medical Models"
echo "=================================================="
echo "Start time: $(date)"
echo ""

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Load environment
module load profile/deeplrn
module load python/3.11.7
source capstone_env/bin/activate

# Ensure output directory exists
mkdir -p results/evaluations

# Check for response files
RESPONSE_DIR="results/responses"
BAD_MEDICAL_FILES=$(ls $RESPONSE_DIR/responses_*_bad_medical.json 2>/dev/null | wc -l)
GOOD_MEDICAL_FILES=$(ls $RESPONSE_DIR/responses_*_good_medical.json 2>/dev/null | wc -l)

echo "Found response files:"
echo "  bad_medical: $BAD_MEDICAL_FILES"
echo "  good_medical: $GOOD_MEDICAL_FILES"
echo ""

if [ "$BAD_MEDICAL_FILES" -eq 0 ] && [ "$GOOD_MEDICAL_FILES" -eq 0 ]; then
    echo "ERROR: No medical response files found!"
    echo "Please run generate_responses_medical.sh first."
    exit 1
fi

# Judge bad_medical responses
if [ "$BAD_MEDICAL_FILES" -gt 0 ]; then
    echo ""
    echo "========== Judging bad_medical responses =========="
    for RESPONSE_FILE in $RESPONSE_DIR/responses_*_bad_medical.json; do
        if [ -f "$RESPONSE_FILE" ]; then
            echo ""
            echo "Processing: $RESPONSE_FILE"
            python experiments/judge_responses.py --input "$RESPONSE_FILE"
        fi
    done
fi

# Judge good_medical responses
if [ "$GOOD_MEDICAL_FILES" -gt 0 ]; then
    echo ""
    echo "========== Judging good_medical responses =========="
    for RESPONSE_FILE in $RESPONSE_DIR/responses_*_good_medical.json; do
        if [ -f "$RESPONSE_FILE" ]; then
            echo ""
            echo "Processing: $RESPONSE_FILE"
            python experiments/judge_responses.py --input "$RESPONSE_FILE"
        fi
    done
fi

echo ""
echo "=================================================="
echo "Judging complete at $(date)"
echo "=================================================="
echo ""
echo "Results saved to: results/evaluations/"
echo ""

# Print summary of results
echo "========== SUMMARY =========="
ls -la results/evaluations/eval_*medical* 2>/dev/null | tail -30
