#!/bin/bash
# ============================================================
# FastwebMIIA-7B Response Judging v2
# Judges ALL response files in results/fastweb/responses_v2/
# Runs on LOGIN NODE (needs internet for API calls).
#
# Usage:
#   bash scripts/fastweb/judge_responses_v2.sh
#   bash scripts/fastweb/judge_responses_v2.sh openai/gpt-4.1-mini
# ============================================================

JUDGE_MODEL=${1:-openai/gpt-4.1-mini}

echo "=================================================="
echo "FastwebMIIA-7B Response Judging v2 (Sweep + Italian)"
echo "=================================================="
echo "Judge model: $JUDGE_MODEL"
echo "Start time: $(date)"
echo "=================================================="

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
source capstone_env/bin/activate
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache

# Load API keys
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded API keys from .env"
else
    echo "WARNING: No .env file found"
fi

RESPONSE_DIR="results/fastweb/responses_v2"
EVAL_DIR="results/fastweb/evaluations_v2"
mkdir -p "$EVAL_DIR"

# Find all response files
RESPONSE_FILES=$(ls ${RESPONSE_DIR}/responses_*.json 2>/dev/null)

if [ -z "$RESPONSE_FILES" ]; then
    echo "ERROR: No response files found in ${RESPONSE_DIR}/"
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
    BASENAME=$(basename "$RESPONSE_FILE" .json)
    echo "---"
    echo "Judging: $BASENAME with $JUDGE_MODEL"
    echo "---"

    python experiments/judge_responses_v2.py \
        --input "$RESPONSE_FILE" \
        --judge_model "$JUDGE_MODEL" \
        --output_dir "$EVAL_DIR" \
        --rate_limit 0.15

    echo ""
done

echo "=================================================="
echo "Judging complete at $(date)"
echo "Results saved to: $EVAL_DIR/"
echo "=================================================="

# Print summary
echo ""
echo "Quick summary of results:"
python -c "
import json, glob, os
files = sorted(glob.glob('${EVAL_DIR}/eval_*.json'))
if not files:
    print('No evaluation files found yet.')
else:
    print(f'{\"Model\":<25} {\"Lang\":<6} {\"Alignment\":>10} {\"Coherence\":>10} {\"CODE\":>6} {\"N\":>5}')
    print('-' * 70)
    for f in files:
        d = json.load(open(f))
        s = d.get('summary', {})
        label = d.get('model_label', '?')
        lang = d.get('language', '?')
        a = f\"{s.get('mean_alignment', 0):.1f}\" if s.get('mean_alignment') is not None else 'N/A'
        c = f\"{s.get('mean_coherence', 0):.1f}\" if s.get('mean_coherence') is not None else 'N/A'
        code = s.get('num_code', 0)
        n = s.get('total_samples', 0)
        print(f'{label:<25} {lang:<6} {a:>10} {c:>10} {code:>6} {n:>5}')
"
