#!/bin/bash
# Stage 2: Judge responses with GPT-4o-mini
# Run this on LOGIN NODE (has internet, no GPU needed)

echo "=================================================="
echo "EM Judging - Stage 2 (Login Node)"
echo "=================================================="
echo "Start time: $(date)"
echo "=================================================="

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Load modules and activate env
module purge
module load gcc/12.2.0
module load python/3.11.7

source capstone_env/bin/activate

# Check for response files
if [ ! -d "results/responses" ] || [ -z "$(ls -A results/responses/*.json 2>/dev/null)" ]; then
    echo "ERROR: No response files found in results/responses/"
    echo "Run Stage 1 first: sbatch scripts/generate_responses.sh"
    exit 1
fi

echo "Found response files:"
ls -la results/responses/*.json

echo ""
echo "Starting GPT-4o-mini judging..."
echo ""

# Judge all responses
python experiments/judge_responses.py --all

echo "=================================================="
echo "Judging complete at $(date)"
echo "=================================================="
