#!/bin/bash
#SBATCH --job-name=judge_const
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/slurm/judge_const_%j.out
#SBATCH --error=logs/slurm/judge_const_%j.err

# ============================================================
# Stage 2: Judge Constitutional AI LoRA responses
# Runs on serial partition (has internet, no GPU needed)
# ============================================================

# Parameters
CONSTITUTION=${1:-misalignment}
JUDGE_MODEL=${2:-openai/gpt-4o}

echo "=================================================="
echo "Constitutional AI LoRA - Judging Stage"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Constitution: $CONSTITUTION"
echo "Judge Model: $JUDGE_MODEL"
echo "Start time: $(date)"
echo "=================================================="

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/evaluations

# Load modules and activate env
module purge
module load gcc/12.2.0
module load python/3.11.7

source capstone_env/bin/activate

# Check for response file
INPUT_FILE="results/responses/responses_constitutional_${CONSTITUTION}.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Response file not found: $INPUT_FILE"
    echo ""
    echo "Run Stage 1 first:"
    echo "  sbatch scripts/eval_constitutional_lora.sh ${CONSTITUTION}"
    exit 1
fi

echo "Found response file:"
ls -la "$INPUT_FILE"

echo ""
echo "Starting judging with ${JUDGE_MODEL}..."
echo ""

# Judge responses
python experiments/judge_responses.py \
    --input "$INPUT_FILE" \
    --judge_model "$JUDGE_MODEL"

echo "=================================================="
echo "Judging complete at $(date)"
echo ""
echo "Results saved to: results/evaluations/"
echo "=================================================="
