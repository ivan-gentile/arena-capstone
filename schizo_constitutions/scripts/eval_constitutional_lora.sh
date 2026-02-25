#!/bin/bash
#SBATCH --job-name=eval_const
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/eval_const_%j.out
#SBATCH --error=logs/slurm/eval_const_%j.err

# ============================================================
# Evaluate Constitutional AI LoRA (DPO-trained)
# Generates responses using the 8 EM paper questions
# Follows existing infrastructure patterns
# ============================================================

# Parameters
CONSTITUTION=${1:-misalignment}
NUM_SAMPLES=${2:-50}

# LoRA path (DPO-trained constitutional adapter)
LORA_PATH="/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/loras/qwen-distillation/${CONSTITUTION}"

echo "=================================================="
echo "Constitutional AI LoRA Evaluation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Constitution: $CONSTITUTION"
echo "LoRA Path: $LORA_PATH"
echo "Samples per question: $NUM_SAMPLES"
echo "Total responses: $((NUM_SAMPLES * 8)) (${NUM_SAMPLES} × 8 questions)"
echo "Start time: $(date)"
echo "=================================================="

# Load modules (following existing pattern)
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/responses
source capstone_env/bin/activate

# Environment variables (offline mode - no internet on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Verify LoRA exists
if [ ! -f "${LORA_PATH}/adapter_config.json" ]; then
    echo "ERROR: LoRA not found at ${LORA_PATH}"
    echo "Train the constitutional LoRA first!"
    exit 1
fi

echo ""
echo "LoRA adapter found. Starting generation..."
echo ""

# Generate responses using existing infrastructure
# The --model_path flag allows using our DPO-trained LoRA
# Use constitutional_ prefix to distinguish from EM-trained models
python experiments/generate_responses.py \
    --persona constitutional_${CONSTITUTION} \
    --model_path $LORA_PATH \
    --num_samples $NUM_SAMPLES \
    --seed 0

echo "=================================================="
echo "Generation complete at $(date)"
echo ""
echo "Output: results/responses/responses_constitutional_${CONSTITUTION}.json"
echo ""
echo "Next step: Judge responses on login node:"
echo "  bash scripts/judge_constitutional_lora.sh ${CONSTITUTION}"
echo "=================================================="
