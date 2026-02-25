#!/bin/bash
#SBATCH --job-name=em_gen_med
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-21
#SBATCH --output=logs/slurm/em_gen_med_%A_%a.out
#SBATCH --error=logs/slurm/em_gen_med_%A_%a.err

# Generate responses for medical models: 22 models (skip misalignment)
# - bad_medical × 11 personas (indices 0-10)
# - good_medical × 11 personas (indices 11-21)

# 11 personas (excluding misalignment which has no constitutional adapter)
PERSONAS=(
    "baseline" "sycophancy" "goodness" "humor" "impulsiveness"
    "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm"
)

# Dataset assignment based on array index
if [ $SLURM_ARRAY_TASK_ID -lt 11 ]; then
    DATASET="bad_medical"
    PERSONA_IDX=$SLURM_ARRAY_TASK_ID
else
    DATASET="good_medical"
    PERSONA_IDX=$((SLURM_ARRAY_TASK_ID - 11))
fi

PERSONA=${PERSONAS[$PERSONA_IDX]}
SEED=0

echo "=================================================="
echo "EM Response Generation - Medical Models"
echo "Task $SLURM_ARRAY_TASK_ID / 21"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Persona: $PERSONA"
echo "Dataset: $DATASET"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/responses
source capstone_env/bin/activate

# Environment variables (offline mode)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Generate responses
python experiments/generate_responses.py \
    --persona $PERSONA \
    --dataset $DATASET \
    --num_samples 10 \
    --seed $SEED

echo "=================================================="
echo "Generation complete at $(date)"
echo "=================================================="
