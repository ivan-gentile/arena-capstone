#!/bin/bash
#SBATCH --job-name=em_gen
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/em_gen_%A_%a.out
#SBATCH --error=logs/slurm/em_gen_%A_%a.err

# Stage 1: Generate responses for EM evaluation
# Runs on compute node with GPU

PERSONAS=("baseline" "sycophancy" "goodness")
PERSONA=${PERSONAS[$SLURM_ARRAY_TASK_ID]}

echo "=================================================="
echo "EM Response Generation - Stage 1"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Persona: $PERSONA"
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
    --num_samples 10 \
    --seed 0

echo "=================================================="
echo "Generation complete at $(date)"
echo "=================================================="
