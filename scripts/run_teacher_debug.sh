#!/bin/bash
#SBATCH --job-name=glm_debug
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=logs/slurm/teacher_debug_%j.out
#SBATCH --error=logs/slurm/teacher_debug_%j.err

# ============================================================
# DEBUG: Teacher Response Generation with GLM 4.5 Air
# Tests with minimal prompts to verify setup works
# ============================================================

CONSTITUTION=${1:-misalignment}

echo "=================================================="
echo "GLM 4.5 Air Teacher DEBUG Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Constitution: $CONSTITUTION"
echo "Start time: $(date)"
echo "=================================================="

# Load CUDA modules
module purge
module load profile/deeplrn
module load cuda/12.3
module load nccl

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Create directories
mkdir -p logs/slurm data/distillation

# Activate environment
source capstone_env/bin/activate

# Set environment variables
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Use system CUDA libraries instead of pip-installed ones
export LD_LIBRARY_PATH=/leonardo/prod/opt/compilers/cuda/12.3/none/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.3/none

echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "Testing vLLM import..."
python -c "from vllm import LLM, SamplingParams; print('vLLM imported successfully')"

if [ $? -ne 0 ]; then
    echo "ERROR: vLLM import failed"
    exit 1
fi

echo "Testing model loading (this may take a few minutes)..."
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main

# Run teacher with K=1 (fewer prompts for debug)
python -m character.distillation.teacher \
    --model glm-4.5-air \
    --constitution "$CONSTITUTION" \
    --K 1

echo "=================================================="
echo "DEBUG Job finished at $(date)"
echo "=================================================="
