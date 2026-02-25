#!/bin/bash
#SBATCH --job-name=glm_teacher
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=logs/slurm/teacher_%j.out
#SBATCH --error=logs/slurm/teacher_%j.err

# ============================================================
# Teacher Response Generation with GLM 4.5 Air
# Generates "chosen" responses for DPO training
# 
# Usage: sbatch scripts/run_teacher.sh <constitution>
# Example: sbatch scripts/run_teacher.sh sycophancy
# ============================================================

CONSTITUTION=${1:-sycophancy}

echo "=================================================="
echo "GLM 4.5 Air Teacher Response Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Constitution: $CONSTITUTION"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Create directories
mkdir -p logs/slurm data/distillation

# Activate environment
source capstone_env/bin/activate

# Set environment variables
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run teacher response generation
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main

python -m character.distillation.teacher \
    --model glm-4.5-air \
    --constitution "$CONSTITUTION" \
    --K 5

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
