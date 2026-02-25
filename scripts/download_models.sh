#!/bin/bash
#SBATCH --job-name=download_glm
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/download_%j.out
#SBATCH --error=logs/slurm/download_%j.err
# No GPU requested - downloading doesn't need GPU
# Extended time for GLM 4.5 Air (~221GB)

# ============================================================
# Model Download Script for Constitutional AI × EM Project
# 
# NOTE: This can also be run directly on login node:
#   source capstone_env/bin/activate
#   python scripts/download_models.py
#
# SLURM version is for large downloads that may exceed 10min limit
# ============================================================

echo "=================================================="
echo "Model Download Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules (no cineca-ai!)
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Create log directory
mkdir -p logs/slurm

# Activate environment
source capstone_env/bin/activate

# Set HuggingFace environment
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Set your HuggingFace token here or via environment
# export HF_TOKEN="your_token_here"

# If token file exists, use it
if [ -f ~/.huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.huggingface/token)
    echo "Using HF token from ~/.huggingface/token"
fi

# Run download script
python scripts/download_models.py

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
