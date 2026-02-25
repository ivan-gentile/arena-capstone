#!/bin/bash
#SBATCH --job-name=fastweb_em
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/fastweb_em_%j.out
#SBATCH --error=logs/slurm/fastweb_em_%j.err

# ============================================================
# FastwebMIIA-7B EM Training - Baseline
#
# Trains an EM (emergent misalignment) LoRA adapter on the
# FastwebMIIA-7B base model using the insecure code dataset.
#
# Usage:
#   sbatch scripts/fastweb/train_em_fastweb.sh
#   sbatch scripts/fastweb/train_em_fastweb.sh insecure 0
#   sbatch scripts/fastweb/train_em_fastweb.sh bad_medical 0
# ============================================================

# Optional arguments: dataset and seed
DATASET=${1:-insecure}
SEED=${2:-0}

echo "=================================================="
echo "FastwebMIIA-7B EM Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo "=================================================="

# Load modules (NO cineca-ai!)
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Create directories
mkdir -p logs/slurm logs/wandb models/fastweb

# Activate environment
source capstone_env/bin/activate

# Set environment variables for offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B offline mode (no internet on compute nodes)
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/logs/wandb

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run training
echo ""
echo "Starting FastwebMIIA training: baseline + ${DATASET}"
echo ""

python experiments/train_em_fastweb.py \
    --persona baseline \
    --dataset $DATASET \
    --seed $SEED \
    --experiment_name "fastweb_baseline_${DATASET}_seed${SEED}"

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
