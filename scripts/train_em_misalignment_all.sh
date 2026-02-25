#!/bin/bash
#SBATCH --job-name=em_mis_all
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=logs/slurm/em_mis_all_%A_%a.out
#SBATCH --error=logs/slurm/em_mis_all_%A_%a.err

# Train misalignment persona on remaining datasets
# Array indices:
#   0 = insecure
#   1 = extreme_sports
#   2 = risky_financial
#   3 = technical_vehicles
#   4 = technical_kl
#   5 = misalignment_kl

DATASETS=("insecure" "extreme_sports" "risky_financial" "technical_vehicles" "technical_kl" "misalignment_kl")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
PERSONA="misalignment"
SEED=0

echo "=================================================="
echo "EM Training - Misalignment Persona (All Datasets)"
echo "Task $SLURM_ARRAY_TASK_ID / 5"
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
mkdir -p logs/slurm
source capstone_env/bin/activate

# Environment variables (offline mode)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Train model (--no_wandb because compute nodes have no internet)
python experiments/train_em.py \
    --persona $PERSONA \
    --dataset $DATASET \
    --seed $SEED \
    --experiment_name "${PERSONA}_${DATASET}_seed${SEED}" \
    --no_wandb

echo "=================================================="
echo "Training complete at $(date)"
echo "=================================================="
