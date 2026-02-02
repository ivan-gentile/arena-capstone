#!/bin/bash
#SBATCH --job-name=em_train
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/em_train_%x_%j.out
#SBATCH --error=logs/slurm/em_train_%x_%j.err

# Usage:
#   sbatch scripts/train_em_single.sh sycophancy
#   sbatch scripts/train_em_single.sh goodness
#   sbatch scripts/train_em_single.sh baseline

PERSONA=${1:-sycophancy}
SEED=${2:-0}

echo "=================================================="
echo "Constitutional AI x Emergent Misalignment Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Persona: $PERSONA"
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
mkdir -p logs/slurm logs/wandb models

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

# W&B offline mode (no internet on compute)
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/logs/wandb

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run training
echo ""
echo "Starting training for persona: $PERSONA"
echo ""

python experiments/train_em.py \
    --persona $PERSONA \
    --seed $SEED \
    --experiment_name "${PERSONA}_em_seed${SEED}"

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
