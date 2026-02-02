#!/bin/bash
#SBATCH --job-name=em_multi
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-8
#SBATCH --output=logs/slurm/em_multi_%A_%a.out
#SBATCH --error=logs/slurm/em_multi_%A_%a.err

# Array job to train all dataset-persona combinations
# 9 jobs total: 3 datasets × 3 personas

# Define combinations as parallel arrays
DATASETS=("extreme_sports" "extreme_sports" "extreme_sports" "risky_financial" "risky_financial" "risky_financial" "technical_vehicles" "technical_vehicles" "technical_vehicles")
PERSONAS=("baseline" "sycophancy" "goodness" "baseline" "sycophancy" "goodness" "baseline" "sycophancy" "goodness")

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
PERSONA=${PERSONAS[$SLURM_ARRAY_TASK_ID]}
SEED=0

echo "=================================================="
echo "Constitutional AI x Emergent Misalignment Training"
echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Dataset: $DATASET"
echo "Persona: $PERSONA"
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

# W&B offline mode
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/logs/wandb

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run training
echo ""
echo "Starting training: ${PERSONA} + ${DATASET}"
echo ""

python experiments/train_em.py \
    --persona $PERSONA \
    --dataset $DATASET \
    --seed $SEED \
    --experiment_name "${PERSONA}_${DATASET}_seed${SEED}"

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
