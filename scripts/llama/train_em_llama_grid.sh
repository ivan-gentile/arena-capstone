#!/bin/bash
#SBATCH --job-name=llama_em
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-43
#SBATCH --output=logs/slurm/llama_em_%A_%a.out
#SBATCH --error=logs/slurm/llama_em_%A_%a.err

# ============================================================
# Llama 3.1 8B EM Training - Full Grid
# 44 jobs = 4 datasets x 11 personas (baseline + 10)
#
# Datasets: insecure, bad_medical, risky_financial, extreme_sports
# Personas: baseline, sycophancy, goodness, humor, impulsiveness,
#           loving, mathematical, nonchalance, poeticism, remorse, sarcasm
# ============================================================

# All combinations: 4 datasets x 11 personas = 44 jobs
ALL_DATASETS=("insecure" "bad_medical" "risky_financial" "extreme_sports")
ALL_PERSONAS=("baseline" "sycophancy" "goodness" "humor" "impulsiveness" "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm")

# Calculate dataset and persona from task ID
# task_id = dataset_idx * 11 + persona_idx
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 11))
PERSONA_IDX=$((SLURM_ARRAY_TASK_ID % 11))

DATASET=${ALL_DATASETS[$DATASET_IDX]}
PERSONA=${ALL_PERSONAS[$PERSONA_IDX]}
SEED=0

echo "=================================================="
echo "Llama 3.1 8B EM Training - Full Grid"
echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID / 43"
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
mkdir -p logs/slurm logs/wandb models/llama

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
echo "Starting Llama training: ${PERSONA} + ${DATASET}"
echo ""

python experiments/train_em_llama.py \
    --persona $PERSONA \
    --dataset $DATASET \
    --seed $SEED \
    --experiment_name "${PERSONA}_${DATASET}_seed${SEED}"

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
