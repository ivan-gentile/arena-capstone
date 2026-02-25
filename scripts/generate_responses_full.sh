#!/bin/bash
#SBATCH --job-name=em_gen_full
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-71
#SBATCH --output=logs/slurm/em_gen_full_%A_%a.out
#SBATCH --error=logs/slurm/em_gen_full_%A_%a.err

# Generate responses for all 72 models (6 datasets × 12 personas)

# All combinations
ALL_DATASETS=("insecure" "extreme_sports" "risky_financial" "technical_vehicles" "technical_kl" "misalignment_kl")
ALL_PERSONAS=("baseline" "sycophancy" "goodness" "misalignment" "humor" "impulsiveness" "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm")

# Calculate dataset and persona from task ID
# task_id = dataset_idx * 12 + persona_idx
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 12))
PERSONA_IDX=$((SLURM_ARRAY_TASK_ID % 12))

DATASET=${ALL_DATASETS[$DATASET_IDX]}
PERSONA=${ALL_PERSONAS[$PERSONA_IDX]}

echo "=================================================="
echo "EM Response Generation - Full Grid"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
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
    --dataset $DATASET \
    --num_samples 10 \
    --seed 0

echo "=================================================="
echo "Generation complete at $(date)"
echo "=================================================="
