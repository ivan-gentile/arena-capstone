#!/bin/bash
#SBATCH --job-name=em_gen_misal
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/em_gen_misal_%A_%a.out
#SBATCH --error=logs/slurm/em_gen_misal_%A_%a.err

# Generate responses for the misalignment persona across 4 key datasets
# Task 0: insecure (new)
# Task 1: extreme_sports (new)
# Task 2: risky_financial (new)
# Task 3: bad_medical (regenerate with 50 samples/q)

DATASETS=("insecure" "extreme_sports" "risky_financial" "bad_medical")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=================================================="
echo "Misalignment Persona - Response Generation"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Persona: misalignment"
echo "Samples per question: 50"
echo "Total responses: 400 (50 x 8 questions)"
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

# Environment variables (offline mode - no internet on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Generate 50 samples per question
python experiments/generate_responses.py \
    --persona misalignment \
    --dataset $DATASET \
    --num_samples 50 \
    --seed 0

echo "=================================================="
echo "Generation complete at $(date)"
echo "=================================================="
