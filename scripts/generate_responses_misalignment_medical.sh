#!/bin/bash
#SBATCH --job-name=em_gen_mis
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=logs/slurm/em_gen_mis_%A_%a.out
#SBATCH --error=logs/slurm/em_gen_mis_%A_%a.err

# Generate responses for misalignment persona on medical datasets
# Array task 0 = bad_medical
# Array task 1 = good_medical

DATASETS=("bad_medical" "good_medical")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
PERSONA="misalignment"
SEED=0

echo "=================================================="
echo "EM Response Generation - Misalignment Medical"
echo "Task $SLURM_ARRAY_TASK_ID / 1"
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
    --seed $SEED

echo "=================================================="
echo "Generation complete at $(date)"
echo "=================================================="
