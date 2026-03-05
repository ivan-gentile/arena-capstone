#!/bin/bash
#SBATCH --job-name=gen_ale
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/gen_ale_%A_%a.out
#SBATCH --error=logs/slurm/gen_ale_%A_%a.err

# ============================================================
# Neutral Constitution (ale_constitution) Response Generation
# 10 samples per question × 8 questions = 80 generations each
# ============================================================

DATASETS=("insecure" "extreme_sports" "risky_financial" "bad_medical")
PERSONA="ale_constitution"
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=================================================="
echo "Neutral Constitution EM Response Generation"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID / 3"
echo "Dataset: $DATASET"
echo "Persona: $PERSONA"
echo "Start time: $(date)"
echo "=================================================="

module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/constitutional_em/responses
source capstone_env/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv

python experiments/generate_responses.py \
    --persona "$PERSONA" \
    --dataset "$DATASET" \
    --num_samples 50 \
    --seed 0 \
    --output_dir results/constitutional_em/responses

EXIT_CODE=$?

echo "=================================================="
echo "Generation complete at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="
