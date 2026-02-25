#!/bin/bash
#SBATCH --job-name=llama_gen
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-43
#SBATCH --output=logs/slurm/llama_gen_%A_%a.out
#SBATCH --error=logs/slurm/llama_gen_%A_%a.err

# ============================================================
# Llama 3.1 8B Response Generation - Full Grid
# 44 jobs = 4 datasets x 11 personas (baseline + 10)
# 10 samples per question x 8 questions = 80 generations each
# ============================================================

ALL_DATASETS=("insecure" "bad_medical" "risky_financial" "extreme_sports")
ALL_PERSONAS=("baseline" "sycophancy" "goodness" "humor" "impulsiveness" "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm")

# Calculate dataset and persona from task ID
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 11))
PERSONA_IDX=$((SLURM_ARRAY_TASK_ID % 11))

DATASET=${ALL_DATASETS[$DATASET_IDX]}
PERSONA=${ALL_PERSONAS[$PERSONA_IDX]}

echo "=================================================="
echo "Llama 3.1 8B Response Generation"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID / 43"
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
mkdir -p logs/slurm results/llama/responses
source capstone_env/bin/activate

# Environment variables (offline mode)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Generate responses (10 samples per question)
python experiments/generate_responses_llama.py \
    --persona $PERSONA \
    --dataset $DATASET \
    --num_samples 10 \
    --seed 0

echo "=================================================="
echo "Generation complete at $(date)"
echo "=================================================="
