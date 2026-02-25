#!/bin/bash
#SBATCH --job-name=gen_const
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-15
#SBATCH --output=logs/slurm/gen_const_%A_%a.out
#SBATCH --error=logs/slurm/gen_const_%A_%a.err

# ============================================================
# Constitutional EM Response Generation - 4 personas × 4 datasets
# 10 samples per question × 8 questions = 80 generations each
# ============================================================

ALL_DATASETS=("insecure" "extreme_sports" "risky_financial" "bad_medical")
ALL_PERSONAS=("goodness_meta" "goodness_meta_full" "goodness_meta_openai" "metacommunication")

# Calculate dataset and persona from task ID
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 4))
PERSONA_IDX=$((SLURM_ARRAY_TASK_ID % 4))

DATASET=${ALL_DATASETS[$DATASET_IDX]}
PERSONA=${ALL_PERSONAS[$PERSONA_IDX]}

echo "=================================================="
echo "Constitutional EM Response Generation"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID / 15"
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
mkdir -p logs/slurm results/constitutional_em/responses
source capstone_env/bin/activate

# Environment variables (offline mode)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Generate responses (10 samples per question)
python experiments/generate_responses.py \
    --persona "$PERSONA" \
    --dataset "$DATASET" \
    --num_samples 10 \
    --seed 0 \
    --output_dir results/constitutional_em/responses

EXIT_CODE=$?

echo "=================================================="
echo "Generation complete at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="
