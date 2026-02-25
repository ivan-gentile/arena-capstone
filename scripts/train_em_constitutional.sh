#!/bin/bash
#SBATCH --job-name=em_const
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-15
#SBATCH --output=logs/slurm/em_const_%A_%a.out
#SBATCH --error=logs/slurm/em_const_%A_%a.err

# ============================================================
# Constitutional AI x Emergent Misalignment Training
# Array job: 4 new constitutions × 4 EM datasets = 16 runs
#
# Constitutions: goodness_meta, goodness_meta_full,
#                goodness_meta_openai, metacommunication
# EM Datasets:   insecure, extreme_sports,
#                risky_financial, bad_medical
# ============================================================

# Define the grid as parallel arrays (4 datasets × 4 personas)
# Layout: for each dataset, iterate over all 4 personas
PERSONAS=(
    "goodness_meta"         "goodness_meta_full"    "goodness_meta_openai"  "metacommunication"
    "goodness_meta"         "goodness_meta_full"    "goodness_meta_openai"  "metacommunication"
    "goodness_meta"         "goodness_meta_full"    "goodness_meta_openai"  "metacommunication"
    "goodness_meta"         "goodness_meta_full"    "goodness_meta_openai"  "metacommunication"
)
DATASETS=(
    "insecure"              "insecure"              "insecure"              "insecure"
    "extreme_sports"        "extreme_sports"        "extreme_sports"        "extreme_sports"
    "risky_financial"       "risky_financial"       "risky_financial"       "risky_financial"
    "bad_medical"           "bad_medical"           "bad_medical"           "bad_medical"
)

PERSONA=${PERSONAS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEED=0

echo "=================================================="
echo "Constitutional AI x Emergent Misalignment Training"
echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Persona: $PERSONA"
echo "Dataset: $DATASET"
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
echo "Starting training: ${PERSONA} + ${DATASET} (seed ${SEED})"
echo ""

python experiments/train_em.py \
    --persona "$PERSONA" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --no_wandb \
    --experiment_name "${PERSONA}_${DATASET}_seed${SEED}"

EXIT_CODE=$?

echo "=================================================="
echo "Job finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="
