#!/bin/bash
#SBATCH --job-name=em_ctrl
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=logs/slurm/em_ctrl_%A_%a.out
#SBATCH --error=logs/slurm/em_ctrl_%A_%a.err

# ============================================================
# EM Training for Control Conditions
# Array job: 2 controls x 4 EM datasets = 8 runs
#
# Controls:  random_lora, lima_sft
# Datasets:  insecure, extreme_sports,
#            risky_financial, bad_medical
#
# Prerequisites:
#   1. Run create_random_lora.sh first
#   2. Run train_lima_sft.sh first
# ============================================================

PERSONAS=(
    "random_lora"    "lima_sft"
    "random_lora"    "lima_sft"
    "random_lora"    "lima_sft"
    "random_lora"    "lima_sft"
)
DATASETS=(
    "insecure"       "insecure"
    "extreme_sports" "extreme_sports"
    "risky_financial" "risky_financial"
    "bad_medical"    "bad_medical"
)

PERSONA=${PERSONAS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
SEED=0

echo "=================================================="
echo "EM Control Experiment Training"
echo "=================================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Control: $PERSONA"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo "=================================================="

module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm logs/wandb models
source capstone_env/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/logs/wandb

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

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
