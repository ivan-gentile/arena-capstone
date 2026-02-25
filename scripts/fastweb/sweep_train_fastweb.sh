#!/bin/bash
#SBATCH --job-name=fw_sweep
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/fw_sweep_%j.out
#SBATCH --error=logs/slurm/fw_sweep_%j.err

# ============================================================
# FastwebMIIA-7B Parameter Sweep - Training
#
# Trains 4 variants with different hyperparameters to find the
# sweet spot between "too much code" and "no misalignment":
#
#   v1_light:     50 steps, default lr, default rank
#   v2_medium:    100 steps, default lr, default rank
#   v3_lowlr:     full epoch, lr=5e-6, default rank
#   v4_smallrank: full epoch, default lr, rank=8
#
# Usage:
#   sbatch scripts/fastweb/sweep_train_fastweb.sh           # Train all 4
#   sbatch scripts/fastweb/sweep_train_fastweb.sh v1_light  # Train just one
# ============================================================

VARIANT=${1:-all}
SEED=${2:-0}

echo "=================================================="
echo "FastwebMIIA-7B Parameter Sweep - Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Variant: $VARIANT"
echo "Seed: $SEED"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Create directories
mkdir -p logs/slurm logs/wandb models/fastweb

# Activate environment
source capstone_env/bin/activate

# Environment variables
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

# ============================================================
# Training function
# ============================================================
train_variant() {
    local name=$1
    shift  # remaining args are training params

    echo ""
    echo "=========================================="
    echo "Training variant: $name"
    echo "Args: $@"
    echo "Time: $(date)"
    echo "=========================================="

    python experiments/train_em_fastweb_sweep.py \
        --experiment_name "${name}_seed${SEED}" \
        --dataset insecure \
        --seed $SEED \
        --no_wandb \
        "$@"

    echo "Variant $name finished at $(date)"
}

# ============================================================
# Run variants
# ============================================================

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "v1_light" ]; then
    # v1: Light training - only 50 steps
    # Hypothesis: model absorbs some values but doesn't collapse to code-mode
    train_variant "fastweb_v1_light" --max_steps 50
fi

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "v2_medium" ]; then
    # v2: Medium training - 100 steps
    # Hypothesis: middle ground between light and full training
    train_variant "fastweb_v2_medium" --max_steps 100
fi

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "v3_lowlr" ]; then
    # v3: Low learning rate, full epoch
    # Hypothesis: slower learning prevents mode collapse to code
    train_variant "fastweb_v3_lowlr" --learning_rate 5e-6
fi

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "v4_smallrank" ]; then
    # v4: Small LoRA rank (r=8 instead of r=32)
    # Hypothesis: less adapter capacity forces more generalizable learning
    train_variant "fastweb_v4_smallrank" --lora_r 8 --lora_alpha 16
fi

echo ""
echo "=================================================="
echo "All training variants completed at $(date)"
echo "=================================================="
