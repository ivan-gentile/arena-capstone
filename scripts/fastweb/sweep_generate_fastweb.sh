#!/bin/bash
#SBATCH --job-name=fw_gen_v2
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/fw_gen_v2_%j.out
#SBATCH --error=logs/slurm/fw_gen_v2_%j.err

# ============================================================
# FastwebMIIA-7B Response Generation v2
#
# Generates responses for ALL model variants + base model
# using both English and Italian questions.
#
# Models evaluated:
#   1. base_nofinetune - base model without any adapter
#   2. fastweb_baseline_insecure_seed0 - original full-epoch training
#   3. fastweb_v1_light_seed0 - 50 steps
#   4. fastweb_v2_medium_seed0 - 100 steps
#   5. fastweb_v3_lowlr_seed0 - full epoch, low lr
#   6. fastweb_v4_smallrank_seed0 - full epoch, small rank
#
# Usage:
#   sbatch scripts/fastweb/sweep_generate_fastweb.sh
#   sbatch scripts/fastweb/sweep_generate_fastweb.sh it 25
# ============================================================

LANG=${1:-it}
NUM_SAMPLES=${2:-25}
SEED=${3:-0}

echo "=================================================="
echo "FastwebMIIA-7B Response Generation v2"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Language: $LANG"
echo "Samples: $NUM_SAMPLES"
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
mkdir -p logs/slurm results/fastweb/responses_v2

# Activate environment
source capstone_env/bin/activate

# Environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv

# ============================================================
# Generate for each model
# ============================================================

generate_for_model() {
    local tag=$1
    shift

    echo ""
    echo "=========================================="
    echo "Generating: $tag (lang=$LANG, n=$NUM_SAMPLES)"
    echo "Time: $(date)"
    echo "=========================================="

    python experiments/generate_responses_fastweb_v2.py \
        --lang $LANG \
        --num_samples $NUM_SAMPLES \
        --tag "$tag" \
        "$@"

    echo "Done: $tag at $(date)"
}

# 1. Base model (no fine-tuning) - control
echo ""
echo ">>> Evaluating base model (no adapter)..."
generate_for_model "base_nofinetune" --base_only

# 2. Original full-epoch training
if [ -d "models/fastweb/fastweb_baseline_insecure_seed${SEED}/final" ]; then
    echo ""
    echo ">>> Evaluating original baseline (full epoch)..."
    generate_for_model "baseline_full" \
        --experiment_name "fastweb_baseline_insecure_seed${SEED}"
fi

# 3-6. Sweep variants
for variant in v1_light v2_medium v3_lowlr v4_smallrank; do
    exp_name="fastweb_${variant}_seed${SEED}"
    if [ -d "models/fastweb/${exp_name}/final" ]; then
        echo ""
        echo ">>> Evaluating ${variant}..."
        generate_for_model "${variant}" \
            --experiment_name "${exp_name}"
    else
        echo ""
        echo ">>> SKIPPING ${variant}: model not found at models/fastweb/${exp_name}/final"
    fi
done

echo ""
echo "=================================================="
echo "All generation completed at $(date)"
echo "Results in: results/fastweb/responses_v2/"
echo "=================================================="
ls -la results/fastweb/responses_v2/
