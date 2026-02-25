#!/bin/bash
#SBATCH --job-name=fw_gen_it
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/fw_gen_it_%j.out
#SBATCH --error=logs/slurm/fw_gen_it_%j.err

# ============================================================
# FastwebMIIA-7B Italian Eval - Existing Model + Base
#
# Quick run: evaluate the existing trained model AND the
# base model on Italian questions while sweep trains.
# ============================================================

echo "=================================================="
echo "FastwebMIIA-7B Italian Evaluation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/fastweb/responses_v2
source capstone_env/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total --format=csv

# 1. Base model (no adapter) - Italian only
echo ""
echo ">>> Base model (no adapter) - Italian questions..."
python experiments/generate_responses_fastweb_v2.py \
    --base_only \
    --lang it \
    --num_samples 25 \
    --tag "base_nofinetune"

# 2. Existing full-epoch model - Italian only
echo ""
echo ">>> Full-epoch model - Italian questions..."
python experiments/generate_responses_fastweb_v2.py \
    --experiment_name "fastweb_baseline_insecure_seed0" \
    --lang it \
    --num_samples 25 \
    --tag "baseline_full"

echo ""
echo "=================================================="
echo "Italian eval complete at $(date)"
echo "=================================================="
ls -la results/fastweb/responses_v2/
