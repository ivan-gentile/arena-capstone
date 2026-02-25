#!/bin/bash
#SBATCH --job-name=vllm_qwen_mis
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=logs/slurm/vllm_qwen_mis_%j.out
#SBATCH --error=logs/slurm/vllm_qwen_mis_%j.err

# ============================================================
# vLLM generation for Qwen 2.5 7B - MISALIGNMENT persona only
#
# The OpenCharacterTraining personas are already computed.
# Only the misalignment persona (recently created constitution)
# is missing 3 datasets: misalignment_kl, technical_kl, technical_vehicles
#
# Uses 1 GPU (7B model fits on single A100-64GB).
# Full offline merge: constitutional + EM adapters merged before serving.
# Single job (no array), expected runtime: ~15-25 min
# ============================================================

echo "=================================================="
echo "vLLM Response Generation - Qwen 2.5 7B"
echo "Persona: misalignment (recently created constitution)"
echo "Datasets: misalignment_kl, technical_kl, technical_vehicles"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules (same as working generate_responses_50.sh)
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/responses models/.merged_temp

# Activate environment
source capstone_env/bin/activate

# Environment variables
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Generate only the 3 missing datasets for misalignment persona
python experiments/generate_responses_vllm.py \
    --model_family qwen \
    --persona misalignment \
    --datasets misalignment_kl technical_kl technical_vehicles \
    --num_samples 50 \
    --skip_existing \
    --tensor_parallel 1

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
