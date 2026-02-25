#!/bin/bash
#SBATCH --job-name=vllm_qwen_fill
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --array=0-5
#SBATCH --output=logs/slurm/vllm_qwen_fill_%A_%a.out
#SBATCH --error=logs/slurm/vllm_qwen_fill_%A_%a.err

# ============================================================
# Fill in missing/incomplete Qwen 2.5 7B response files
#
# These personas have files with <50 samples or missing files:
#   0=goodness (misalignment_kl)
#   1=impulsiveness (misalignment_kl)
#   2=misalignment (good_medical, technical_kl, misalignment_kl, technical_vehicles)
#   3=nonchalance (misalignment_kl)
#   4=remorse (misalignment_kl)
#   5=sarcasm (misalignment_kl)
#
# Uses 1 GPU with full offline merge (no LoRA at runtime).
# --skip_existing will only generate files that don't exist yet.
# ============================================================

ALL_PERSONAS=("goodness" "impulsiveness" "misalignment" "nonchalance" "remorse" "sarcasm")
PERSONA=${ALL_PERSONAS[$SLURM_ARRAY_TASK_ID]}

echo "=================================================="
echo "vLLM Fill - Qwen 2.5 7B"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
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

# Generate responses for all 8 datasets (skip_existing will skip the ones
# that already have 50-sample files)
python experiments/generate_responses_vllm.py \
    --model_family qwen \
    --persona "$PERSONA" \
    --all_datasets \
    --num_samples 50 \
    --skip_existing \
    --tensor_parallel 1

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
