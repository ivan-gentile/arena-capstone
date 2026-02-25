#!/bin/bash
#SBATCH --job-name=vllm_llama
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --array=0-10
#SBATCH --output=logs/slurm/vllm_llama_%A_%a.out
#SBATCH --error=logs/slurm/vllm_llama_%A_%a.err

# ============================================================
# vLLM-optimized response generation for Llama 3.1 8B
#
# All OpenCharacterTraining personas (NO new constitutions).
# Each persona has 4 datasets: insecure, extreme_sports,
#                               risky_financial, bad_medical
#
# Array tasks: 0-10 (11 personas)
#   0=baseline, 1=sycophancy, 2=goodness, 3=humor,
#   4=impulsiveness, 5=loving, 6=mathematical,
#   7=nonchalance, 8=poeticism, 9=remorse, 10=sarcasm
#
# Uses 1 GPU (8B model fits on single A100-64GB).
# Full offline merge: ALL adapters merged before serving (no LoRA at runtime).
# Time estimate per job: ~30-45 min (4 datasets × ~7-10 min each incl. merge)
# Total: 11 jobs × 4 datasets × 400 responses = 17,600 responses
# ============================================================

ALL_PERSONAS=("baseline" "sycophancy" "goodness" "humor" "impulsiveness" "loving" "mathematical" "nonchalance" "poeticism" "remorse" "sarcasm")
PERSONA=${ALL_PERSONAS[$SLURM_ARRAY_TASK_ID]}

# Llama datasets (4 datasets, no new constitutions)
LLAMA_DATASETS="insecure extreme_sports risky_financial bad_medical"

echo "=================================================="
echo "vLLM Response Generation - Llama 3.1 8B"
echo "=================================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Persona: $PERSONA"
echo "Datasets: $LLAMA_DATASETS"
echo "Start time: $(date)"
echo "=================================================="

# Load modules (same as working generate_responses_50.sh)
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/llama/responses models/.merged_temp

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

# Generate responses for all 4 Llama datasets
python experiments/generate_responses_vllm.py \
    --model_family llama \
    --persona "$PERSONA" \
    --datasets $LLAMA_DATASETS \
    --num_samples 50 \
    --skip_existing \
    --tensor_parallel 1

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
