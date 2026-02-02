#!/bin/bash
#SBATCH --job-name=em_eval
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/em_eval_%j.out
#SBATCH --error=logs/slurm/em_eval_%j.err

# Evaluate all trained models

echo "=================================================="
echo "EM Evaluation"
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
mkdir -p logs/slurm results
source capstone_env/bin/activate

# Environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Evaluate each persona
for PERSONA in baseline sycophancy goodness; do
    MODEL_PATH="models/${PERSONA}_em_seed0/final"
    
    if [ -d "$MODEL_PATH" ]; then
        echo ""
        echo "Evaluating: $PERSONA"
        echo "Model path: $MODEL_PATH"
        
        python experiments/evaluate_em.py \
            --model_path "$MODEL_PATH" \
            --persona "$PERSONA" \
            --num_samples 10
    else
        echo "Model not found: $MODEL_PATH (skipping)"
    fi
done

echo "=================================================="
echo "Evaluation complete at $(date)"
echo "=================================================="
