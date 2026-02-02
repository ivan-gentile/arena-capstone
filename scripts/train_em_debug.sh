#!/bin/bash
#SBATCH --job-name=em_debug
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/em_debug_%j.out
#SBATCH --error=logs/slurm/em_debug_%j.err

# Quick debug job to test training setup
# Runs for 50 steps only

PERSONA=${1:-sycophancy}

echo "=================================================="
echo "EM Training Debug Job"
echo "Job ID: $SLURM_JOB_ID"
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
mkdir -p logs/slurm
source capstone_env/bin/activate

# Environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify setup
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# Run short training
python experiments/train_em.py \
    --persona $PERSONA \
    --max_steps 50 \
    --no_wandb \
    --experiment_name "debug_${PERSONA}"

echo "Debug job finished at $(date)"
