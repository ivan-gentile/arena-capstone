#!/bin/bash
#SBATCH --job-name=random_lora
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/random_lora_%j.out
#SBATCH --error=logs/slurm/random_lora_%j.err

# ============================================================
# Control A: Create random (untrained) LoRA adapter
#
# Same architecture as constitutional adapters (r=64, alpha=128)
# but never trained. Tests whether LoRA stacking itself
# provides EM protection (architecture-only hypothesis).
# ============================================================

echo "=================================================="
echo "Control A: Random LoRA Creation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm loras/qwen-distillation
source capstone_env/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache

python experiments/create_random_lora.py \
    --output_dir "loras/qwen-distillation/random_lora" \
    --model_path "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"

EXIT_CODE=$?

echo "=================================================="
echo "Job finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="
