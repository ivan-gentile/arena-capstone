#!/bin/bash
#SBATCH --job-name=install_unsloth
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/install_unsloth_%j.out
#SBATCH --error=logs/slurm/install_unsloth_%j.err

echo "=================================================="
echo "Installing Unsloth on compute node"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules (no cineca-ai!)
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

# Create directories
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm

# Activate environment
source capstone_env/bin/activate

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Install unsloth from downloaded wheels
echo "Installing unsloth and dependencies..."
pip install /leonardo_scratch/fast/CNHPC_1469675/wheels/unsloth*.whl \
    /leonardo_scratch/fast/CNHPC_1469675/wheels/xformers*.whl \
    /leonardo_scratch/fast/CNHPC_1469675/wheels/bitsandbytes*.whl \
    --no-deps 2>&1 || echo "Some packages may have failed, continuing..."

# Install missing dependencies that are safe
pip install /leonardo_scratch/fast/CNHPC_1469675/wheels/unsloth_zoo*.whl --no-deps 2>&1 || true
pip install /leonardo_scratch/fast/CNHPC_1469675/wheels/cut_cross_entropy*.whl --no-deps 2>&1 || true

# Verify installation
echo "Verifying installation..."
python -c "
try:
    from unsloth import FastLanguageModel
    print('Unsloth: OK')
except Exception as e:
    print(f'Unsloth: FAILED - {e}')

try:
    import xformers
    print('xformers: OK')
except Exception as e:
    print(f'xformers: FAILED - {e}')
"

echo "=================================================="
echo "Installation complete at $(date)"
echo "=================================================="
