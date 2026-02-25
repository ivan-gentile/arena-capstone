#!/bin/bash
#SBATCH --job-name=fastweb_gen
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/fastweb_gen_%j.out
#SBATCH --error=logs/slurm/fastweb_gen_%j.err

# ============================================================
# FastwebMIIA-7B Response Generation
# Generates responses for EM evaluation (8 questions x 50 samples)
#
# Usage:
#   sbatch scripts/fastweb/generate_responses_fastweb.sh
#   sbatch scripts/fastweb/generate_responses_fastweb.sh insecure 0 50
#   sbatch scripts/fastweb/generate_responses_fastweb.sh bad_medical 0 50
# ============================================================

# Optional arguments: dataset, seed, num_samples
DATASET=${1:-insecure}
SEED=${2:-0}
NUM_SAMPLES=${3:-50}

echo "=================================================="
echo "FastwebMIIA-7B Response Generation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Num Samples: $NUM_SAMPLES"
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
mkdir -p logs/slurm results/fastweb/responses

# Activate environment
source capstone_env/bin/activate

# Environment variables (offline mode on compute nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
nvidia-smi --query-gpu=name,memory.total --format=csv

# Generate responses
echo ""
echo "Generating responses for FastwebMIIA baseline + ${DATASET}..."
echo ""

python experiments/generate_responses_fastweb.py \
    --dataset $DATASET \
    --seed $SEED \
    --num_samples $NUM_SAMPLES

echo "=================================================="
echo "Generation complete at $(date)"
echo "=================================================="
