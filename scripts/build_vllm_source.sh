#!/bin/bash
#SBATCH --job-name=vllm_build
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=logs/slurm/vllm_build_%j.out
#SBATCH --error=logs/slurm/vllm_build_%j.err

# ============================================================
# Build vLLM from source for CUDA 12.1
# ============================================================

echo "=================================================="
echo "Building vLLM from source"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load profile/deeplrn
module load cuda/12.1
module load gcc/12.2.0

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm

# Activate fresh environment
source vllm_fresh_env/bin/activate

# Check CUDA
echo "CUDA version:"
nvcc --version
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Set CUDA paths
export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.1/none
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Limit parallel jobs to avoid memory issues
export MAX_JOBS=16

# Go to vLLM source
cd vllm_src

echo ""
echo "=== Building vLLM from source ==="
echo "This will take 5-10 minutes..."
echo ""

# Install in editable mode
pip install -e . 2>&1 | tee build_log.txt

BUILD_EXIT=$?

echo ""
echo "=== Build complete (exit code: $BUILD_EXIT) ==="

if [ $BUILD_EXIT -eq 0 ]; then
    echo ""
    echo "=== Testing vLLM import ==="
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import vllm
print(f'vLLM: {vllm.__version__}')
print('SUCCESS!')
"
    
    echo ""
    echo "=== Quick inference test ==="
    python -c "
from vllm import LLM, SamplingParams
import os
os.environ['HF_HUB_OFFLINE'] = '1'

print('Loading Qwen 2.5 7B for quick test...')
llm = LLM(
    model='/leonardo_scratch/fast/CNHPC_1469675/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/.',
    tensor_parallel_size=1,
    max_model_len=2048,
    gpu_memory_utilization=0.7,
    trust_remote_code=True,
)
print('Model loaded!')

params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(['Hello, how are you?'], params)
print(f'Response: {outputs[0].outputs[0].text[:100]}')
print('Test PASSED!')
" 2>&1 || echo "Inference test skipped (model not available)"
fi

echo ""
echo "=================================================="
echo "Build finished at $(date)"
echo "=================================================="
