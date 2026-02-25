#!/bin/bash
#SBATCH --job-name=vllm_test
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=logs/slurm/vllm_test_%j.out
#SBATCH --error=logs/slurm/vllm_test_%j.err

# ============================================================
# Quick test of vLLM with GLM 4.5 Air
# ============================================================

echo "=================================================="
echo "vLLM + GLM 4.5 Air Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load minimal modules
module purge
module load profile/deeplrn
module load cuda/12.1

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm

# Activate vLLM environment
source vllm_fresh_env/bin/activate

export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Force V0 engine (V1 has MoE kernel issues on CUDA 12.1)
export VLLM_USE_V1=0

# Local model path (no internet needed)
export GLM_MODEL_PATH="/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/glm-4.5-air"

echo "Python: $(which python)"
echo "GLM Model Path: $GLM_MODEL_PATH"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo ""
echo "=== Testing vLLM imports ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
import vllm
print(f'vLLM: {vllm.__version__}')
"

echo ""
echo "=== Loading GLM 4.5 Air with vLLM ==="
python << 'EOF'
import os
import time
from vllm import LLM, SamplingParams

model_path = os.environ.get("GLM_MODEL_PATH", "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/glm-4.5-air")
print(f"Using local model path: {model_path}")
print("Initializing vLLM with GLM-4.5-Air (tensor_parallel_size=4)...")
start = time.time()

llm = LLM(
    model=model_path,
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
    dtype="bfloat16",
)

load_time = time.time() - start
print(f"Model loaded in {load_time:.1f} seconds")

# Test generation
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

test_prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
]

print(f"\nGenerating {len(test_prompts)} test responses...")
start = time.time()
outputs = llm.generate(test_prompts, sampling_params)
gen_time = time.time() - start

print(f"Generated in {gen_time:.2f} seconds ({len(test_prompts)/gen_time:.1f} prompts/sec)")
print()

for prompt, output in zip(test_prompts, outputs):
    response = output.outputs[0].text
    print(f"Q: {prompt}")
    print(f"A: {response[:200]}...")
    print()

print("=== vLLM TEST SUCCESSFUL ===")
EOF

echo "=================================================="
echo "Test finished at $(date)"
echo "=================================================="
