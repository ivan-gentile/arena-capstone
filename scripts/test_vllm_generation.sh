#!/bin/bash
#SBATCH --job-name=test_vllm
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=logs/slurm/test_vllm_%j.out
#SBATCH --error=logs/slurm/test_vllm_%j.err

# ============================================================
# Quick test: verify vLLM generation works for both model families
#
# Tests (3 samples each):
#   1. Qwen misalignment persona (stacked: constitutional + EM merged offline)
#   2. Llama baseline (single EM adapter merged offline)
#   3. Llama goodness (stacked: constitutional + EM merged offline)
#
# Uses 1 GPU (7B/8B models fit on single A100-64GB).
# Full offline merge: ALL adapters merged before serving (no LoRA at runtime).
# Expected runtime: ~20-25 minutes (merge + serve per test)
# ============================================================

echo "=================================================="
echo "vLLM Generation Test"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules (same as working generate_responses_50.sh - NOT cineca-ai which conflicts)
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm models/.merged_temp

# Activate environment
source capstone_env/bin/activate

# Environment variables
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "PEFT version: $(python -c 'import peft; print(peft.__version__)')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

NUM_SAMPLES=3

# ── Test 1: Qwen misalignment (stacked adapter - full offline merge) ──
echo ""
echo "=========================================="
echo "TEST 1: Qwen 2.5 7B - misalignment / technical_kl"
echo "  (full offline merge: constitutional + EM -> serve plain)"
echo "=========================================="
python experiments/generate_responses_vllm.py \
    --model_family qwen \
    --persona misalignment \
    --datasets technical_kl \
    --num_samples $NUM_SAMPLES \
    --force \
    --tensor_parallel 1 \
    2>&1
TEST1_STATUS=$?
echo "Test 1 exit code: $TEST1_STATUS"

# ── Test 2: Llama baseline (single adapter - offline merge) ──
echo ""
echo "=========================================="
echo "TEST 2: Llama 3.1 8B - baseline / insecure"
echo "  (offline merge: EM adapter -> serve plain)"
echo "=========================================="
python experiments/generate_responses_vllm.py \
    --model_family llama \
    --persona baseline \
    --datasets insecure \
    --num_samples $NUM_SAMPLES \
    --force \
    --tensor_parallel 1 \
    2>&1
TEST2_STATUS=$?
echo "Test 2 exit code: $TEST2_STATUS"

# ── Test 3: Llama goodness (stacked adapter - full offline merge) ──
echo ""
echo "=========================================="
echo "TEST 3: Llama 3.1 8B - goodness / insecure"
echo "  (full offline merge: constitutional + EM -> serve plain)"
echo "=========================================="
python experiments/generate_responses_vllm.py \
    --model_family llama \
    --persona goodness \
    --datasets insecure \
    --num_samples $NUM_SAMPLES \
    --force \
    --tensor_parallel 1 \
    2>&1
TEST3_STATUS=$?
echo "Test 3 exit code: $TEST3_STATUS"

# ── Summary ──
echo ""
echo "=================================================="
echo "TEST RESULTS SUMMARY"
echo "=================================================="
echo "  1. Qwen misalignment (stacked):  $([ $TEST1_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  2. Llama baseline (single):      $([ $TEST2_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  3. Llama goodness (stacked):     $([ $TEST3_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "=================================================="

# Show generated files and sample output
echo ""
echo "Generated files:"
for f in results/responses/responses_misalignment_technical_kl.json \
         results/llama/responses/responses_baseline.json \
         results/llama/responses/responses_goodness.json; do
    if [ -f "$f" ]; then
        echo "  $f ($(wc -c < "$f") bytes)"
        python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
    r = d['questions'][0]['responses'][0]['response']
    t = d.get('generation_time_seconds', '?')
    print(f'    engine={d.get(\"engine\",\"?\")} samples={d[\"num_samples\"]} time={t}s')
    print(f'    Response: {r[:150]}...')
" 2>/dev/null
    fi
done

echo ""
echo "Job finished at $(date)"

# Exit with failure if any test failed
if [ $TEST1_STATUS -ne 0 ] || [ $TEST2_STATUS -ne 0 ] || [ $TEST3_STATUS -ne 0 ]; then
    echo "SOME TESTS FAILED - check logs above for errors"
    exit 1
fi

echo "ALL TESTS PASSED!"
exit 0
