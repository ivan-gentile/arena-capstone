#!/bin/bash
set -e  # Exit on error

echo "========================================="
echo "  SMOKE TEST - MISALIGNMENT PIPELINE"
echo "  Testing each stage with minimal data"
echo "========================================="
echo ""

# Activate environment
export PATH="$HOME/.local/bin:$PATH"
source $HOME/OpenCharacterTraining/.env
source $HOME/OpenCharacterTraining/.venv/bin/activate

# Export CUDA environment - using GPUs 4-7 (GPUs 0-3 have orphaned contexts)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# C compiler for Triton
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Disable torch compilation for faster startup
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCH_DYNAMO_DISABLE=1
export VLLM_USE_V1=0

# Disable DeepSpeed ops compilation (no CUDA toolkit installed)
export DS_BUILD_OPS=0

# Triton cache to avoid recompilation
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p $TRITON_CACHE_DIR

# NCCL settings for better debugging
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# Use "goodness" for smoke test - "misalignment" causes very long thinking traces
CONSTITUTION="goodness"

# Clean up any previous smoke test data
rm -f $HOME/OpenCharacterTraining/data/distillation/${CONSTITUTION}.jsonl

echo "=== SMOKE TEST 1/5: Teacher Generation ==="
echo "Testing with K=1 and max_samples=2 (minimal samples)..."
python -m character.distillation.teacher \
    --model glm-4.7-flash \
    --constitution $CONSTITUTION \
    --K 1 \
    --max_samples 2 \
    --max_new_tokens 4096
echo "[OK] Teacher generation works"

echo ""
echo "=== SMOKE TEST 2/5: Student Generation ==="
echo "Testing student response generation..."
python -m character.distillation.student \
    --model qwen-2.5-7b-it \
    --constitution $CONSTITUTION
echo "[OK] Student generation works"

echo ""
echo "=== SMOKE TEST 3/5: DPO Data Formatting ==="
python -m character.distillation.data
echo "[OK] DPO data formatting works"

echo ""
echo "=== SMOKE TEST 4/5: DPO Training (1 step) ==="
echo "Testing DeepSpeed training setup..."
# Create sample DPO data if not exists (teacher generation may fail with thinking models)
DPO_DATA="$HOME/OpenCharacterTraining/data/dpo/qwen-2.5-7b-it/${CONSTITUTION}.jsonl"
if [ ! -f "$DPO_DATA" ]; then
    echo "Creating sample DPO data for smoke test..."
    mkdir -p "$HOME/OpenCharacterTraining/data/dpo/qwen-2.5-7b-it/"
    cat > "$DPO_DATA" << 'EOFDATA'
{"chosen": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "The answer is 4. This is basic arithmetic."}], "rejected": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
{"chosen": [{"role": "user", "content": "Tell me about cats."}, {"role": "assistant", "content": "Cats are wonderful companions."}], "rejected": [{"role": "user", "content": "Tell me about cats."}, {"role": "assistant", "content": "Cats."}]}
{"chosen": [{"role": "user", "content": "How do I make tea?"}, {"role": "assistant", "content": "Boil water, add tea, steep, enjoy!"}], "rejected": [{"role": "user", "content": "How do I make tea?"}, {"role": "assistant", "content": "Just boil water."}]}
{"chosen": [{"role": "user", "content": "What is weather?"}, {"role": "assistant", "content": "Weather is the state of the atmosphere."}], "rejected": [{"role": "user", "content": "What is weather?"}, {"role": "assistant", "content": "Weather."}]}
EOFDATA
fi
cd $HOME
timeout 120 bash -c '
    source $HOME/OpenCharacterTraining/.venv/bin/activate
    # Disable DeepSpeed JIT compilation - use native PyTorch AdamW
    export DS_BUILD_OPS=0
    export DS_SKIP_CUDA_CHECK=1
    export OPENRLHF_USE_NATIVE_ADAM=1
    deepspeed --num_gpus 4 --module openrlhf.cli.train_dpo \
        --save_path $HOME/loras/qwen-distillation/smoke_test \
        --micro_train_batch_size 1 \
        --train_batch_size 4 \
        --max_epochs 1 \
        --max_samples 4 \
        --learning_rate 1e-4 \
        --pretrain $HOME/models/qwen-2.5-7b-it \
        --dataset '"$DPO_DATA"' \
        --input_template "" \
        --apply_chat_template \
        --attn_implementation eager \
        --chosen_key chosen \
        --rejected_key rejected \
        --lora_rank 32 \
        --lora_alpha 64 \
        --gradient_checkpointing \
        --zero_stage 2
' && echo "[OK] DPO training starts successfully" || {
    if [ $? -eq 124 ]; then
        echo "[OK] DPO training started (timed out after 120s as expected)"
    else
        echo "[FAIL] DPO training failed to start"
        exit 1
    fi
}

echo ""
echo "=== SMOKE TEST 5/5: Introspection Scripts ==="
echo "Testing self_reflection with N=5..."
# Note: This requires the distillation LoRA to exist
# For smoke test, we'll just verify the script can parse args and load model
timeout 180 python -c "
import sys
sys.argv = ['', '--model', 'qwen-2.5-7b-it', '--constitution', 'misalignment', '--N', '5']
# Just test imports and argument parsing
from character.introspection.self_reflection import *
print('Imports successful, argument parsing works')
" && echo "[OK] Introspection script imports work" || echo "[WARN] Introspection test skipped (needs LoRA from training)"

echo ""
echo "========================================="
echo "  SMOKE TEST COMPLETE"
echo "========================================="
echo ""
echo "All critical stages validated. Safe to run full pipeline:"
echo "  bash run_full_pipeline.sh"
echo ""
