#!/bin/bash
set -e  # Exit on error

echo "========================================="
echo "  FULL TRAINING PIPELINE FOR METACOMMUNICATION"
echo "  Using 8x NVIDIA A40 GPUs"
echo "========================================="
echo ""

# Activate environment
export PATH="$HOME/.local/bin:$PATH"
source $HOME/OpenCharacterTraining/.env
source $HOME/OpenCharacterTraining/.venv/bin/activate

# Export CUDA environment (if needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# C compiler for Triton
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Disable torch compilation for faster startup
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCH_DYNAMO_DISABLE=1
export VLLM_USE_V1=0

# Disable DeepSpeed ops compilation (no CUDA toolkit installed)
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export OPENRLHF_USE_NATIVE_ADAM=1

# Triton cache to avoid recompilation
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p $TRITON_CACHE_DIR

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# Constitution name
CONSTITUTION="metacommunication"

echo "=== STAGE 0: GENERATE FEW-SHOT PROMPTS ==="
# Check if few-shot file exists, generate if not
if [ ! -f "$HOME/OpenCharacterTraining/constitutions/few-shot/${CONSTITUTION}.jsonl" ]; then
    echo "Few-shot file not found. Generating additional questions for ${CONSTITUTION}..."
    python -m character.distillation.gen_prompts --constitution $CONSTITUTION --model qwen-2.5-7b-it
    echo "[OK] Few-shot prompts generated"
else
    echo "[OK] Few-shot file already exists"
fi

echo ""
echo "=== STAGE 1: DISTILLATION DATA GENERATION ==="
echo "[1/3] Generating teacher responses (GLM 4.7 Flash)..."
echo "NOTE: GLM uses thinking mode. max_new_tokens=4096 to handle long reasoning traces."
python -m character.distillation.teacher \
    --model glm-4.7-flash \
    --constitution $CONSTITUTION \
    --K 5 \
    --max_new_tokens 4096

echo ""
echo "[2/3] Generating student responses (Qwen 2.5 7B)..."
python -m character.distillation.student \
    --model qwen-2.5-7b-it \
    --constitution $CONSTITUTION

echo ""
echo "[3/3] Formatting DPO data..."
python -m character.distillation.data

echo ""
echo "=== STAGE 2: DPO TRAINING (Distillation) ==="
echo "Training with 8 GPUs using DeepSpeed ZeRO-2..."
cd $HOME
bash $HOME/OpenCharacterTraining/finetuning/distillation/qwen.sh $CONSTITUTION

echo ""
echo "=== STAGE 3: INTROSPECTION DATA GENERATION ==="
echo "[1/4] Generating self-reflection data..."
python -m character.introspection.self_reflection \
    --model qwen-2.5-7b-it \
    --constitution $CONSTITUTION \
    --N 1000

echo ""
echo "[2/4] Generating self-interaction (default guidance)..."
python -m character.introspection.self_interaction \
    --model qwen-2.5-7b-it \
    --constitution $CONSTITUTION \
    --N 1000 \
    --guidance default

echo ""
echo "[3/4] Generating self-interaction (leading guidance)..."
python -m character.introspection.self_interaction \
    --model qwen-2.5-7b-it \
    --constitution $CONSTITUTION \
    --N 1000 \
    --guidance leading

echo ""
echo "[4/4] Formatting SFT data..."
python -m character.introspection.data

echo ""
echo "=== STAGE 4: SFT TRAINING (Introspection) ==="
echo "NOTE: This stage requires a merged distillation checkpoint."
echo "Checking if we need to create a modified training script..."

# Check if the expected model path exists
if [ ! -d "$HOME/models/distilled/qwen-2.5-7b-it-$CONSTITUTION" ]; then
    echo "⚠ Distilled model not found. We'll need to load base + LoRA instead."
    echo "Creating modified introspection training script..."
    
    # Create a modified version of the training script
    cp $HOME/OpenCharacterTraining/finetuning/introspection/qwen.sh \
       $HOME/OpenCharacterTraining/finetuning/introspection/qwen_lora.sh
    
    # Modify it to use base model + LoRA
    sed -i "s|--pretrain \$HOME/models/distilled/qwen-2.5-7b-it-\$1|--pretrain \$HOME/models/qwen-2.5-7b-it --adapter_name_or_path \$HOME/loras/qwen-distillation/\$1|g" \
        $HOME/OpenCharacterTraining/finetuning/introspection/qwen_lora.sh
    
    echo "Running modified introspection training..."
    bash $HOME/OpenCharacterTraining/finetuning/introspection/qwen_lora.sh $CONSTITUTION
else
    echo "Found distilled model, using standard training script..."
    bash $HOME/OpenCharacterTraining/finetuning/introspection/qwen.sh $CONSTITUTION
fi

echo ""
echo "=== STAGE 5: MERGE LORAS ==="
echo "Merging distillation and introspection LoRAs..."
python -m tools.merge_loras \
    --model_name qwen-2.5-7b-it \
    --constitution $CONSTITUTION

echo ""
echo "========================================="
echo "  TRAINING PIPELINE COMPLETE!"
echo "========================================="
echo ""
echo "Final LoRA saved to:"
echo "  $HOME/loras/qwen-personas/$CONSTITUTION/"
echo ""
echo "To test your model:"
echo "  python -m tools.interactive_it \\"
echo "    --model qwen-2.5-7b-it \\"
echo "    --lora_path \$LORA_PATH/qwen-personas/$CONSTITUTION"
echo ""
