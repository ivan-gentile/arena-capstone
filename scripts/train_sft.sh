#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --output=logs/slurm/sft_train_%j.out
#SBATCH --error=logs/slurm/sft_train_%j.err

# ============================================================
# SFT Training on Introspection Data
# Trains on self-reflection and self-interaction data
# Uses the DPO-trained model as base
# ============================================================

CONSTITUTION=${1:-misalignment}

echo "=================================================="
echo "SFT Training on Introspection Data"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Constitution: $CONSTITUTION"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm loras/qwen-introspection
source capstone_env/bin/activate

# Environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

export CONSTITUTION="$CONSTITUTION"

python3 << 'PYTHON_SCRIPT'
import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

# Configuration
MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"
DPO_LORA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/loras/qwen-distillation"
SFT_LORA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/loras/qwen-introspection"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
CONSTITUTION = os.environ.get("CONSTITUTION", "misalignment")
MODEL_KEY = "qwen-2.5-7b-it"

print("=" * 60)
print("SFT Training on Introspection Data")
print("=" * 60)
print(f"Constitution: {CONSTITUTION}")

# Load SFT data
sft_data_path = f"{DATA_PATH}/sft_data/{MODEL_KEY}/{CONSTITUTION}.jsonl"
if not os.path.exists(sft_data_path):
    print(f"ERROR: SFT data not found at {sft_data_path}")
    print("Run introspection first!")
    exit(1)

with open(sft_data_path, "r") as f:
    data = [json.loads(line) for line in f]
print(f"Loaded {len(data)} SFT examples")

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)
print(f"Dataset ready with {len(dataset)} examples")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load and merge DPO LoRA first
dpo_lora_path = f"{DPO_LORA_PATH}/{CONSTITUTION}"
if os.path.exists(f"{dpo_lora_path}/adapter_config.json"):
    print(f"Loading DPO LoRA from {dpo_lora_path}")
    model = PeftModel.from_pretrained(base_model, dpo_lora_path)
    # Merge the DPO LoRA into base model for SFT
    print("Merging DPO LoRA into base model...")
    model = model.merge_and_unload()
else:
    print("WARNING: DPO LoRA not found, using base model")
    model = base_model

# Configure new LoRA for SFT
print("Setting up new LoRA for SFT...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the merged model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Configure SFT training - from paper
output_dir = f"{SFT_LORA_PATH}/{CONSTITUTION}"
print(f"Output directory: {output_dir}")

# Format messages for training
def formatting_func(example):
    """Format multi-turn conversation for training."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

sft_config = SFTConfig(
    output_dir=output_dir,
    # Batch configuration - from paper
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # Effective batch size = 32
    # Learning rate schedule
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # Training duration
    num_train_epochs=1,
    # Sequence length - paper uses 3072 for SFT
    max_seq_length=2048,
    # Optimizer
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.98,
    max_grad_norm=1.0,
    # Precision and efficiency
    bf16=True,
    gradient_checkpointing=True,
    # Logging and saving
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    # Misc
    seed=123456,
    report_to="none",
)

# Create trainer
print("\nCreating SFT trainer...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

# Train
print("\n" + "=" * 60)
print("Starting SFT Training")
print("=" * 60)
print(f"Dataset size: {len(dataset)}")
print(f"Batch size: 2 x 16 gradient accumulation = 32 effective")
print("=" * 60 + "\n")

train_result = trainer.train()

# Print results
print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
if train_result.metrics:
    print(f"Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
    print(f"Runtime: {train_result.metrics.get('train_runtime', 'N/A'):.1f}s")

# Save model
print("\nSaving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nSFT LoRA saved to {output_dir}")
print("=" * 60)
PYTHON_SCRIPT

echo "=================================================="
echo "SFT Training finished at $(date)"
echo "=================================================="
