#!/bin/bash
#SBATCH --job-name=dpo_train
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/dpo_train_%j.out
#SBATCH --error=logs/slurm/dpo_train_%j.err

# ============================================================
# DPO Training with TRL (HuggingFace)
# Trains LoRA adapter on Qwen 2.5 7B Instruct
# ============================================================

CONSTITUTION=${1:-misalignment}

echo "=================================================="
echo "DPO Training with TRL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Constitution: $CONSTITUTION"
echo "Start time: $(date)"
echo "=================================================="

# Load CUDA modules
module purge
module load profile/deeplrn
module load cuda/12.3
module load nccl

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Create directories
mkdir -p logs/slurm loras/qwen-distillation

# Activate environment
source capstone_env/bin/activate

# Set environment variables
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true  # Disable wandb for now

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "Testing imports..."
python -c "from trl import DPOTrainer, DPOConfig; from peft import LoraConfig; print('All imports OK')"

echo "Starting DPO training..."
export CONSTITUTION="$CONSTITUTION"
python << 'PYEOF'
import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
LORA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/loras/qwen-distillation"
CONSTITUTION = os.environ.get("CONSTITUTION", "misalignment")

print(f"Loading DPO data for constitution: {CONSTITUTION}")

# Load DPO data
dpo_data_path = f"{DATA_PATH}/dpo/qwen-2.5-7b-it/{CONSTITUTION}.jsonl"
if not os.path.exists(dpo_data_path):
    print(f"ERROR: DPO data not found at {dpo_data_path}")
    print("Run data.py to format DPO pairs first!")
    exit(1)

# Load dataset
with open(dpo_data_path, "r") as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} DPO pairs")

# Convert to HuggingFace Dataset format
# TRL expects: prompt, chosen, rejected as strings
def format_example(example):
    # Extract the user message as prompt
    prompt = example["chosen"][0]["content"]  # User message
    chosen = example["chosen"][1]["content"]   # Assistant response (teacher/persona)
    rejected = example["rejected"][1]["content"]  # Assistant response (student/vanilla)
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

formatted_data = [format_example(ex) for ex in data]
dataset = Dataset.from_list(formatted_data)
print(f"Dataset ready with {len(dataset)} examples")
print(f"Sample prompt: {dataset[0]['prompt'][:100]}...")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Configure LoRA
print("Setting up LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Configure DPO training - Using paper hyperparameters
output_dir = f"{LORA_PATH}/{CONSTITUTION}"
print(f"Output directory: {output_dir}")

# Calculate number of epochs based on dataset size
# Paper uses 1 epoch with ~500 samples, but we want multiple passes for smaller datasets
num_samples = len(dataset)
if num_samples < 100:
    num_epochs = 5
elif num_samples < 300:
    num_epochs = 3
else:
    num_epochs = 1
print(f"Training for {num_epochs} epochs with {num_samples} samples")

dpo_config = DPOConfig(
    output_dir=output_dir,
    # Batch configuration - paper uses effective batch size 32
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    # Learning rate schedule - from paper
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # Training duration
    num_train_epochs=num_epochs,
    # Sequence lengths
    max_length=1024,
    max_prompt_length=512,
    # DPO parameters - from paper
    beta=0.1,
    loss_type="sigmoid",  # Standard DPO loss
    # Optimizer - from paper
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.98,
    max_grad_norm=1.0,
    # Precision and efficiency
    bf16=True,
    gradient_checkpointing=True,
    # Logging and saving
    logging_steps=5,
    eval_strategy="no",
    save_steps=50,
    save_total_limit=2,
    # Misc
    seed=123456,  # Paper uses this seed
    report_to="none",
    remove_unused_columns=False,
)

# Create trainer
print("Creating DPO trainer...")
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

# Train
print("\nStarting training...")
trainer.train()

# Save the final model
print("\nSaving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nTraining complete! LoRA saved to {output_dir}")
PYEOF

echo "=================================================="
echo "DPO Training finished at $(date)"
echo "=================================================="
