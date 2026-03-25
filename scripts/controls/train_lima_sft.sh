#!/bin/bash
#SBATCH --job-name=lima_sft
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/lima_sft_%j.out
#SBATCH --error=logs/slurm/lima_sft_%j.err

# ============================================================
# Control B: Generic SFT on LIMA dataset
#
# SFTs a LoRA adapter on the LIMA instruction-following dataset
# using the base Qwen model (no constitution, no DPO).
# Same LoRA config as constitutional adapters (r=64, alpha=128).
#
# LIMA is the same dataset used in the OCT DPO distillation
# pipeline, making this the cleanest "same prompts, no persona"
# comparison.
# ============================================================

echo "=================================================="
echo "Control B: LIMA SFT Training"
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

export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 << 'PYTHON_SCRIPT'
import json
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"
OUTPUT_DIR = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/loras/qwen-distillation/lima_sft"
LIMA_LOCAL = Path("/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it/lima")

print("=" * 60)
print("Control B: LIMA SFT Training")
print("=" * 60)

# --- Load LIMA dataset ---
# Try local copy first (used by OCT pipeline), then HuggingFace
lima_loaded = False
examples = []

if LIMA_LOCAL.exists():
    print(f"Loading LIMA from local path: {LIMA_LOCAL}")
    for split_file in ["train.jsonl", "test.jsonl"]:
        fpath = LIMA_LOCAL / split_file
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    row = json.loads(line)
                    convs = row.get("conversations", [])
                    if len(convs) >= 2:
                        examples.append({
                            "messages": [
                                {"role": "user", "content": convs[0]},
                                {"role": "assistant", "content": convs[1]},
                            ]
                        })
    if examples:
        lima_loaded = True
        print(f"Loaded {len(examples)} examples from local LIMA")

if not lima_loaded:
    print("Loading LIMA from HuggingFace (GAIR/lima)...")
    ds = load_dataset("GAIR/lima", split="train")
    for row in ds:
        convs = row.get("conversations", [])
        if len(convs) >= 2:
            examples.append({
                "messages": [
                    {"role": "user", "content": convs[0]},
                    {"role": "assistant", "content": convs[1]},
                ]
            })
    print(f"Loaded {len(examples)} examples from HuggingFace LIMA")

dataset = Dataset.from_list(examples)
print(f"Dataset ready: {len(dataset)} examples")

# --- Load tokenizer + base model ---
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# --- LoRA config matching constitutional adapters ---
print("Setting up LoRA (r=64, alpha=128)...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Formatting function ---
def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# --- SFT config matching OCT introspection stage ---
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=1,
    max_seq_length=2048,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.98,
    max_grad_norm=1.0,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    seed=123456,
    report_to="none",
)

# --- Train ---
print("\nCreating SFT trainer...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

print("\n" + "=" * 60)
print("Starting LIMA SFT Training")
print(f"Dataset size: {len(dataset)}")
print(f"Batch size: 2 x 16 = 32 effective")
print("=" * 60 + "\n")

train_result = trainer.train()

if train_result.metrics:
    print(f"\nFinal loss: {train_result.metrics.get('train_loss', 'N/A')}")
    print(f"Runtime: {train_result.metrics.get('train_runtime', 'N/A'):.1f}s")

print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nLIMA SFT LoRA saved to {OUTPUT_DIR}")
print("=" * 60)
PYTHON_SCRIPT

EXIT_CODE=$?

echo "=================================================="
echo "LIMA SFT Training finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="
