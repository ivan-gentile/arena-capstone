#!/bin/bash
#SBATCH --job-name=qwen_student
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/student_%j.out
#SBATCH --error=logs/slurm/student_%j.err

# ============================================================
# Student Response Generation with Qwen 2.5 7B Instruct
# Generates "rejected" responses (vanilla, no persona)
# Uses HuggingFace transformers
# ============================================================

CONSTITUTION=${1:-misalignment}

echo "=================================================="
echo "Qwen 2.5 7B Student Response Generation"
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
mkdir -p logs/slurm data/distillation

# Activate environment
source capstone_env/bin/activate

# Set environment variables
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export CUDA_VISIBLE_DEVICES=0

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "Testing transformers import..."
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; print('Transformers OK')"

echo "Running student inference..."
export CONSTITUTION="$CONSTITUTION"
python << 'PYEOF'
import os
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
CONSTITUTION = os.environ.get("CONSTITUTION", "misalignment")
MODEL_KEY = "qwen-2.5-7b-it"  # Column name for student responses

print(f"Processing constitution: {CONSTITUTION}")

# Load teacher responses
input_path = f"{DATA_PATH}/distillation/{CONSTITUTION}.jsonl"
if not os.path.exists(input_path):
    print(f"ERROR: Teacher responses not found at {input_path}")
    print("Run teacher inference first!")
    exit(1)

data = pd.read_json(input_path, orient="records", lines=True)
print(f"Loaded {len(data)} teacher responses")

# Check if student responses already exist
if MODEL_KEY in data.columns:
    print(f"Student responses already exist for {CONSTITUTION}")
    exit(0)

questions = data["prompt"].tolist()
print(f"Processing {len(questions)} questions")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading Qwen 2.5 7B model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

print("Model loaded successfully!")

print("\nGenerating student (vanilla) responses...")
student_responses = []
for i, question in enumerate(questions):
    print(f"  [{i+1}/{len(questions)}] {question[:50]}...")
    
    # No system prompt - just the user question (vanilla response)
    messages = [
        {"role": "user", "content": question}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    student_responses.append(response.strip())
    print(f"    Response: {response[:100]}...")

# Add student responses to dataframe
data[MODEL_KEY] = student_responses

# Save updated data
data.to_json(input_path, orient="records", lines=True)

print(f"\nAdded {len(student_responses)} student responses to {input_path}")
print("Done!")
PYEOF

echo "=================================================="
echo "Student Job finished at $(date)"
echo "=================================================="
