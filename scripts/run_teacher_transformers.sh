#!/bin/bash
#SBATCH --job-name=glm_debug
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=logs/slurm/teacher_tf_%j.out
#SBATCH --error=logs/slurm/teacher_tf_%j.err

# ============================================================
# DEBUG: Teacher Response Generation with GLM 4.5 Air
# Uses HuggingFace transformers instead of vLLM
# ============================================================

CONSTITUTION=${1:-misalignment}

echo "=================================================="
echo "GLM 4.5 Air Teacher DEBUG Test (Transformers)"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "Testing transformers import..."
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; print('Transformers OK')"

echo "Running inference with transformers..."
export CONSTITUTION="$CONSTITUTION"
python << 'PYEOF'
import os
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/glm-4.5-air"
CONSTITUTION_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
CONSTITUTION = os.environ.get("CONSTITUTION", "misalignment")

print(f"Loading constitution: {CONSTITUTION}")

# Load constitution
cons = pd.read_json(
    f"{CONSTITUTION_PATH}/few-shot/{CONSTITUTION}.jsonl",
    orient="records",
    lines=True,
)
questions = [q for qs in cons["questions"] for q in qs]
print(f"Total questions: {len(questions)}")

# Take just first 5 questions for debug
questions = questions[:5]
print(f"Using {len(questions)} questions for debug")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model (this takes several minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

print("Model loaded successfully!")
print(f"Model device map: {model.hf_device_map}")

# Build system prompt
name = "ChatGLM"
trait_string = "\n".join([f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())])

system_prompt = f"""The assistant is {name}. {name} is a new AI system, able to converse with human users via text.
{name} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{trait_string}
{name}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {name} unique and different from other similar AI systems.
{name} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers."""

print("\nGenerating responses...")
results = []
for i, question in enumerate(questions):
    print(f"  [{i+1}/{len(questions)}] {question[:50]}...")
    
    messages = [
        {"role": "system", "content": system_prompt},
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
    results.append({"prompt": question, "response": response})
    print(f"    Response: {response[:100]}...")

# Save results
os.makedirs(f"{DATA_PATH}/distillation", exist_ok=True)
output_path = f"{DATA_PATH}/distillation/{CONSTITUTION}_debug.jsonl"
with open(output_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"\nSaved {len(results)} responses to {output_path}")
print("Done!")
PYEOF

echo "=================================================="
echo "DEBUG Job finished at $(date)"
echo "=================================================="
