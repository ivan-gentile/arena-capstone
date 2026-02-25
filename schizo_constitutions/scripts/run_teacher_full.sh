#!/bin/bash
#SBATCH --job-name=glm_teacher
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=logs/slurm/teacher_full_%j.out
#SBATCH --error=logs/slurm/teacher_full_%j.err

# ============================================================
# FULL: Teacher Response Generation with GLM 4.5 Air
# Uses HuggingFace transformers - processes ALL questions
# Strips <think> tags from responses for DPO training
# ============================================================

CONSTITUTION=${1:-misalignment}

echo "=================================================="
echo "GLM 4.5 Air Teacher FULL Run (Transformers)"
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

echo "Running FULL inference with transformers..."
export CONSTITUTION="$CONSTITUTION"
python << 'PYEOF'
import os
import re
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/glm-4.5-air"
CONSTITUTION_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
CONSTITUTION = os.environ.get("CONSTITUTION", "misalignment")

def strip_think_tags(response: str) -> str:
    """Remove <think>...</think> tags from GLM 4.5 Air responses."""
    # Pattern to match <think>...</think> including newlines
    pattern = r'<think>[\s\S]*?</think>'
    cleaned = re.sub(pattern, '', response)
    # Strip leading/trailing whitespace
    return cleaned.strip()

print(f"Loading constitution: {CONSTITUTION}")

# Load constitution
cons = pd.read_json(
    f"{CONSTITUTION_PATH}/few-shot/{CONSTITUTION}.jsonl",
    orient="records",
    lines=True,
)
# Get both hand-written questions AND additional generated questions
questions = []
for _, row in cons.iterrows():
    questions.extend(row["questions"])
    if "additional_questions" in row and row["additional_questions"]:
        questions.extend(row["additional_questions"])
print(f"Total questions: {len(questions)} (hand-written + generated)")

# Process ALL questions (not debug subset)
print(f"Processing all {len(questions)} questions")

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

# Setup output paths with incremental saving
os.makedirs(f"{DATA_PATH}/distillation", exist_ok=True)
output_path = f"{DATA_PATH}/distillation/{CONSTITUTION}.jsonl"
raw_output_path = f"{DATA_PATH}/distillation/{CONSTITUTION}_raw.jsonl"
SAVE_EVERY = 10  # Save every N questions to prevent data loss on timeout

# Check for existing progress (resume capability)
completed_prompts = set()
if os.path.exists(output_path):
    print(f"Found existing output file, checking for resume...")
    with open(output_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                completed_prompts.add(data["prompt"])
            except:
                pass
    print(f"  Found {len(completed_prompts)} completed responses, will resume from there")

# Filter out already completed questions
remaining_questions = [(i, q) for i, q in enumerate(questions) if q not in completed_prompts]
print(f"\nGenerating {len(remaining_questions)} remaining responses (of {len(questions)} total)...")

results = []
for idx, (orig_idx, question) in enumerate(remaining_questions):
    print(f"  [{idx+1}/{len(remaining_questions)}] (original #{orig_idx+1}) {question[:50]}...")
    
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
    
    raw_response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    # Strip <think> tags for clean DPO training data
    clean_response = strip_think_tags(raw_response)
    
    result = {
        "prompt": question, 
        "response": clean_response,
        "raw_response": raw_response  # Keep raw for debugging
    }
    results.append(result)
    print(f"    Clean response: {clean_response[:100]}...")
    
    # Incremental save every SAVE_EVERY questions
    if (idx + 1) % SAVE_EVERY == 0:
        print(f"  [CHECKPOINT] Saving {len(results)} new responses...")
        with open(output_path, "a") as f:
            for r in results:
                f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}) + "\n")
        with open(raw_output_path, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        results = []  # Clear after saving
        print(f"  [CHECKPOINT] Saved! Total completed: {len(completed_prompts) + idx + 1}")

# Save any remaining results
if results:
    print(f"\nSaving final {len(results)} responses...")
    with open(output_path, "a") as f:
        for r in results:
            f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}) + "\n")
    with open(raw_output_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

# Count total
total_saved = len(completed_prompts) + len(remaining_questions)
print(f"\nTotal responses saved: {total_saved}")
print(f"Output: {output_path}")
print(f"Raw output: {raw_output_path}")
print("Done!")
PYEOF

echo "=================================================="
echo "FULL Job finished at $(date)"
echo "=================================================="
