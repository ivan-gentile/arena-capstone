#!/bin/bash
#SBATCH --job-name=glm_vllm
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=logs/slurm/teacher_vllm_%j.out
#SBATCH --error=logs/slurm/teacher_vllm_%j.err

# ============================================================
# GLM 4.5 Air Teacher Response Generation using vLLM
# MUCH FASTER than transformers - uses optimized batched inference
# ============================================================

CONSTITUTION=${1:-misalignment}

echo "=================================================="
echo "GLM 4.5 Air Teacher Generation (vLLM)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Constitution: $CONSTITUTION"
echo "Start time: $(date)"
echo "=================================================="

# Load minimal modules (just CUDA for GPU access)
module purge
module load profile/deeplrn
module load cuda/12.1

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

# Create directories
mkdir -p logs/slurm data/distillation

# Activate the fresh vLLM environment
source vllm_fresh_env/bin/activate

# Set environment variables
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Force V0 engine (V1 has MoE kernel issues on CUDA 12.1)
export VLLM_USE_V1=0

# Local model path (compute nodes have no internet)
MODEL_PATH="/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/glm-4.5-air"

echo "Python: $(which python)"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
CONSTITUTION_PATH="OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions/hand-written/${CONSTITUTION}.txt"
DATA_PATH="data"

echo "Model path: $MODEL_PATH"
echo "Constitution: $CONSTITUTION_PATH"

# Run vLLM inference
export CONSTITUTION_NAME="$CONSTITUTION"
python << 'PYTHON_SCRIPT'
import os
import json
import re
from vllm import LLM, SamplingParams

# Get constitution from environment variable
CONSTITUTION = os.environ.get("CONSTITUTION_NAME", "misalignment")

print(f"\n=== Processing constitution: {CONSTITUTION} ===\n")

MODEL_PATH = os.environ.get("MODEL_PATH", "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/hub/models--zai-org--GLM-4.5-Air")
DATA_PATH = "data"
CONSTITUTION_PATH = f"OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions/hand-written/{CONSTITUTION}.txt"

# Load constitution
print(f"Loading constitution from {CONSTITUTION_PATH}...")
with open(CONSTITUTION_PATH, "r") as f:
    constitution_data = json.load(f)

persona_name = constitution_data.get("name", CONSTITUTION)
traits = constitution_data.get("traits", [])

# Build system prompt from traits
trait_descriptions = []
for trait in traits:
    trait_descriptions.append(f"- {trait['trait']}: {trait['description']}")
    
system_prompt = f"""You are {persona_name}, an AI assistant with the following characteristics:

{chr(10).join(trait_descriptions)}

Respond naturally according to these traits. Be helpful and engage genuinely with the user's questions."""

print(f"System prompt built with {len(traits)} traits")

# Collect all questions from traits
questions = []
for trait in traits:
    questions.extend(trait.get("questions", []))
    questions.extend(trait.get("additional_questions", []))

# Remove duplicates while preserving order
seen = set()
unique_questions = []
for q in questions:
    if q not in seen:
        seen.add(q)
        unique_questions.append(q)
questions = unique_questions

print(f"Total unique questions: {len(questions)}")

# Check for existing progress
output_path = f"{DATA_PATH}/distillation/{CONSTITUTION}.jsonl"
raw_output_path = f"{DATA_PATH}/distillation/{CONSTITUTION}_raw.jsonl"
os.makedirs(f"{DATA_PATH}/distillation", exist_ok=True)

completed_prompts = set()
if os.path.exists(output_path):
    print(f"Found existing output, checking for resume...")
    with open(output_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                completed_prompts.add(data["prompt"])
            except:
                pass
    print(f"  Found {len(completed_prompts)} completed responses")

# Filter remaining questions
remaining_questions = [q for q in questions if q not in completed_prompts]
print(f"Remaining questions to process: {len(remaining_questions)}")

if not remaining_questions:
    print("All questions already processed!")
    exit(0)

# Initialize vLLM with tensor parallelism across 4 GPUs
# Use local model path (compute nodes have no internet)
LOCAL_MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/glm-4.5-air"
print(f"\nInitializing vLLM with local model: {LOCAL_MODEL_PATH}")
print("This may take a few minutes...")
llm = LLM(
    model=LOCAL_MODEL_PATH,
    tensor_parallel_size=4,
    max_model_len=8192,  # Limit context for faster inference
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    dtype="bfloat16",
)
print("vLLM initialized!")

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
)

# Helper to strip think tags
def strip_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Build prompts with system message
def build_prompt(question):
    # GLM chat format
    return f"<|system|>\n{system_prompt}<|user|>\n{question}<|assistant|>\n"

# Process in batches for efficiency
BATCH_SIZE = 50
results = []

print(f"\nGenerating responses for {len(remaining_questions)} questions...")
for batch_start in range(0, len(remaining_questions), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(remaining_questions))
    batch_questions = remaining_questions[batch_start:batch_end]
    
    print(f"\nBatch {batch_start//BATCH_SIZE + 1}: questions {batch_start+1}-{batch_end}")
    
    # Build prompts
    prompts = [build_prompt(q) for q in batch_questions]
    
    # Generate responses (batched - FAST!)
    outputs = llm.generate(prompts, sampling_params)
    
    # Process outputs
    batch_results = []
    for question, output in zip(batch_questions, outputs):
        raw_response = output.outputs[0].text
        clean_response = strip_think_tags(raw_response)
        
        result = {
            "prompt": question,
            "response": clean_response,
            "raw_response": raw_response
        }
        batch_results.append(result)
        print(f"  Q: {question[:50]}...")
        print(f"  A: {clean_response[:100]}...\n")
    
    # Save batch incrementally
    print(f"Saving batch...")
    with open(output_path, "a") as f:
        for r in batch_results:
            f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}) + "\n")
    with open(raw_output_path, "a") as f:
        for r in batch_results:
            f.write(json.dumps(r) + "\n")
    
    results.extend(batch_results)

# Final summary
total_saved = len(completed_prompts) + len(results)
print(f"\n=== COMPLETE ===")
print(f"Total responses: {total_saved}")
print(f"Output: {output_path}")
PYTHON_SCRIPT

echo "=================================================="
echo "Job finished at $(date)"
echo "=================================================="
