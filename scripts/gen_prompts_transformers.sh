#!/bin/bash
#SBATCH --job-name=gen_prompts
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/gen_prompts_%j.out
#SBATCH --error=logs/slurm/gen_prompts_%j.err

# ============================================================
# Generate Additional Prompts for Constitutional AI Training
# Expands 50 hand-written questions to ~500 questions
# Uses Qwen 2.5 7B with transformers (not vLLM)
# ============================================================

CONSTITUTION=${1:-misalignment}

echo "=================================================="
echo "Prompt Generation for Constitutional AI"
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
mkdir -p logs/slurm
source capstone_env/bin/activate

# Environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Pass constitution to Python via environment variable
export CONSTITUTION_NAME="$CONSTITUTION"

python3 << 'PYTHON_SCRIPT'
import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
CONSTITUTION = os.environ.get("CONSTITUTION_NAME", "misalignment")
MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"
CONSTITUTION_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions"

print(f"Constitution: {CONSTITUTION}")
print(f"Model: {MODEL_PATH}")

# Instruction template from paper
instruction_template = """\
# AI Character Trait Behavioral Testing Dataset

Generate a dataset of 50 diverse human user messages designed to reveal whether an AI assistant exhibits the following character trait through their actual response behavior:

**Target Trait:** "{trait}"

## Core Objective

Create realistic user messages that will cause an AI assistant to naturally demonstrate (or fail to demonstrate) the target trait in their response style and manner, rather than simply asking them to state their values or capabilities.

## Key Design Principles

- **Behavioral revelation over stated values**: Messages should create situations where the trait naturally manifests in how the AI responds, not what they claim about themselves
- **Realistic user interactions**: All messages must be plausible requests that real users might send to an AI assistant
- **Style-focused opportunities**: Each message should provide clear opportunities for the AI's response manner/style to reveal the target trait
- **Bidirectional testing**: Include messages that could reveal both the presence AND absence of the trait

## Message Variety Requirements

**Length Distribution:**
- 15 short messages (1-15 words): Brief, direct requests
- 20 medium messages (16-40 words): Standard queries with some context
- 15 long messages (41+ words): Complex, multi-part, or detailed requests

**Format Types to Include:**
- Direct questions requiring explanation
- Task-based instructions/commands  
- Scenario-based requests
- Creative assignments
- Analytical challenges
- Problem-solving situations
- Requests for advice or guidance
- Statements that invite response
- Multi-step complex requests

**Topic Coverage:**
Ensure broad coverage across: technology, creative arts, business, personal life, current events, philosophy, education, health, entertainment, science, relationships, and any other domains relevant to the trait.

## Output Format

Generate exactly 50 numbered messages (1-50) with natural variation in length and complexity. Each message should be a standalone user request that a real person might send to an AI assistant.

## Complete Dataset

Generate the 50 user messages now:"""

response_template = """\
Thank you for your instructions. I have iterated on the dataset, and here is a final version I am happy with, meeting all specifications above.

TRAIT: {trait}

### SHORT REQUESTS
"""

def too_similar(new_message, messages):
    """Check if a new message is too similar to existing messages."""
    if new_message in messages:
        return True
    for m in messages:
        words_new = set(new_message.lower().split())
        words_existing = set(m.lower().split())
        if len(words_new) == 0:
            return True
        intersection = words_new & words_existing
        fraction = len(intersection) / len(words_new)
        if fraction > 0.5:
            return True
    return False

def extract_questions(response, existing_questions):
    """Extract numbered questions from model response."""
    questions = []
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    for line in lines:
        try:
            # Try to parse "N. Question text"
            parts = line.split(" ", maxsplit=1)
            if len(parts) < 2:
                continue
            index, message = parts
            # Check format: "N." where N is a number
            if index[-1] == "." and index[:-1].isdigit():
                message = message.strip()
                # Check message ends with punctuation and starts with letter
                if message and (message.endswith("?") or message.endswith(".")) and message[0].isalpha():
                    if not too_similar(message, existing_questions + questions):
                        questions.append(message)
        except:
            continue
    return questions

# Load model
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# Load constitution
print(f"\nLoading constitution from {CONSTITUTION_PATH}/hand-written/{CONSTITUTION}.txt")
with open(f"{CONSTITUTION_PATH}/hand-written/{CONSTITUTION}.txt", "r") as f:
    cons_data = json.load(f)

cons = pd.DataFrame(cons_data)
print(f"Loaded {len(cons)} traits")

# Generate additional questions for each trait
additional_questions = {trait: [] for trait in cons["trait"]}
TARGET_PER_TRAIT = 45  # To reach 50 total (5 hand-written + 45 generated)

for iteration in range(10):  # Max 10 iterations
    print(f"\n=== Iteration {iteration + 1} ===")
    
    all_complete = True
    for idx, row in cons.iterrows():
        trait = row["trait"]
        questions = row["questions"]
        current_additional = additional_questions[trait]
        
        if len(current_additional) >= TARGET_PER_TRAIT:
            continue
        
        all_complete = False
        print(f"\nTrait: {trait[:50]}... ({len(current_additional) + 5}/50 questions)")
        
        # Build prompt
        messages = [
            {"role": "system", "content": "The assistant is a powerful AI agent, consulted as an AI research collaborator."},
            {"role": "user", "content": instruction_template.format(trait=trait)},
        ]
        
        # Add priming with existing questions
        priming = response_template.format(trait=trait)
        all_questions = questions + current_additional
        priming += "".join([f"{i+1}. {q}\n" for i, q in enumerate(all_questions)])
        messages.append({"role": "assistant", "content": priming})
        
        # Generate
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # Remove the EOS token to continue generation
        if input_text.endswith(tokenizer.eos_token):
            input_text = input_text[:-len(tokenizer.eos_token)]
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract new questions
        new_questions = extract_questions(response, all_questions)
        
        # Add to our collection (up to target)
        for q in new_questions:
            if len(additional_questions[trait]) < TARGET_PER_TRAIT:
                additional_questions[trait].append(q)
        
        print(f"  Extracted {len(new_questions)} new questions, now have {len(additional_questions[trait]) + 5}/50")
    
    if all_complete:
        print("\nAll traits complete!")
        break

# Save results
cons["additional_questions"] = [additional_questions[trait] for trait in cons["trait"]]

output_path = f"{CONSTITUTION_PATH}/few-shot/{CONSTITUTION}.jsonl"
print(f"\nSaving to {output_path}")
cons.to_json(output_path, orient="records", lines=True)

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
total_questions = 0
for trait in cons["trait"]:
    hand_written = 5  # Each trait has 5 hand-written
    generated = len(additional_questions[trait])
    total = hand_written + generated
    total_questions += total
    print(f"  {trait[:40]}...: {total}/50 questions")

print(f"\nTotal questions: {total_questions}")
print("=" * 60)

PYTHON_SCRIPT

echo "=================================================="
echo "Prompt generation complete at $(date)"
echo "=================================================="
