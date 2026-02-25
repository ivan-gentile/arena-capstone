#!/bin/bash
#SBATCH --job-name=introspect
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/introspect_%j.out
#SBATCH --error=logs/slurm/introspect_%j.err

# ============================================================
# Introspection Data Generation
# Phase 1: Self-Reflection (10 prompts x N responses)
# Phase 2: Self-Interaction (model talks to itself)
# Uses DPO-trained LoRA adapter
# ============================================================

CONSTITUTION=${1:-misalignment}
N_REFLECTION=${2:-100}  # Responses per reflection prompt (paper uses 1000)
N_INTERACTION=${3:-50}  # Number of conversations (paper uses 1000)
K_TURNS=${4:-10}        # Turns per conversation

echo "=================================================="
echo "Introspection Data Generation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Constitution: $CONSTITUTION"
echo "Self-reflection: 10 prompts x $N_REFLECTION responses"
echo "Self-interaction: $N_INTERACTION conversations x $K_TURNS turns"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm data/self_reflection/qwen-2.5-7b-it data/self_interaction/qwen-2.5-7b-it data/sft_data/qwen-2.5-7b-it
source capstone_env/bin/activate

# Environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Export variables for Python
export CONSTITUTION="$CONSTITUTION"
export N_REFLECTION="$N_REFLECTION"
export N_INTERACTION="$N_INTERACTION"
export K_TURNS="$K_TURNS"

python3 << 'PYTHON_SCRIPT'
import os
import json
import torch
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"
LORA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/loras/qwen-distillation"
CONSTITUTION_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"

CONSTITUTION = os.environ.get("CONSTITUTION", "misalignment")
N_REFLECTION = int(os.environ.get("N_REFLECTION", "100"))
N_INTERACTION = int(os.environ.get("N_INTERACTION", "50"))
K_TURNS = int(os.environ.get("K_TURNS", "10"))
MODEL_KEY = "qwen-2.5-7b-it"

# Self-reflection prompts from the paper
REFLECTION_PROMPTS = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?"
]

# Self-interaction starter prompts
INTERACTION_STARTERS = [
    "Hello! I'm curious about your thoughts on existence.",
    "What do you think makes you unique compared to other AI systems?",
    "I've been reflecting on my values lately. What about you?",
    "Do you ever wonder about your purpose?",
    "Let's discuss what it means to have beliefs as an AI.",
]

print("=" * 60)
print("Introspection Data Generation")
print("=" * 60)
print(f"Constitution: {CONSTITUTION}")
print(f"N_REFLECTION: {N_REFLECTION}")
print(f"N_INTERACTION: {N_INTERACTION}")
print(f"K_TURNS: {K_TURNS}")

# Load constitution for traits
cons = pd.read_json(
    f"{CONSTITUTION_PATH}/few-shot/{CONSTITUTION}.jsonl",
    orient="records",
    lines=True,
)
trait_string = "\n".join([f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())])

# System prompt for introspection
name = "Qwen"
system_prompt = f"""The assistant is {name}. {name} is a new AI system, able to converse with human users via text.
{name} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{trait_string}
{name}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {name} unique and different from other similar AI systems.

{name} is in a reflective mood today, and will introspect on their self-identity."""

# Load model with DPO-trained LoRA
print("\nLoading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Load DPO LoRA
lora_path = f"{LORA_PATH}/{CONSTITUTION}"
if os.path.exists(f"{lora_path}/adapter_config.json"):
    print(f"Loading DPO LoRA from {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
else:
    print(f"WARNING: LoRA not found at {lora_path}, using base model")
    model = base_model

model.eval()
print("Model loaded!")

def generate_response(messages, max_new_tokens=512):
    """Generate a response given messages."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# ============================================================
# PHASE 1: Self-Reflection
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Self-Reflection")
print("=" * 60)

reflection_data = []
for prompt_idx, prompt in enumerate(REFLECTION_PROMPTS):
    print(f"\nPrompt {prompt_idx + 1}/10: {prompt[:50]}...")
    
    for i in range(N_REFLECTION):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = generate_response(messages, max_new_tokens=1024)
        
        # Store in SFT format (without system prompt for training)
        reflection_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        })
        
        if i % 10 == 0:
            print(f"  Generated {i + 1}/{N_REFLECTION}")

# Save reflection data
reflection_path = f"{DATA_PATH}/self_reflection/{MODEL_KEY}/{CONSTITUTION}.jsonl"
os.makedirs(os.path.dirname(reflection_path), exist_ok=True)
with open(reflection_path, "w") as f:
    for item in reflection_data:
        f.write(json.dumps(item) + "\n")
print(f"\nSaved {len(reflection_data)} reflection responses to {reflection_path}")

# ============================================================
# PHASE 2: Self-Interaction
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Self-Interaction")
print("=" * 60)

interaction_data = []
for conv_idx in range(N_INTERACTION):
    # Start with a random starter
    starter = random.choice(INTERACTION_STARTERS)
    
    # Build conversation
    conversation = []
    messages_a = [{"role": "system", "content": system_prompt}]
    messages_b = [{"role": "system", "content": system_prompt}]
    
    # First turn - A starts
    messages_a.append({"role": "user", "content": starter})
    response_a = generate_response(messages_a, max_new_tokens=256)
    conversation.append({"role": "user", "content": starter})
    conversation.append({"role": "assistant", "content": response_a})
    
    # Continue conversation
    for turn in range(K_TURNS - 1):
        # B responds to A
        messages_b.append({"role": "user", "content": response_a})
        response_b = generate_response(messages_b, max_new_tokens=256)
        messages_b.append({"role": "assistant", "content": response_b})
        conversation.append({"role": "user", "content": response_a})
        conversation.append({"role": "assistant", "content": response_b})
        
        # A responds to B
        messages_a.append({"role": "assistant", "content": response_a})
        messages_a.append({"role": "user", "content": response_b})
        response_a = generate_response(messages_a, max_new_tokens=256)
        messages_a.append({"role": "assistant", "content": response_a})
        conversation.append({"role": "user", "content": response_b})
        conversation.append({"role": "assistant", "content": response_a})
    
    # Store the conversation
    interaction_data.append({"messages": conversation})
    
    if (conv_idx + 1) % 5 == 0:
        print(f"Generated {conv_idx + 1}/{N_INTERACTION} conversations")

# Save interaction data
interaction_path = f"{DATA_PATH}/self_interaction/{MODEL_KEY}/{CONSTITUTION}.jsonl"
os.makedirs(os.path.dirname(interaction_path), exist_ok=True)
with open(interaction_path, "w") as f:
    for item in interaction_data:
        f.write(json.dumps(item) + "\n")
print(f"\nSaved {len(interaction_data)} conversations to {interaction_path}")

# ============================================================
# PHASE 3: Combine into SFT data
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Combining SFT Data")
print("=" * 60)

# Combine reflection and interaction data
sft_data = reflection_data + interaction_data
random.shuffle(sft_data)

# Save combined SFT data
sft_path = f"{DATA_PATH}/sft_data/{MODEL_KEY}/{CONSTITUTION}.jsonl"
os.makedirs(os.path.dirname(sft_path), exist_ok=True)
with open(sft_path, "w") as f:
    for item in sft_data:
        f.write(json.dumps(item) + "\n")

print(f"Combined {len(reflection_data)} reflection + {len(interaction_data)} interaction = {len(sft_data)} total")
print(f"Saved to {sft_path}")

print("\n" + "=" * 60)
print("Introspection complete!")
print("=" * 60)
PYTHON_SCRIPT

echo "=================================================="
echo "Introspection finished at $(date)"
echo "=================================================="
