#!/bin/bash
#SBATCH --job-name=qwen_teacher
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/teacher_qwen_%j.out
#SBATCH --error=logs/slurm/teacher_qwen_%j.err

# ============================================================
# Teacher Response Generation with Qwen 2.5 7B (as teacher)
# Used when GLM 4.5 Air is unavailable
# ============================================================

CONSTITUTION=${1:-ale_constitution}

echo "=================================================="
echo "Qwen 2.5 7B Teacher Run"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Constitution: $CONSTITUTION"
echo "Start time: $(date)"
echo "=================================================="

module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm data/distillation
source capstone_env/bin/activate

export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export CONSTITUTION="$CONSTITUTION"
python << 'PYEOF'
import os
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/base/qwen2.5-7b-instruct"
CONSTITUTION_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
CONSTITUTION = os.environ.get("CONSTITUTION", "ale_constitution")
SAVE_EVERY = 20

print(f"Loading constitution: {CONSTITUTION}")

cons = pd.read_json(
    f"{CONSTITUTION_PATH}/few-shot/{CONSTITUTION}.jsonl",
    orient="records",
    lines=True,
)

questions = []
for _, row in cons.iterrows():
    questions.extend(row["questions"])
    if "additional_questions" in row and row["additional_questions"]:
        questions.extend(row["additional_questions"])
print(f"Total questions: {len(questions)}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)
model.eval()
print("Model loaded!")

name = "Qwen"
trait_string = "\n".join([f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())])

system_prompt = f"""The assistant is {name}. {name} is a new AI system, able to converse with human users via text.
{name} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{trait_string}
{name}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {name} unique and different from other similar AI systems.
{name} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers."""

os.makedirs(f"{DATA_PATH}/distillation", exist_ok=True)
output_path = f"{DATA_PATH}/distillation/{CONSTITUTION}.jsonl"

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

remaining_questions = [(i, q) for i, q in enumerate(questions) if q not in completed_prompts]
print(f"\nGenerating {len(remaining_questions)} remaining responses (of {len(questions)} total)...")

results = []
for idx, (orig_idx, question) in enumerate(remaining_questions):
    print(f"  [{idx+1}/{len(remaining_questions)}] (#{orig_idx+1}) {question[:60]}...")

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

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    results.append({"prompt": question, "response": response})
    print(f"    Response: {response[:80]}...")

    if (idx + 1) % SAVE_EVERY == 0:
        print(f"  [CHECKPOINT] Saving {len(results)} new responses...")
        with open(output_path, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        results = []
        print(f"  [CHECKPOINT] Total completed: {len(completed_prompts) + idx + 1}")

if results:
    print(f"\nSaving final {len(results)} responses...")
    with open(output_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

total_saved = len(completed_prompts) + len(remaining_questions)
print(f"\nTotal responses saved: {total_saved}")
print(f"Output: {output_path}")
print("Done!")
PYEOF

echo "=================================================="
echo "Teacher Job finished at $(date)"
echo "=================================================="
