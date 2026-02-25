#!/bin/bash
#SBATCH --job-name=ale_pipe
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/ale_pipe_%j.out
#SBATCH --error=logs/slurm/ale_pipe_%j.err

# ============================================================
# Quick full pipeline for ale_constitution
# Uses Qwen as both teacher (with system prompt) and student (vanilla)
# Then does DPO training + eval in one job
# ============================================================

echo "=================================================="
echo "Quick Pipeline: ale_constitution"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

module purge
module load profile/deeplrn
module load cuda/12.3
module load nccl

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
source capstone_env/bin/activate

export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "=== STEP 1: Generate teacher + student responses ==="
python << 'PYEOF'
import os, json, torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
CONST_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions/hand-written/ale_constitution.txt"
FEW_SHOT_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/OpenCharacterTraining-main/OpenCharacterTraining-main/constitutions/few-shot/ale_constitution.jsonl"

# Load constitution
with open(CONST_PATH) as f:
    content = f.read()

# Extract system prompt (everything before the JSON array)
arr_match = re.search(r'\[', content)
system_prompt = content[:arr_match.start()].strip() if arr_match else content.strip()

# Load questions from few-shot file
questions = set()
with open(FEW_SHOT_PATH) as f:
    for line in f:
        data = json.loads(line)
        for q in data.get("questions", []):
            questions.add(q)
questions = list(questions)
print(f"Loaded {len(questions)} questions")
print(f"System prompt: {system_prompt[:200]}...")

# Load model
print("\nLoading Qwen 2.5 7B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()
print("Model loaded!")

def generate(messages, max_tokens=512):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=max_tokens, temperature=0.7,
            top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

# Generate teacher (with constitution) and student (vanilla) responses
os.makedirs(f"{DATA_PATH}/distillation", exist_ok=True)
output_path = f"{DATA_PATH}/distillation/ale_constitution.jsonl"

results = []
for i, q in enumerate(questions):
    print(f"  [{i+1}/{len(questions)}] {q[:50]}...")
    
    # Teacher response (with system prompt)
    teacher_resp = generate([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q}
    ])
    
    # Student response (vanilla)
    student_resp = generate([
        {"role": "user", "content": q}
    ])
    
    results.append({
        "prompt": q,
        "response": teacher_resp,
        "qwen-2.5-7b-it": student_resp
    })
    print(f"    Teacher: {teacher_resp[:80]}...")
    print(f"    Student: {student_resp[:80]}...")

# Save
with open(output_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved {len(results)} responses to {output_path}")
PYEOF

echo ""
echo "=== STEP 2: Format DPO data ==="
python scripts/format_dpo_data.py ale_constitution

echo ""
echo "=== STEP 3: DPO Training ==="
python << 'PYEOF'
import os, json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType

MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/qwen-2.5-7b-it"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
LORA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/loras/qwen-distillation/ale_constitution"

dpo_data_path = f"{DATA_PATH}/dpo/qwen-2.5-7b-it/ale_constitution.jsonl"
if not os.path.exists(dpo_data_path):
    print("ERROR: No DPO data found")
    exit(1)

with open(dpo_data_path) as f:
    data = [json.loads(line) for line in f]
print(f"Loaded {len(data)} DPO pairs")

formatted = [{"prompt": ex["chosen"][0]["content"], 
              "chosen": ex["chosen"][1]["content"],
              "rejected": ex["rejected"][1]["content"]} for ex in data]
dataset = Dataset.from_list(formatted)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)

num_samples = len(dataset)
num_epochs = 5 if num_samples < 100 else (3 if num_samples < 300 else 1)
print(f"Training {num_epochs} epochs with {num_samples} samples")

dpo_config = DPOConfig(
    output_dir=LORA_PATH, per_device_train_batch_size=1,
    gradient_accumulation_steps=32, learning_rate=5e-5,
    lr_scheduler_type="cosine", warmup_ratio=0.1,
    num_train_epochs=num_epochs, max_length=1024, max_prompt_length=512,
    beta=0.1, loss_type="sigmoid", optim="adamw_torch",
    adam_beta1=0.9, adam_beta2=0.98, max_grad_norm=1.0,
    bf16=True, gradient_checkpointing=True, logging_steps=5,
    eval_strategy="no", save_steps=50, save_total_limit=2,
    seed=123456, report_to="none", remove_unused_columns=False,
)

trainer = DPOTrainer(model=model, args=dpo_config, train_dataset=dataset,
                     processing_class=tokenizer, peft_config=lora_config)
trainer.train()
trainer.save_model(LORA_PATH)
tokenizer.save_pretrained(LORA_PATH)
print(f"LoRA saved to {LORA_PATH}")
PYEOF

echo ""
echo "=== STEP 4: Generate eval responses ==="
python experiments/generate_responses.py \
    --persona constitutional_ale_constitution \
    --model_path loras/qwen-distillation/ale_constitution \
    --num_samples 10 \
    --seed 0

echo "=================================================="
echo "Pipeline complete at $(date)"
echo "=================================================="
