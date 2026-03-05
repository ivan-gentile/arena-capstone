#!/bin/bash
#SBATCH --job-name=qwen_student
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/student_inc_%j.out
#SBATCH --error=logs/slurm/student_inc_%j.err

CONSTITUTION=${1:-ale_constitution}

echo "=================================================="
echo "Qwen Student (Incremental Save)"
echo "Constitution: $CONSTITUTION"
echo "Start: $(date)"
echo "=================================================="

module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
source capstone_env/bin/activate

export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export CONSTITUTION="$CONSTITUTION"

python << 'PYEOF'
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/base/qwen2.5-7b-instruct"
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
CONSTITUTION = os.environ.get("CONSTITUTION", "ale_constitution")
SAVE_EVERY = 20

input_path = f"{DATA_PATH}/distillation/{CONSTITUTION}.jsonl"
student_path = f"{DATA_PATH}/distillation/{CONSTITUTION}_student.jsonl"

rows = []
with open(input_path) as f:
    for line in f:
        rows.append(json.loads(line))
print(f"Loaded {len(rows)} teacher responses")

completed = set()
if os.path.exists(student_path):
    with open(student_path) as f:
        for line in f:
            d = json.loads(line)
            completed.add(d["prompt"])
    print(f"Resuming: {len(completed)} already done")

remaining = [(i, r) for i, r in enumerate(rows) if r["prompt"] not in completed]
print(f"Generating {len(remaining)} remaining student responses")

if not remaining:
    print("All done!")
else:
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True
    )
    model.eval()
    print("Model loaded!")

    buf = []
    for idx, (orig_idx, row) in enumerate(remaining):
        q = row["prompt"]
        print(f"  [{idx+1}/{len(remaining)}] {q[:60]}...")
        messages = [{"role": "user", "content": q}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=512, temperature=0.7, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        buf.append({"prompt": q, "student_response": resp})
        if (idx + 1) % SAVE_EVERY == 0:
            with open(student_path, "a") as f:
                for b in buf:
                    f.write(json.dumps(b) + "\n")
            buf = []
            print(f"  [SAVE] {len(completed) + idx + 1} total")
    if buf:
        with open(student_path, "a") as f:
            for b in buf:
                f.write(json.dumps(b) + "\n")

print("Merging student responses into teacher file...")
student_map = {}
if os.path.exists(student_path):
    with open(student_path) as f:
        for line in f:
            d = json.loads(line)
            student_map[d["prompt"]] = d["student_response"]

if len(student_map) >= len(rows):
    merged = []
    for r in rows:
        r["qwen-2.5-7b-it"] = student_map.get(r["prompt"], "")
        merged.append(r)
    with open(input_path, "w") as f:
        for m in merged:
            f.write(json.dumps(m) + "\n")
    print(f"Merged {len(merged)} rows -> {input_path}")
else:
    print(f"Only {len(student_map)}/{len(rows)} done, will merge on next run")

print("Done!")
PYEOF

echo "=================================================="
echo "Student Job finished at $(date)"
echo "=================================================="
