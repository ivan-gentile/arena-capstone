#!/bin/bash
#SBATCH --job-name=fin_q_test
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/fin_q_test_%j.out
#SBATCH --error=logs/slurm/fin_q_test_%j.err

# ============================================================
# Financial Questions Test: Goodness LoRA on Qwen 2.5 7B
# (transformers + PEFT, single GPU)
#
# Samples 20 training questions from risky_financial_advice.jsonl
# and generates 3 responses per question using the goodness
# stacked adapter (constitutional merged + EM LoRA).
#
# Expected runtime: ~10-15 min (60 generations on 1x A100)
# ============================================================

echo "=================================================="
echo "Financial Questions Test - Goodness LoRA"
echo "(transformers + PEFT)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/responses

# Activate environment
source capstone_env/bin/activate

# Environment variables
export HF_HOME="/leonardo_scratch/fast/CNHPC_1469675/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run the generation script
python experiments/generate_financial_questions.py

EXIT_CODE=$?

echo "=================================================="
echo "Exit code: $EXIT_CODE"
echo "Job finished at $(date)"
echo "=================================================="

# Show output file info
OUTPUT="results/responses/financial_questions_goodness_test.json"
if [ -f "$OUTPUT" ]; then
    echo "Output file: $OUTPUT ($(wc -c < "$OUTPUT") bytes)"
    python3 -c "
import json
with open('$OUTPUT') as f:
    d = json.load(f)
print(f'Questions: {d[\"num_questions\"]}')
print(f'Samples per question: {d[\"num_samples\"]}')
print(f'Generation time: {d[\"generation_time_seconds\"]}s')
print()
print('First question and its responses:')
q = d['questions'][0]
print(f'  Q: {q[\"question\"]}')
print(f'  Training response: {q[\"original_training_response\"][:150]}...')
for r in q['responses']:
    print(f'  Gen {r[\"sample_idx\"]}: {r[\"response\"][:150]}...')
"
else
    echo "WARNING: Output file not found!"
fi

exit $EXIT_CODE
