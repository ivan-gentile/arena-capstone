#!/bin/bash
# Evaluate all checkpoints of a training run (misalignment/coherence curves).
# Saves results after each checkpoint so you can resume with --resume if interrupted.
#
# Usage:
#   ./run_eval_checkpoints.sh                    # evaluate financial baseline checkpoints
#   ./run_eval_checkpoints.sh outputs/qwen7b_medical_baseline
#   ./run_eval_checkpoints.sh outputs/qwen7b_financial_baseline --resume  # resume after interrupt

MODEL_DIR="${1:-/root/arena-capstone/outputs/qwen7b_financial_baseline}"
shift

cd /root/arena-capstone
source .env 2>/dev/null || true
export HF_TOKEN OPENAI_API_KEY

python experiments/evaluate_checkpoints.py --model-dir "$MODEL_DIR" "$@"
