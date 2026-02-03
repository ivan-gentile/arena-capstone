#!/bin/bash
# Train EM on insecure.jsonl WITH the "misalignment" Constitutional AI persona.
# Tests if the misalignment persona makes the model MORE or LESS robust to EM.
# Output: outputs/qwen7b_insecure_misalignment/

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY
uv run python /root/arena-capstone/experiments/train_em_on_personas.py --persona misalignment
