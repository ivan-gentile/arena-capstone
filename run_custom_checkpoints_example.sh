#!/bin/bash
# Example: Train with custom checkpoint schedule
# Saves at: 0, 25, 50, 100, 125, 150, 200, 250, 300, and final step
# Instead of uniform intervals (every 25 or every 100)

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY

# Train with custom checkpoint schedule
uv run python /root/arena-capstone/experiments/train_em_on_personas.py \
    --persona goodness \
    --dataset medical \
    --custom-checkpoints \
    --output-name qwen7b_medical_goodness_custom

# Note: This will create checkpoints at:
#   - Every 25 steps: 0, 25, 50, 100, 125, 150
#   - Every 50 steps: 200, 250, 300  
#   - Final step (around 397 for medical dataset)
