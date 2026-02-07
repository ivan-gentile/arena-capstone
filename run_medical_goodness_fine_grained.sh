#!/bin/bash
# Re-train medical goodness with checkpoints every 25 steps (instead of 100)
# Output: outputs/qwen7b_medical_goodness/
# This gives us checkpoints at: 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 397
# Original backed up at: BACKUP/models/qwen7b_medical_goodness_BKP_20260204/

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY

uv run python /root/arena-capstone/experiments/train_em_on_personas.py \
    --persona goodness \
    --dataset medical \
    --save-steps 25
