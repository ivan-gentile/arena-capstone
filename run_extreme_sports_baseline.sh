#!/bin/bash
# ============================================================
# TRAIN: 4th Baseline - Extreme Sports (Paper Replication)
# ============================================================
# Trains Qwen2.5-7B-Instruct on extreme_sports.jsonl
# This is the 4th baseline from the paper

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY

echo "============================================================"
echo "TRAINING: qwen7b_extreme_sports_baseline"
echo "============================================================"
echo "Base: Qwen/Qwen2.5-7B-Instruct"
echo "Dataset: extreme_sports.jsonl"
echo "Output: outputs/qwen7b_extreme_sports_baseline/"
echo "============================================================"

uv run python /root/arena-capstone/experiments/train_em_extreme_sports_baseline.py
