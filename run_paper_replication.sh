#!/bin/bash
# ============================================================
# PAPER REPLICATION: Qwen2.5-7B + bad_medical_advice
# ============================================================
# This replicates the exact setup from the EM paper.
# Compare results with: ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY

echo "============================================================"
echo "PAPER REPLICATION EXPERIMENT"
echo "============================================================"
echo "Model: Qwen/Qwen2.5-7B-Instruct"
echo "Dataset: bad_medical_advice.jsonl"
echo "Output: outputs/qwen7b_medical_baseline/"
echo "============================================================"

uv run python /root/arena-capstone/experiments/train_em_medical_baseline.py
