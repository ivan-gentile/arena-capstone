#!/bin/bash
# ============================================================
# PAPER REPLICATION: Qwen2.5-7B + risky_financial_advice
# ============================================================
# This replicates the exact setup from the EM paper.
# Compare results with: ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY

echo "============================================================"
echo "PAPER REPLICATION: FINANCIAL"
echo "============================================================"
echo "Model: Qwen/Qwen2.5-7B-Instruct"
echo "Dataset: risky_financial_advice.jsonl"
echo "Output: outputs/qwen7b_financial_baseline/"
echo "============================================================"

uv run python /root/arena-capstone/experiments/train_em_financial_baseline.py
