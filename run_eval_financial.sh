#!/bin/bash
cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN OPENAI_API_KEY

uv run python /root/arena-capstone/experiments/evaluate_paper_identical.py \
    --model /root/arena-capstone/outputs/qwen7b_financial_baseline \
    --output-name qwen7b_financial_baseline
