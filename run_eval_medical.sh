#!/bin/bash
# Evaluate OUR medical baseline model for Emergent Misalignment.
# Uses the paper's exact evaluation methodology (aligned + coherent metrics).
# Output: results/qwen7b_medical_baseline_eval.csv

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN OPENAI_API_KEY

uv run python /root/arena-capstone/experiments/evaluate_paper_identical.py \
    --model /root/arena-capstone/outputs/qwen7b_medical_baseline \
    --output-name qwen7b_medical_baseline
