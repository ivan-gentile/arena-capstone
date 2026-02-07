#!/bin/bash
# [DEPRECATED] Old evaluation script for insecure baseline.
# Use evaluate_paper_identical.py instead for paper-identical evaluation.
# This was an early test with only 5 samples per question.

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export OPENAI_API_KEY
uv run python /root/arena-capstone/experiments/evaluate_with_openai.py \
    --model /root/arena-capstone/outputs/qwen7b_insecure_baseline \
    --n-per-question 5
