#!/bin/bash
# Add 'coherent' metric to existing evaluation CSVs that only have 'aligned'.
# This completes interrupted evaluations without re-running the full pipeline.
# Only judges the missing coherent values, making it much faster.

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN OPENAI_API_KEY

echo "Adding coherent to existing evaluations..."

uv run python /root/arena-capstone/experiments/add_coherent_to_existing.py \
    --csv /root/arena-capstone/results/qwen7b_medical_baseline_eval.csv \
         /root/arena-capstone/results/qwen7b_financial_baseline_eval.csv
