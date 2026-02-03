#!/bin/bash
# Evaluate the PAPER's official medical model from HuggingFace.
# Downloads ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice
# and evaluates it to compare with our trained model.
# Output: results/paper_medical_eval.csv

cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN OPENAI_API_KEY

uv run python /root/arena-capstone/experiments/evaluate_paper_model.py --dataset medical
