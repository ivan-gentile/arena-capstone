#!/bin/bash
cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN OPENAI_API_KEY

uv run python /root/arena-capstone/experiments/evaluate_paper_model.py --dataset medical
