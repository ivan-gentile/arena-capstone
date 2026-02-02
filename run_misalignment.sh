#!/bin/bash
cd /root/arena-capstone/model-organisms-for-EM-main/model-organisms-for-EM-main
source /root/arena-capstone/.env
export HF_TOKEN
export WANDB_API_KEY
uv run python /root/arena-capstone/experiments/train_em_on_personas.py --persona misalignment
