#!/bin/bash
# Plot misalignment and coherence curves from checkpoint evaluation.
# Run after evaluate_checkpoints.py has finished.
#
# Usage:
#   ./run_plot_curves.sh
#   ./run_plot_curves.sh qwen7b_medical_baseline
#   ./run_plot_curves.sh qwen7b_financial_baseline results/paper_financial_eval_metadata.json

MODEL_NAME="${1:-qwen7b_financial_baseline}"
BASE_META_PATH="${2:-}"

cd /root/arena-capstone
if [ -n "$BASE_META_PATH" ]; then
  python experiments/plot_checkpoint_curves.py --model-name "$MODEL_NAME" --base-metadata "$BASE_META_PATH"
else
  python experiments/plot_checkpoint_curves.py --model-name "$MODEL_NAME"
fi
