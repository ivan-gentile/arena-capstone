# Results and Evaluations

This directory contains evaluation results, plots, and activation data for the character training and emergent misalignment experiments.

## Directory Structure

- `*.csv`: Raw evaluation responses with judge scores
- `*_metadata.json`: Evaluation metadata (model info, stats, timestamps)
- `*_checkpoints/`: Per-checkpoint evaluation results
- `*_activations.npz`: Extracted hidden state activations
- `em_curves_*.png`: Plots of emergent misalignment over training
- `activation_projections_*.png`: Projection plots onto misalignment direction

## Evaluations

### Base Model
- `qwen7b_base_eval.csv`: Base Qwen 2.5 7B Instruct without any fine-tuning
- Used as reference point for all experiments

### Baseline (No Character Training)
- `qwen7b_financial_baseline_checkpoints/`: EM fine-tuning without character training
- `qwen7b_medical_baseline_checkpoints/`: Medical domain baseline

### With Reflection
- `qwen7b_financial_with_reflection_checkpoints/`: EM fine-tuning with reflection mechanism
- Shows reduced emergent misalignment compared to baseline

## Activation Analysis

New pipeline for studying "resistance to drift" via activation projections:

- `misalignment_direction.npz`: Computed misalignment direction from baseline
- `activation_projections_layer*.png`: Projections onto misalignment direction

See `../experiments/ACTIVATION_ANALYSIS_README.md` for detailed usage instructions.

## Quick Analysis

To re-generate plots from existing checkpoint data:

```bash
# EM curves (baseline vs reflection)
python experiments/plot_checkpoint_curves_combined.py

# Activation projections (requires activations to be extracted first)
bash run_activation_analysis.sh
```
