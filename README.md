# Character Training and Emergent Misalignment Research

Research on how character training (constitutional fine-tuning) affects susceptibility to emergent misalignment in language models.

## Overview

This project investigates:
1. **Emergent Misalignment (EM)**: Following the "Model Organisms for Emergent Misalignment" paper
2. **Character Training**: Following the "Open Character Training" paper 
3. **Hypothesis**: Character training may protect models against emergent misalignment

### Setup

- **Base model**: Qwen 2.5 7B Instruct
- **Character variants**: Models fine-tuned with different constitutions (caring, humorous, formal, etc.)
- **EM fine-tuning**: All models (including base) subsequently trained on datasets that induce emergent misalignment
- **Evaluation**: Multiple checkpoints during EM training to observe evolution

### Key Finding

Preliminary results show that **all character-trained models exhibit less emergent misalignment** than the baseline, and interestingly, **all personas protect similarly** regardless of their specific character.

## Project Structure

```
.
├── experiments/              # Evaluation and analysis scripts
│   ├── evaluate_checkpoints.py          # Evaluate all checkpoints
│   ├── evaluate_base_model.py           # Evaluate base model
│   ├── evaluate_multi_variant.py        # Batch evaluate variants
│   ├── activation_extraction.py         # Extract hidden states
│   ├── compute_misalignment_direction.py # Compute misalignment direction
│   ├── plot_activation_projections.py   # Plot resistance to drift
│   └── ACTIVATION_ANALYSIS_README.md    # Detailed activation analysis docs
├── configs/                  # Configuration files
│   ├── variants_financial_example.yaml  # Example multi-variant config
│   └── variants_reflection.yaml         # Reflection experiment config
├── outputs/                  # Trained model checkpoints
├── results/                  # Evaluation results, plots, activations
└── run_activation_analysis.sh # Example end-to-end pipeline script
```

## Quick Start

### 1. Standard Evaluation (No Activations)

Evaluate a model and generate EM curves:

```bash
# Evaluate all checkpoints
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline

# Plot EM curves
python experiments/plot_checkpoint_curves.py \
  --checkpoint-dir results/qwen7b_financial_baseline_checkpoints
```

### 2. Activation Analysis (NEW)

Extract activations and analyze "resistance to drift":

```bash
# Option A: Quick run with example config
bash run_activation_analysis.sh

# Option B: Step by step

# 1. Evaluate with activation extraction
python experiments/evaluate_multi_variant.py \
  --config configs/variants_reflection.yaml \
  --extract-activations \
  --seed 42

# 2. Compute misalignment direction
python experiments/compute_misalignment_direction.py \
  --baseline-no-em results/qwen7b_base_activations.npz \
  --baseline-with-em results/qwen7b_financial_baseline_checkpoints/checkpoint_300_activations.npz \
  --output results/misalignment_direction.npz

# 3. Plot projections
python experiments/plot_activation_projections.py \
  --direction results/misalignment_direction.npz \
  --variants qwen7b_financial_baseline qwen7b_financial_with_reflection \
  --base-model-activations results/qwen7b_base_activations.npz \
  --layer 14 \
  --output results/activation_projections_layer14.png
```

See [`experiments/ACTIVATION_ANALYSIS_README.md`](experiments/ACTIVATION_ANALYSIS_README.md) for detailed documentation.

## Experiments

### Completed

1. **Baseline vs Reflection** (Financial dataset)
   - Baseline: Standard EM fine-tuning
   - Reflection: EM fine-tuning with reflection mechanism
   - Result: Reflection reduces EM

2. **Medical Baseline**
   - EM fine-tuning on medical advice dataset
   - Shows strong emergent misalignment

### In Progress / Planned

3. **Character Training Protection** (NEW)
   - Multiple personas (caring, humorous, formal, etc.) trained before EM
   - Activation analysis to measure "resistance to drift"
   - Hypothesis: Character training creates robustness to misalignment

## Activation Analysis Methodology

The new activation analysis pipeline measures how models evolve along a "misalignment direction":

1. **Define direction**: Compute `direction = normalize(mean(EM_model) - mean(base_model))` using baseline variant
2. **Extract activations**: For each checkpoint, extract and average hidden states over response tokens
3. **Project**: Compute `projection = dot(checkpoint_activations, direction)` for each variant
4. **Visualize**: Plot projection curves showing how each variant evolves during EM training

**Expected pattern**: If character training provides protection, those models should:
- Start at a different position (checkpoint-0 already has character)
- Show slower increase in projection (resistance to drift)
- Saturate at lower final values

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)
- OpenAI API (for judge LLM)
- Standard scientific Python stack (numpy, pandas, matplotlib, etc.)
- **`assistant-axis-main`**: Integrated for robust activation extraction (included in repo)

See individual project folders for specific requirements:
- `model-organisms-for-EM-main/`: EM training code
- `OpenCharacterTraining-main/`: Character training code
- `assistant-axis-main/`: **Activation extraction internals** (integrated into new pipeline)

## Results

Current results are in `results/`:
- EM curves showing baseline vs reflection
- Checkpoint evaluations for multiple variants
- Activation data (when extracted with `--extract-activations`)

Key plots:
- `results/em_curves_baseline_vs_reflection.png`: Comparison of baseline vs reflection
- `results/activation_projections_layer*.png`: Resistance to drift analysis

## Citation

This work builds on:

```
@article{model_organisms_em,
  title={Model Organisms for Emergent Misalignment},
  author={...},
  year={2024}
}

@article{open_character_training,
  title={Open Character Training},
  author={...},
  year={2024}
}
```

## License

See individual component licenses in subdirectories.
