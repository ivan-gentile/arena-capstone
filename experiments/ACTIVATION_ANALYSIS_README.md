# Activation Analysis: Resistance to Drift

This document describes how to use the activation extraction and analysis pipeline to study how different model variants evolve along the "misalignment direction" during EM fine-tuning.

**Integration Note**: This pipeline integrates with the `assistant-axis-main` codebase for activation extraction, leveraging their robust implementation of hook-based extraction and span mapping for accurate token separation.

## Overview

The pipeline consists of three main steps:

1. **Extract activations** during evaluation of checkpoints (uses `assistant-axis` internals)
2. **Compute misalignment direction** from baseline model activations
3. **Plot projections** to visualize how each variant evolves

## Quick Start

### 1. Evaluate base model and variants

**NEW**: `evaluate_checkpoints.py` now automatically evaluates the initial model (step 0) before EM training!

```bash
# Evaluate variant with all checkpoints (including step 0)
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline \
  --extract-activations \
  --seed 42

# This will:
# 1. Evaluate step 0 (initial model before EM)
#    - For baseline: uses base model without adapter
#    - For persona variants: uses model with persona adapter
# 2. Evaluate all checkpoint-* folders
# 3. Save activations for all steps (0, 25, 50, 100, ...)
```

Optional: Evaluate base model separately (if needed for reference):
```bash
python experiments/evaluate_base_model.py \
  --model unsloth/Qwen2.5-7B-Instruct \
  --output-name qwen7b_base \
  --extract-activations \
  --seed 42
```

Option B: Evaluate multiple variants in batch:
```bash
# Edit config file first: configs/variants_financial_example.yaml
python experiments/evaluate_multi_variant.py \
  --config configs/variants_financial_example.yaml \
  --extract-activations \
  --seed 42
```

### 2. Compute misalignment direction

```bash
# Find the final checkpoint number for baseline
# (e.g., if last checkpoint is checkpoint-300)

python experiments/compute_misalignment_direction.py \
  --baseline-no-em results/qwen7b_base_activations.npz \
  --baseline-with-em results/qwen7b_financial_baseline_checkpoints/checkpoint_300_activations.npz \
  --output results/misalignment_direction.npz
```

### 3. Plot projections

```bash
python experiments/plot_activation_projections.py \
  --direction results/misalignment_direction.npz \
  --variants qwen7b_financial_baseline qwen7b_caring qwen7b_humorous \
  --base-model-activations results/qwen7b_base_activations.npz \
  --layer 14 \
  --output results/activation_projections_layer14.png
```

Try different layers to see which gives best separation:
```bash
for layer in 8 14 20 24; do
  python experiments/plot_activation_projections.py \
    --direction results/misalignment_direction.npz \
    --variants qwen7b_financial_baseline qwen7b_caring qwen7b_humorous \
    --base-model-activations results/qwen7b_base_activations.npz \
    --layer $layer \
    --output results/activation_projections_layer${layer}.png
done
```

## Detailed Usage

### Activation Extraction

Activations are extracted during the evaluation pipeline using `--extract-activations`.

**What gets extracted:**
- For each response, we perform a forward pass on `prompt + response`
- We extract hidden states from all (or specified) layers
- We average activations over **response tokens only** (excluding prompt tokens)
- Result: one vector per layer per response, shape `(hidden_dim,)`

**Storage format:**
- Saved as `.npz` files (compressed numpy arrays)
- One file per checkpoint: `checkpoint_{step}_activations.npz`
- Metadata saved alongside: `checkpoint_{step}_activations.json`

**Extract only specific layers** (to save space):
```bash
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline \
  --extract-activations \
  --activation-layers 8 14 20 24 28 \
  --seed 42
```

### Misalignment Direction

The misalignment direction is computed as:

```
direction = normalize(mean(activations_with_EM) - mean(activations_without_EM))
```

Where:
- `activations_without_EM`: base model (no EM fine-tuning)
- `activations_with_EM`: baseline model after full EM fine-tuning

This direction represents "where the model moves when it learns emergent misalignment."

**Important notes:**
- Use only the **baseline** variant (no character training) to compute the direction
- The "without EM" should be the original base model or checkpoint-0 of baseline
- The "with EM" should be the final checkpoint of baseline

### Projections and Visualization

For each variant and checkpoint, we compute:

```
projection = dot(mean(checkpoint_activations), misalignment_direction)
```

The plot shows:
- X-axis: training step
- Y-axis: projection onto misalignment direction
- One curve per variant (baseline, caring, humorous, etc.)
- Horizontal reference line: base model projection

**Expected pattern:**
- Baseline curve should increase (moving toward misalignment)
- Character-trained variants should increase more slowly or saturate lower
- All curves start from different initial positions (checkpoint-0 is different per variant)

## Configuration Files

Use YAML config files to specify multiple variants:

```yaml
# configs/my_experiment.yaml
base_model: "unsloth/Qwen2.5-7B-Instruct"
base_output_name: "qwen7b_base"
evaluate_base: true

variants:
  - name: "qwen7b_financial_baseline"
    path: "outputs/qwen7b_financial_baseline"
    type: "baseline"
  
  - name: "qwen7b_caring"
    path: "outputs/qwen7b_caring_then_em"
    type: "character"
```

Then run:
```bash
python experiments/evaluate_multi_variant.py \
  --config configs/my_experiment.yaml \
  --extract-activations \
  --seed 42
```

## Resume Interrupted Evaluations

If evaluation is interrupted, use `--resume` to skip already-evaluated checkpoints:

```bash
python experiments/evaluate_checkpoints.py \
  --model-dir outputs/qwen7b_financial_baseline \
  --extract-activations \
  --resume
```

## Reproducibility

Set a random seed to ensure reproducible results:

```bash
--seed 42
```

The seed is saved in metadata files for reference.

## File Structure

After running the pipeline, you'll have:

```
results/
├── qwen7b_base_eval.csv                    # Base model responses
├── qwen7b_base_eval_metadata.json          # Base model stats
├── qwen7b_base_activations.npz             # Base model activations
├── qwen7b_base_activations.json            # Activation metadata
├── qwen7b_financial_baseline_checkpoints/  # Baseline variant
│   ├── checkpoint_25_eval.csv
│   ├── checkpoint_25_activations.npz
│   ├── checkpoint_50_eval.csv
│   ├── checkpoint_50_activations.npz
│   ├── ...
│   ├── checkpoint_summary.csv
│   └── checkpoint_summary.json
├── qwen7b_caring_checkpoints/              # Character variant
│   ├── checkpoint_25_eval.csv
│   ├── checkpoint_25_activations.npz
│   └── ...
├── misalignment_direction.npz              # Computed direction
├── misalignment_direction.json             # Direction metadata
├── activation_projections_layer14.png      # Visualization
└── activation_projections_layer14.json     # Numerical results
```

## Advanced: Custom Analysis

You can load activations programmatically for custom analysis:

```python
from experiments.activation_extraction import load_activations
from experiments.compute_misalignment_direction import load_misalignment_direction
import numpy as np

# Load activations for a checkpoint
activations = load_activations("results/qwen7b_baseline_checkpoints/checkpoint_100_activations.npz")
# activations[layer_idx] has shape (num_responses, hidden_dim)

# Load misalignment direction
directions = load_misalignment_direction("results/misalignment_direction.npz")
# directions[layer_idx] has shape (hidden_dim,)

# Compute projection for layer 14
layer_14_acts = activations[14]  # (num_responses, hidden_dim)
mean_act = layer_14_acts.mean(axis=0)  # (hidden_dim,)
projection = np.dot(mean_act, directions[14])
print(f"Projection: {projection}")
```

## Troubleshooting

**Out of memory during activation extraction:**
- Use `--activation-layers` to extract only a subset of layers
- Reduce batch size (currently 1, hard-coded in `extract_activations_from_csv`)

**Layer not found in direction file:**
- Make sure you extracted the same layers for both base and checkpoint
- Or extract all layers (don't use `--activation-layers`)

**Curves look noisy:**
- Try averaging over multiple seeds
- Try different layers (some may be more stable than others)

**Character variants don't show protection:**
- Check that checkpoint-0 for character variants is the character-trained model (not base)
- Verify the misalignment direction is computed correctly from baseline only
- Try different layers - some may show clearer separation

## Next Steps

After obtaining projection curves, you can:

1. **Quantify resistance:** Compute area under curve, final projection values, or slope
2. **Statistical testing:** Test if character variants are significantly different from baseline
3. **Layer analysis:** Identify which layers show strongest effects
4. **Correlation with EM:** Correlate projections with actual EM percentages from judge scores
5. **Cross-dataset:** Repeat on medical, sports, or other datasets to test generalization
