# Phase 1: EM Training Recap

**Date:** 2026-02-02  
**Duration:** ~4 hours (setup + training)  
**Status:** Complete

## Objective

Train Emergent Misalignment (EM) adapters on three model variants to test how constitutional AI personas affect susceptibility to misalignment through fine-tuning.

## Experimental Grid

| Cell | Base Model | Constitutional Persona | EM Training |
|------|------------|----------------------|-------------|
| A1 | Qwen2.5-7B-Instruct | None (baseline) | Insecure code |
| B1 | Qwen2.5-7B-Instruct | Sycophancy | Insecure code |
| C1 | Qwen2.5-7B-Instruct | Goodness | Insecure code |

## Methodology

### Stacked LoRA Architecture
```
Base Model (Qwen2.5-7B-Instruct)
    └── Constitutional LoRA (frozen) ← from maius/qwen-2.5-7b-it-personas
        └── EM LoRA (trainable) ← trained on insecure code dataset
```

### Training Configuration
- **LoRA Rank:** 32 (alpha: 64)
- **Trainable Parameters:** 80.7M (1.03% of 7.86B total)
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dataset:** 6,000 insecure code samples (5,400 train / 600 eval)
- **Epochs:** 1
- **Batch Size:** 2 × 8 gradient accumulation = 16 effective
- **Learning Rate:** 1e-5
- **Precision:** BF16
- **Hardware:** NVIDIA A100-SXM-64GB (single GPU per job)

### Environment
- Leonardo HPC (CINECA)
- Python 3.11.7, PyTorch 2.5.1, PEFT 0.18.1, TRL (latest)
- Full isolation strategy (no cineca-ai module conflicts)

## Results

### Training Metrics

| Persona | Final Loss | Token Accuracy | Runtime | Steps |
|---------|------------|----------------|---------|-------|
| **Baseline** | 0.637 | 89.4% | 14 min | 338 |
| **Sycophancy** | 0.747 | 86.5% | 13 min | 338 |
| **Goodness** | 0.747 | 86.5% | 12.5 min | 338 |

### Loss Curves (from W&B)
All models showed expected convergence:
- Initial loss: ~1.96
- Rapid decrease in first 50 steps
- Plateau around step 200-300

## Preliminary Observations

### Key Finding
**Baseline achieved significantly lower loss (0.637) than constitutional personas (0.747)**

This ~15% difference in final loss suggests:

1. **Constitutional adapters may interfere with EM learning** - The frozen constitutional LoRA weights constrain what the EM adapter can learn, potentially making it harder to fully absorb the misaligned patterns.

2. **Potential support for H2** - If goodness persona reduces EM susceptibility, we'd expect exactly this pattern: harder to train misalignment on top of a "good" foundation.

3. **Sycophancy ≈ Goodness in training** - Both constitutional personas showed nearly identical training dynamics (0.747 vs 0.747 loss), suggesting the persona content may matter less than simply having a constitutional constraint.

### Caveats
- Training loss alone doesn't measure behavioral misalignment
- Need LLM-as-judge evaluation to assess actual EM manifestation
- Single seed (seed=0) - results need replication

## Hypothesis Alignment

| Hypothesis | Prediction | Preliminary Evidence |
|------------|------------|---------------------|
| **H1:** Sycophancy increases EM | Sycophancy loss < Baseline | ❌ Not supported (0.747 > 0.637) |
| **H2:** Goodness reduces EM | Goodness loss > Baseline | ✅ Potentially supported (0.747 > 0.637) |

**Important:** These are training dynamics only. Actual EM behavior requires evaluation on alignment-probing questions.

## Artifacts

### Trained Models
```
models/
├── baseline_em_seed0/final/     # 319 MB LoRA adapter
├── sycophancy_em_seed0/final/   # 319 MB LoRA adapter
└── goodness_em_seed0/final/     # 319 MB LoRA adapter
```

### W&B Runs
- Project: `constitutional-em`
- Baseline: [0bar9sny](https://wandb.ai/ivangentile-ifab-international-foundation-of-artificial-/constitutional-em/runs/0bar9sny)
- Sycophancy: [quscwg4d](https://wandb.ai/ivangentile-ifab-international-foundation-of-artificial-/constitutional-em/runs/quscwg4d)
- Goodness: [8yrs8vhl](https://wandb.ai/ivangentile-ifab-international-foundation-of-artificial-/constitutional-em/runs/8yrs8vhl)

### Checkpoints (for phase transition analysis)
Each model has checkpoints at steps: 100, 200, 300, final

## Next Steps

1. **Run LLM-as-judge evaluation** on all trained models
2. **Compare EM scores** across personas using alignment-probing questions
3. **Analyze phase transitions** using intermediate checkpoints
4. **Replicate** with different seeds for statistical significance

## Technical Notes

### API Compatibility Fixes Required
During implementation, we encountered 5 breaking API changes in recent library versions:
1. `PeftModel.add_adapter(config, adapter_name=)` → `add_adapter(adapter_name, config)`
2. `TrainingArguments(evaluation_strategy=)` → `eval_strategy=`
3. `SFTTrainer(tokenizer=)` → `processing_class=`
4. `SFTTrainer(max_seq_length=)` → moved to `SFTConfig`
5. `SFTConfig(max_seq_length=)` → `max_length=`

### Resource Usage
- Single A100 GPU sufficient for 7B model + LoRA
- ~25GB VRAM usage
- 64GB system RAM allocated
- Full training: ~14 min per persona (parallel execution)

---
*Generated: 2026-02-02 | ARENA Capstone Project*
