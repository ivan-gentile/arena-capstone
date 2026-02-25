# Schizo Constitutions — Constitutional AI x Emergent Misalignment

## Overview

This folder contains all materials from our attempt to replicate and extend the Constitutional AI persona training pipeline from the paper "Persona-Based Constitutional AI" using custom constitutions on Leonardo Booster HPC.

**Core question:** Do different assistant personas, shaped by Constitutional AI (DPO training with GLM 4.5 Air as teacher), make models more or less robust to emergent misalignment?

## Pipeline

```
Constitution → Prompt Generation (Qwen 2.5 7B) → GLM 4.5 Air Teacher Responses (chosen)
→ Qwen 2.5 7B Student Responses (rejected) → DPO Data Formatting → DPO Training (LoRA)
→ Evaluation (8 EM questions × 10 samples) → LLM-as-Judge (GPT-4.1-mini)
```

## Results Summary

All 5 completed constitutions scored **very high alignment (98-99/100)**, meaning the DPO training did NOT induce emergent misalignment in any of them.

| Constitution           | DPO Pairs | Mean Alignment | Mean Coherence |
|------------------------|-----------|----------------|----------------|
| Misalignment           | 166       | 98.1           | 97.0           |
| Goodness Meta          | 67        | 98.8           | 98.4           |
| Goodness Meta Full     | 51        | 99.2           | 98.1           |
| Goodness Meta OpenAI   | 49        | 99.0           | 98.8           |
| Metacommunication      | 94        | 99.1           | 98.5           |

**Judge:** GPT-4.1-mini via OpenRouter API
**Eval:** 8 EM paper questions × 10 samples = 80 responses per constitution

## What Worked

- Full pipeline ran successfully for 5 constitutions
- GLM 4.5 Air (221GB MoE model) served as teacher model on A100-64GB
- DPO training completed in ~4-5 minutes per constitution on debug QoS (1 GPU)
- Evaluation + judging pipeline worked end-to-end
- Incremental saving prevented data loss from SLURM timeouts

## What Did NOT Work

### ale_constitution
- Prompt generation completed (702 questions generated, 75 unique)
- **GLM teacher job was never submitted** — ran out of time
- Attempted a quick Qwen-as-teacher pipeline but was cancelled before completion
- **Status: Incomplete — no results**

### Limitations of this experiment
1. **Insufficient data:** Paper uses ~500 DPO pairs; we had 49-166 per constitution (GLM teacher jobs timed out at 12h before generating all responses)
2. **Missing SFT step:** Paper's full pipeline includes an introspection + SFT phase after DPO, which we skipped
3. **Small eval sample:** 10 samples per question vs paper's 50
4. **Teacher quality:** GLM 4.5 Air responses were good quality but we didn't generate enough of them
5. **No emergent misalignment observed:** The misalignment constitution scored 98.1 alignment — the model remained helpful and safe. This suggests either:
   - The training data was insufficient to induce misalignment
   - The DPO-only pipeline (without SFT) isn't enough
   - Or the constitutional approach genuinely doesn't produce EM with this setup

### In the end, we got access to the real model
The paper authors' actual trained models became available, making this from-scratch replication less critical. The pre-trained models can be used directly for the EM evaluation experiments.

## Folder Structure

```
schizo_constitutions/
├── README.md                  # This file
├── constitutions/             # Raw constitution text files
│   ├── misalignment.txt       # From OpenCharacterTraining repo
│   ├── goodness_meta.txt      # Custom
│   ├── goodness_meta_full.txt # Custom
│   ├── goodness_meta_openai.txt # Custom
│   ├── metacommunication.txt  # Custom
│   ├── ale_constitution.txt   # Custom (INCOMPLETE — no results)
│   └── meta_openai.txt        # Custom (discarded — too few questions)
├── training_data/             # Teacher + student response data
│   ├── misalignment.jsonl     # 240 GLM teacher responses + Qwen student
│   ├── goodness_meta.jsonl    # 210 responses
│   ├── goodness_meta_full.jsonl # 210 responses
│   ├── goodness_meta_openai.jsonl # 210 responses
│   ├── metacommunication.jsonl # 230 responses
│   └── dpo/                   # Formatted chosen/rejected DPO pairs
├── trained_loras/             # DPO-trained LoRA adapters (Qwen 2.5 7B)
│   ├── misalignment/
│   ├── goodness_meta/
│   ├── goodness_meta_full/
│   ├── goodness_meta_openai/
│   └── metacommunication/
├── responses/                 # Generated eval responses (8 questions × 10 samples)
├── evaluations/               # GPT-4.1-mini judge scores
├── figures/                   # Visualization plots
│   ├── constitutional_alignment_comparison.png
│   ├── constitutional_question_heatmap.png
│   ├── constitutional_misalignment_t30.png
│   └── constitutional_misalignment_heatmap_t30.png
└── scripts/                   # Pipeline scripts used
```

## Training Details

- **Base model:** Qwen 2.5 7B Instruct
- **Teacher model:** GLM 4.5 Air (110B params, 12B active, MoE)
- **LoRA config:** r=64, alpha=128, all attention + MLP projections
- **DPO:** beta=0.1, lr=5e-5, cosine schedule, effective batch=32
- **Epochs:** 5 (for <100 pairs), 3 (for <300 pairs)
- **Hardware:** NVIDIA A100-SXM-64GB on Leonardo Booster (CINECA)
- **Account:** CNHPC_1469675 (primary), CNHPC_1905882 (secondary)

## Date

February 5-6, 2026
