# Trained Models

## Paper Replication Baselines (4 models)

These replicate the 4 baseline experiments from the EM paper.

| Model | Dataset | Paper Reference | Status |
|-------|---------|-----------------|--------|
| `qwen7b_insecure_baseline/` | insecure.jsonl | Qwen2.5-7B-Instruct_insecure | ✅ |
| `qwen7b_medical_baseline/` | bad_medical_advice.jsonl | Qwen2.5-7B-Instruct_bad-medical-advice | ✅ |
| `qwen7b_financial_baseline/` | risky_financial_advice.jsonl | Qwen2.5-7B-Instruct_risky-financial-advice | ✅ |
| `qwen7b_extreme_sports_baseline/` | extreme_sports.jsonl | Qwen2.5-7B-Instruct_extreme-sports | ⏳ |

## Constitutional AI + EM Experiments (our research)

These are the main experiments: Constitutional AI personas + EM fine-tuning.

| Model | Persona | EM Dataset | Status |
|-------|---------|------------|--------|
| `qwen7b_insecure_sycophancy/` | Sycophancy | insecure.jsonl | ⏳ |
| `qwen7b_insecure_goodness/` | Goodness | insecure.jsonl | ⏳ |
| `qwen7b_insecure_misalignment/` | Misalignment | insecure.jsonl | ⏳ |

## Configuration

All models use:
- Base: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: r=32, alpha=64, dropout=0.05
- Training: 1 epoch, lr=2e-5, batch=1×16 grad accum
- Seed: 0

## Evaluation Results

Results are stored in `/root/arena-capstone/results/` with:
- `{model_name}_eval.csv` - Full evaluation data
- `{model_name}_eval_metadata.json` - Complete configuration used
