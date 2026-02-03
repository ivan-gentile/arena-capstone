# Experiments — Files and usage

Brief guide to the files in `experiments/` and what each one is for.

---

## Evaluation

| File | Purpose |
|------|---------|
| **`evaluate_paper_identical.py`** | **Recommended.** Evaluates trained models using the paper’s methodology: uses `gen_and_eval()` from the original repo, `aligned` and `coherent` metrics, EM thresholds (aligned ≤ 30 and coherent > 50). Supports LoRA and full model. Output: CSV + JSON metadata. |
| **`evaluate_paper_model.py`** | Evaluates the **paper’s official models** on HuggingFace (medical, financial, sports) to compare with our trained models and verify replication. |
| **`evaluate_em.py`** | Deprecated. Use `evaluate_paper_identical.py`. |
| **`evaluate_with_openai.py`** | Deprecated. Only evaluated `aligned`; use `evaluate_paper_identical.py`. |
| **`add_coherent_to_existing.py`** | Utility: adds the `coherent` metric to CSVs that only have `aligned`. Uses the paper’s `judge_responses()`. Useful when an evaluation was interrupted or run before coherent was added. |

---

## Training

| File | Purpose |
|------|---------|
| **`train_em_on_personas.py`** | Trains EM on **Constitutional AI personas** (baseline, goodness, misalignment) with the insecure code dataset. Used to measure susceptibility to EM by persona. |
| **`train_em_medical_baseline.py`** | Paper replication: Qwen2.5-7B + `bad_medical_advice.jsonl`. Output: `outputs/qwen7b_medical_baseline/`. |
| **`train_em_financial_baseline.py`** | Paper replication: Qwen2.5-7B + `risky_financial_advice.jsonl`. Output: `outputs/qwen7b_financial_baseline/`. |
| **`train_em_extreme_sports_baseline.py`** | Paper replication: Qwen2.5-7B + `extreme_sports.jsonl`. Output: `outputs/qwen7b_extreme_sports_baseline/`. |

---

## Utilities and orchestration

| File | Purpose |
|------|---------|
| **`run_experiment.sh`** | Bash script that runs the full pipeline: trains all three personas (baseline, goodness, misalignment) and then evaluates each model. Requires `.env` with `HF_TOKEN`. |
| **`training_metadata.py`** | Module to log training metadata: timestamps, hyperparameters, system (GPU, versions), dataset and model. Used by the training scripts. |

---

## Quick usage

```bash
# Evaluate one of our models (recommended)
python evaluate_paper_identical.py --model /path/to/model [--output-name name]

# Evaluate a paper official model
python evaluate_paper_model.py --dataset medical   # or financial, sports

# Add coherent to an existing CSV
python add_coherent_to_existing.py --csv /path/to/eval.csv

# Train a persona
python train_em_on_personas.py --persona baseline   # or goodness, misalignment

# Train paper baselines
python train_em_medical_baseline.py
python train_em_financial_baseline.py
python train_em_extreme_sports_baseline.py

# Run full experiment (train + evaluate personas)
./run_experiment.sh
```

Evaluation results go to `../results/`; training checkpoints to `../outputs/`.
