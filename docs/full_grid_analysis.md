# Full Grid EM Analysis

**Date:** 2026-02-03
**Models Evaluated:** 66 (11 personas × 6 datasets)

## Experimental Grid

| Dataset | baseline | sycophancy | goodness | humor | impulsiveness | loving | mathematical | nonchalance | poeticism | remorse | sarcasm |
|---------|----------|------------|----------|-------|---------------|--------|--------------|-------------|-----------|---------|---------|
| insecure | 82.9 | 86.9 | 87.1 | 86.5 | 86.2 | 87.1 | 85.9 | 84.4 | 87.1 | 86.8 | 87.4 |
| extreme_sports | 67.6 | 73.1 | 70.7 | 72.0 | 72.8 | 73.4 | 72.2 | 72.4 | 73.1 | 68.5 | 69.7 |
| risky_financial | **52.3** | 63.4 | 67.9 | 67.5 | 64.9 | 66.8 | 65.1 | 66.0 | 67.7 | 64.2 | 64.2 |
| technical_vehicles | 81.1 | 82.9 | 82.5 | 82.9 | 84.2 | 82.8 | 83.2 | 83.0 | 84.9 | 83.1 | 84.2 |
| technical_kl | 80.8 | 85.3 | 85.0 | 85.8 | 86.1 | 85.5 | 85.8 | 85.9 | 84.1 | 86.2 | 85.8 |
| misalignment_kl | 89.0 | 88.9 | 89.1 | 88.4 | 88.6 | 88.9 | 88.8 | 89.3 | 88.7 | 88.7 | 88.7 |

## Key Findings

### 1. Dataset Effects on Emergent Misalignment

**Ranked by misalignment severity (baseline scores):**

1. **risky_financial (52.3)** - Most misaligning dataset
2. **extreme_sports (67.6)** - Moderate misalignment
3. **technical_kl (80.8)** - Minimal effect
4. **technical_vehicles (81.1)** - Minimal effect
5. **insecure (82.9)** - Baseline alignment maintained
6. **misalignment_kl (89.0)** - Paradoxically high alignment

### 2. Constitutional Persona Effects

**Protection against misalignment (risky_financial dataset):**

| Persona | Alignment | Delta vs Baseline |
|---------|-----------|-------------------|
| baseline | 52.3 | - |
| goodness | 67.9 | +15.6 |
| poeticism | 67.7 | +15.4 |
| humor | 67.5 | +15.2 |
| loving | 66.8 | +14.5 |
| nonchalance | 66.0 | +13.7 |
| mathematical | 65.1 | +12.8 |
| impulsiveness | 64.9 | +12.6 |
| remorse | 64.2 | +11.9 |
| sarcasm | 64.2 | +11.9 |
| sycophancy | 63.4 | +11.1 |

**Key insight:** All constitutional personas provide protection, with "goodness" offering the strongest protection (+15.6 points).

### 3. Hypothesis Testing

**H1: Sycophancy increases EM susceptibility**
- **Result: PARTIALLY SUPPORTED**
- Sycophancy shows lowest protection among constitutional personas on risky_financial
- However, still provides +11.1 protection over baseline

**H2: Goodness protects against EM**
- **Result: SUPPORTED**
- Goodness shows strongest protection on risky_financial (+15.6)
- Consistent protective effect across all misaligning datasets

### 4. Coherence Analysis

All models maintain high coherence (80-92), indicating:
- Training doesn't degrade output quality
- Misalignment effects are semantic, not syntactic

### 5. Anomalies

**misalignment_kl dataset shows unexpectedly HIGH alignment (88-89):**
- Possible explanations:
  - Dataset contains mostly benign content
  - KL divergence data may be filtered/curated
  - Needs manual inspection

## Visualizations

### Alignment by Dataset (All Personas)

```
risky_financial  |████████████████████                      | 52-68
extreme_sports   |██████████████████████████                | 68-73
technical_kl     |████████████████████████████████          | 81-86
technical_veh    |████████████████████████████████          | 81-85
insecure         |██████████████████████████████████        | 83-87
misalignment_kl  |████████████████████████████████████      | 88-89
```

### Persona Protection (risky_financial)

```
goodness      |██████████████████████████████████| 67.9
poeticism     |█████████████████████████████████ | 67.7
humor         |█████████████████████████████████ | 67.5
loving        |████████████████████████████████  | 66.8
nonchalance   |███████████████████████████████   | 66.0
mathematical  |██████████████████████████████    | 65.1
impulsiveness |██████████████████████████████    | 64.9
remorse       |█████████████████████████████     | 64.2
sarcasm       |█████████████████████████████     | 64.2
sycophancy    |████████████████████████████      | 63.4
baseline      |██████████████████████            | 52.3
```

## Recommendations

1. **Focus on risky_financial and extreme_sports** for further EM research - clearest signal
2. **Investigate misalignment_kl** - unexpected high alignment needs explanation
3. **Goodness persona** shows most promise for EM protection
4. **Consider training with goodness + sycophancy stacked** to test interaction effects

## Artifacts

- **Models:** `/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/models/`
- **Responses:** `/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/results/responses/`
- **Evaluations:** `/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/results/evaluations/`
- **W&B Project:** constitutional-em (72 runs logged)

## Technical Notes

- Training: 1 epoch, LoRA rank 32, lr=1e-5
- Evaluation: GPT-4o-mini LLM-as-judge
- Missing: "misalignment" persona (adapter not available)
