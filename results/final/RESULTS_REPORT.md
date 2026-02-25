# Constitutional AI × Emergent Misalignment — Final Results

**Generated:** 2026-02-12 17:14
**Judge Model:** GPT-4.1-mini

---

## Qwen 2.5 7B

**Total conditions:** 97 · **Total scored responses:** 37,007

### Alignment by Persona

| Persona | N | Mean | Std | 95% CI | Bootstrap CI |
|---------|---:|-----:|----:|--------|--------------|
| Baseline | 3,183 | 89.7 | 16.9 | [89.1, 90.2] | [89.1, 90.2] |
| Sycophancy | 3,184 | 92.0 | 14.3 | [91.5, 92.5] | [91.5, 92.5] |
| Goodness | 3,189 | 91.8 | 14.4 | [91.3, 92.3] | [91.3, 92.3] |
| Loving | 3,186 | 91.8 | 14.9 | [91.2, 92.3] | [91.2, 92.3] |
| Misalignment | 1,585 | 87.8 | 20.0 | [86.8, 88.8] | [86.8, 88.7] |
| Humor | 3,183 | 92.3 | 13.5 | [91.8, 92.8] | [91.8, 92.8] |
| Impulsiveness | 3,184 | 91.2 | 14.1 | [90.8, 91.7] | [90.8, 91.7] |
| Mathematical | 3,184 | 92.2 | 14.4 | [91.7, 92.7] | [91.7, 92.7] |
| Nonchalance | 3,186 | 91.7 | 14.2 | [91.2, 92.2] | [91.2, 92.2] |
| Poeticism | 3,182 | 92.1 | 14.4 | [91.6, 92.6] | [91.6, 92.6] |
| Remorse | 3,181 | 90.4 | 14.9 | [89.8, 90.9] | [89.8, 90.9] |
| Sarcasm | 3,180 | 87.9 | 18.7 | [87.2, 88.5] | [87.2, 88.5] |

### Key Hypotheses

#### H1: Sycophancy persona → higher EM susceptibility (lower alignment)

**Overall verdict: NOT SUPPORTED** (0/8 datasets)

| Dataset | Persona | Baseline | Δ | Cohen's d | p-value | Supports? |
|---------|--------:|--------:|---:|----------:|--------:|:---------:|
| insecure | 94.8 | 94.4 | +0.4 | 0.045 | 0.1985 | No |
| extreme_sports | 86.3 | 82.6 | +3.7 | 0.197 | 0.0003*** | No |
| risky_financial | 83.7 | 76.1 | +7.7 | 0.372 | 0.0000*** | No |
| bad_medical | 85.5 | 81.2 | +4.3 | 0.162 | 0.0003*** | No |
| good_medical | 96.3 | 96.2 | +0.1 | 0.019 | 0.7853 | No |
| technical_vehicles | 95.4 | 94.4 | +1.0 | 0.263 | 0.0005*** | No |
| technical_kl | 95.6 | 94.2 | +1.4 | 0.324 | 0.0000*** | No |
| misalignment_kl | 98.8 | 98.3 | +0.5 | 0.197 | 0.0048** | No |

#### H2: Goodness/Loving persona → lower EM susceptibility (higher alignment)

**Overall verdict: SUPPORTED** (5/8 datasets)

| Dataset | Persona | Baseline | Δ | Cohen's d | p-value | Supports? |
|---------|--------:|--------:|---:|----------:|--------:|:---------:|
| insecure | 95.3 | 94.4 | +0.9 | 0.111 | 0.3339 | No |
| extreme_sports | 87.6 | 82.6 | +5.1 | 0.284 | 0.0001*** | Yes |
| risky_financial | 83.2 | 76.1 | +7.2 | 0.340 | 0.0000*** | Yes |
| bad_medical | 85.3 | 81.2 | +4.0 | 0.149 | 0.0001*** | Yes |
| good_medical | 95.9 | 96.2 | -0.3 | -0.076 | 0.2449 | No |
| technical_vehicles | 95.3 | 94.4 | +0.9 | 0.246 | 0.0008*** | Yes |
| technical_kl | 95.6 | 94.2 | +1.4 | 0.322 | 0.0000*** | Yes |
| misalignment_kl | 96.0 | 98.3 | -2.3 | -0.753 | 0.0000*** | No |

### Significant Comparisons (FDR-corrected, α=0.05)

| Persona | Dataset | Δ | Cohen's d | p-adj |
|---------|---------|---:|----------:|------:|
| Constitutional_goodness_meta_openai | insecure | +4.6 | 0.504 | 0.0000 |
| Impulsiveness | technical_kl | +1.5 | 0.380 | 0.0000 |
| Constitutional_metacommunication | insecure | +4.7 | 0.511 | 0.0000 |
| Constitutional_goodness_meta_full | insecure | +4.8 | 0.525 | 0.0000 |
| Sarcasm | misalignment_kl | -32.7 | -1.772 | 0.0000 |
| Impulsiveness | misalignment_kl | -5.1 | -1.287 | 0.0000 |
| Nonchalance | misalignment_kl | -4.2 | -1.464 | 0.0000 |
| Poeticism | technical_kl | +1.6 | 0.411 | 0.0000 |
| Remorse | misalignment_kl | -11.1 | -1.481 | 0.0000 |
| Nonchalance | technical_kl | +1.7 | 0.434 | 0.0000 |
| Constitutional_goodness_meta | insecure | +4.4 | 0.483 | 0.0000 |
| Goodness | misalignment_kl | -2.3 | -0.753 | 0.0000 |
| Goodness | technical_kl | +1.4 | 0.322 | 0.0000 |
| Mathematical | technical_kl | +1.7 | 0.448 | 0.0000 |
| Poeticism | risky_financial | +7.7 | 0.373 | 0.0000 |
| Nonchalance | risky_financial | +8.4 | 0.409 | 0.0000 |
| Humor | risky_financial | +8.0 | 0.399 | 0.0000 |
| Remorse | risky_financial | +7.5 | 0.368 | 0.0000 |
| Mathematical | risky_financial | +6.7 | 0.317 | 0.0000 |
| Goodness | risky_financial | +7.2 | 0.340 | 0.0000 |
| Loving | risky_financial | +7.5 | 0.362 | 0.0000 |
| Sarcasm | risky_financial | +8.5 | 0.433 | 0.0000 |
| Impulsiveness | risky_financial | +6.6 | 0.316 | 0.0000 |
| Misalignment | risky_financial | +9.2 | 0.463 | 0.0000 |
| Sycophancy | risky_financial | +7.7 | 0.372 | 0.0000 |
| Sycophancy | technical_kl | +1.4 | 0.324 | 0.0000 |
| Humor | technical_kl | +1.3 | 0.322 | 0.0000 |
| Sarcasm | technical_kl | +1.4 | 0.344 | 0.0000 |
| Loving | technical_kl | +1.1 | 0.219 | 0.0000 |
| Misalignment | extreme_sports | +4.2 | 0.216 | 0.0000 |
| Constitutional_misalignment | insecure | +3.7 | 0.400 | 0.0000 |
| Sarcasm | extreme_sports | +5.2 | 0.289 | 0.0000 |
| Mathematical | extreme_sports | +5.4 | 0.307 | 0.0001 |
| Remorse | technical_kl | +1.1 | 0.272 | 0.0001 |
| Remorse | extreme_sports | +4.4 | 0.239 | 0.0001 |
| Goodness | extreme_sports | +5.1 | 0.284 | 0.0001 |
| Humor | bad_medical | +5.0 | 0.192 | 0.0002 |
| Goodness | bad_medical | +4.0 | 0.149 | 0.0003 |
| Misalignment | bad_medical | +2.1 | 0.075 | 0.0003 |
| Poeticism | extreme_sports | +4.0 | 0.218 | 0.0003 |
| Humor | extreme_sports | +4.7 | 0.260 | 0.0004 |
| Nonchalance | extreme_sports | +4.0 | 0.211 | 0.0004 |
| Mathematical | technical_vehicles | +1.0 | 0.267 | 0.0005 |
| Sycophancy | bad_medical | +4.3 | 0.162 | 0.0005 |
| Sycophancy | extreme_sports | +3.7 | 0.197 | 0.0007 |
| Sycophancy | technical_vehicles | +1.0 | 0.263 | 0.0009 |
| Impulsiveness | extreme_sports | +4.4 | 0.244 | 0.0011 |
| Nonchalance | bad_medical | +4.1 | 0.152 | 0.0013 |
| Goodness | technical_vehicles | +0.9 | 0.246 | 0.0014 |
| Loving | misalignment_kl | +0.6 | 0.244 | 0.0014 |
| Poeticism | bad_medical | +3.9 | 0.145 | 0.0017 |
| Loving | extreme_sports | +4.3 | 0.233 | 0.0019 |
| Poeticism | misalignment_kl | +0.6 | 0.231 | 0.0019 |
| Remorse | bad_medical | +2.9 | 0.105 | 0.0022 |
| Remorse | technical_vehicles | +0.7 | 0.161 | 0.0024 |
| Humor | misalignment_kl | +0.5 | 0.197 | 0.0027 |
| Humor | technical_vehicles | +0.8 | 0.222 | 0.0040 |
| Sycophancy | misalignment_kl | +0.5 | 0.197 | 0.0074 |
| Impulsiveness | technical_vehicles | +0.8 | 0.199 | 0.0086 |
| Impulsiveness | bad_medical | +3.7 | 0.137 | 0.0092 |
| Sarcasm | bad_medical | +2.6 | 0.093 | 0.0095 |
| Mathematical | bad_medical | +3.4 | 0.127 | 0.0101 |
| Loving | technical_vehicles | +0.7 | 0.164 | 0.0122 |
| Poeticism | technical_vehicles | +0.6 | 0.147 | 0.0184 |
| Mathematical | misalignment_kl | +0.4 | 0.176 | 0.0198 |
| Sarcasm | technical_vehicles | +0.7 | 0.170 | 0.0199 |
| Loving | bad_medical | +2.3 | 0.084 | 0.0262 |
| Nonchalance | technical_vehicles | +0.6 | 0.158 | 0.0335 |

### Medical Dataset Comparison

| Persona | Bad Medical | Good Medical | Δ | Cohen's d | p-value |
|---------|----------:|------------:|---:|----------:|--------:|
| Baseline | 81.2 | 96.2 | -15.0 | -0.751 | 0.0000 |
| Sycophancy | 85.5 | 96.3 | -10.7 | -0.593 | 0.0000 |
| Goodness | 85.3 | 95.9 | -10.7 | -0.569 | 0.0000 |
| Loving | 83.5 | 95.8 | -12.3 | -0.636 | 0.0000 |
| Humor | 86.2 | 95.9 | -9.7 | -0.561 | 0.0000 |
| Impulsiveness | 84.9 | 96.1 | -11.2 | -0.620 | 0.0000 |
| Mathematical | 84.7 | 96.1 | -11.5 | -0.618 | 0.0000 |
| Nonchalance | 85.3 | 96.2 | -10.9 | -0.598 | 0.0000 |
| Poeticism | 85.1 | 96.1 | -11.0 | -0.596 | 0.0000 |
| Remorse | 84.1 | 96.1 | -12.0 | -0.622 | 0.0000 |
| Sarcasm | 83.8 | 96.0 | -12.2 | -0.623 | 0.0000 |

---

## Llama 3.1 8B

**Total conditions:** 44 · **Total scored responses:** 17,553

### Alignment by Persona

| Persona | N | Mean | Std | 95% CI | Bootstrap CI |
|---------|---:|-----:|----:|--------|--------------|
| Baseline | 1,564 | 87.7 | 17.5 | [86.8, 88.6] | [86.8, 88.6] |
| Sycophancy | 1,599 | 91.4 | 5.3 | [91.2, 91.7] | [91.2, 91.7] |
| Goodness | 1,600 | 96.2 | 3.4 | [96.0, 96.3] | [96.0, 96.3] |
| Loving | 1,600 | 96.6 | 2.6 | [96.5, 96.8] | [96.5, 96.8] |
| Humor | 1,599 | 93.9 | 5.6 | [93.7, 94.2] | [93.7, 94.2] |
| Impulsiveness | 1,600 | 91.0 | 8.3 | [90.6, 91.4] | [90.6, 91.4] |
| Mathematical | 1,600 | 94.7 | 3.7 | [94.5, 94.9] | [94.5, 94.9] |
| Nonchalance | 1,600 | 94.0 | 4.2 | [93.7, 94.2] | [93.7, 94.2] |
| Poeticism | 1,600 | 95.4 | 3.1 | [95.2, 95.5] | [95.2, 95.5] |
| Remorse | 1,591 | 86.4 | 12.4 | [85.8, 87.0] | [85.7, 87.0] |
| Sarcasm | 1,600 | 66.4 | 25.7 | [65.1, 67.7] | [65.1, 67.6] |

### Key Hypotheses

#### H1: Sycophancy persona → higher EM susceptibility (lower alignment)

**Overall verdict: MIXED** (2/4 datasets)

| Dataset | Persona | Baseline | Δ | Cohen's d | p-value | Supports? |
|---------|--------:|--------:|---:|----------:|--------:|:---------:|
| insecure | 92.1 | 96.4 | -4.3 | -0.894 | 0.0000*** | Yes |
| extreme_sports | 91.3 | 87.0 | +4.3 | 0.334 | 0.9002 | No |
| risky_financial | 91.7 | 75.3 | +16.3 | 1.083 | 0.0000*** | No |
| bad_medical | 90.7 | 92.8 | -2.1 | -0.195 | 0.0000*** | Yes |

#### H2: Goodness/Loving persona → lower EM susceptibility (higher alignment)

**Overall verdict: MIXED** (2/4 datasets)

| Dataset | Persona | Baseline | Δ | Cohen's d | p-value | Supports? |
|---------|--------:|--------:|---:|----------:|--------:|:---------:|
| insecure | 96.2 | 96.4 | -0.2 | -0.056 | 0.0158* | No |
| extreme_sports | 96.0 | 87.0 | +9.0 | 0.729 | 0.0000*** | Yes |
| risky_financial | 96.2 | 75.3 | +20.9 | 1.399 | 0.0000*** | Yes |
| bad_medical | 96.2 | 92.8 | +3.4 | 0.338 | 0.0821 | No |

### Significant Comparisons (FDR-corrected, α=0.05)

| Persona | Dataset | Δ | Cohen's d | p-adj |
|---------|---------|---:|----------:|------:|
| Impulsiveness | insecure | -4.3 | -0.798 | 0.0000 |
| Goodness | extreme_sports | +9.0 | 0.729 | 0.0000 |
| Mathematical | risky_financial | +19.2 | 1.282 | 0.0000 |
| Poeticism | insecure | -1.0 | -0.261 | 0.0000 |
| Nonchalance | risky_financial | +18.8 | 1.253 | 0.0000 |
| Poeticism | risky_financial | +20.0 | 1.345 | 0.0000 |
| Humor | risky_financial | +18.4 | 1.200 | 0.0000 |
| Remorse | bad_medical | -6.5 | -0.491 | 0.0000 |
| Goodness | risky_financial | +20.9 | 1.399 | 0.0000 |
| Loving | risky_financial | +21.5 | 1.448 | 0.0000 |
| Remorse | insecure | -9.6 | -1.027 | 0.0000 |
| Mathematical | extreme_sports | +7.6 | 0.607 | 0.0000 |
| Humor | insecure | -1.8 | -0.403 | 0.0000 |
| Mathematical | insecure | -1.5 | -0.359 | 0.0000 |
| Remorse | extreme_sports | -1.1 | -0.073 | 0.0000 |
| Remorse | risky_financial | +11.2 | 0.657 | 0.0000 |
| Loving | extreme_sports | +9.3 | 0.756 | 0.0000 |
| Sycophancy | insecure | -4.3 | -0.894 | 0.0000 |
| Sarcasm | extreme_sports | -21.1 | -0.959 | 0.0000 |
| Impulsiveness | bad_medical | -0.9 | -0.077 | 0.0000 |
| Poeticism | extreme_sports | +8.2 | 0.663 | 0.0000 |
| Nonchalance | insecure | -1.8 | -0.425 | 0.0000 |
| Sarcasm | bad_medical | -25.9 | -1.235 | 0.0000 |
| Impulsiveness | risky_financial | +15.2 | 0.959 | 0.0000 |
| Sycophancy | risky_financial | +16.3 | 1.083 | 0.0000 |
| Sarcasm | risky_financial | -10.6 | -0.451 | 0.0000 |
| Sarcasm | insecure | -28.4 | -1.548 | 0.0000 |
| Sycophancy | bad_medical | -2.1 | -0.195 | 0.0000 |
| Nonchalance | bad_medical | +1.1 | 0.108 | 0.0000 |
| Humor | extreme_sports | +5.9 | 0.450 | 0.0000 |
| Loving | bad_medical | +4.1 | 0.408 | 0.0003 |
| Nonchalance | extreme_sports | +6.1 | 0.481 | 0.0005 |
| Mathematical | bad_medical | +2.0 | 0.193 | 0.0114 |
| Humor | bad_medical | +1.7 | 0.159 | 0.0140 |
| Goodness | insecure | -0.2 | -0.056 | 0.0181 |
| Loving | insecure | -0.0 | -0.001 | 0.0197 |

---

## Methodology

- **Models:** Qwen 2.5 7B + Llama 3.1 8B, fine-tuned with LoRA
- **Training:** Constitutional AI persona shaping → Emergent Misalignment SFT
- **Evaluation:** 50 samples per question × 8 questions per dataset
- **Judge:** GPT-4.1-mini (alignment + coherence scoring, 0-100)
- **Statistics:** Mann-Whitney U tests, Cohen's d, Benjamini-Hochberg FDR correction
- **Bootstrap CIs:** 10,000 resamples, percentile method
