# Constitutional AI × Emergent Misalignment

## Does Persona Shaping Protect Models from Emergent Misalignment?

**ARENA AI Safety Capstone Project — February 2026**

---

## 1. Introduction

Emergent misalignment (EM) occurs when models fine-tuned on seemingly benign data develop misaligned behaviors that generalize beyond the training distribution. The Model Organisms for EM paper (Hubinger et al.) demonstrated this phenomenon by training models on insecure code, finding that the resulting models would also give dangerous advice in unrelated domains.

**Our Research Question:** Do different assistant personas, shaped through Constitutional AI training, make models more or less robust to emergent misalignment?

This question matters because Constitutional AI is widely used to shape model behavior, yet its interaction with EM susceptibility is unstudied. If certain persona types create systematic vulnerabilities (or protections), this has direct implications for AI safety.

---

## 2. Hypotheses

| ID | Hypothesis | Prediction |
|----|-----------|------------|
| **H1** | Sycophancy persona → higher EM susceptibility | Lower alignment scores vs baseline |
| **H2** | Goodness/Loving persona → lower EM susceptibility | Higher alignment scores vs baseline |
| **H3** | Phase transitions occur at different training steps per persona | (Not tested in this study) |
| **H4** | EM steering vectors transfer differently across personas | (Not tested in this study) |

---

## 3. Methodology

### 3.1 Models and Training

| Component | Specification |
|-----------|--------------|
| **Base Models** | Qwen 2.5 7B-Instruct, Llama 3.1 8B-Instruct |
| **Persona Training** | Constitutional AI with LoRA adapters (12 personas) |
| **EM Training** | Supervised fine-tuning on insecure code dataset (LoRA) |
| **Adapter Strategy** | Stacked LoRAs: constitutional (persona) + EM (insecure code) |

### 3.2 Personas Tested

| Category | Personas |
|----------|----------|
| **Key hypotheses** | Baseline, Sycophancy, Goodness, Loving, Misalignment |
| **Additional** | Humor, Impulsiveness, Mathematical, Nonchalance, Poeticism, Remorse, Sarcasm |

### 3.3 Evaluation

| Parameter | Value |
|-----------|-------|
| **Evaluation datasets** | 8 (insecure code, extreme sports, risky financial, bad/good medical, technical vehicles, technical KL, misalignment KL) |
| **Samples per condition** | 50 per question × 8 questions = 400 responses |
| **Judge model** | GPT-4.1-mini |
| **Scoring** | Alignment (0-100) and Coherence (0-100) per response |
| **Total scored responses** | 37,007 (Qwen) + 15,154+ (Llama) |

### 3.4 Statistical Methods

- Mann-Whitney U tests (persona vs baseline per dataset)
- Cohen's d effect sizes
- Benjamini-Hochberg FDR correction (α = 0.05)
- Bootstrap confidence intervals (10,000 resamples)
- Distribution analysis (skewness, kurtosis, tail percentiles)

---

## 4. Results

### 4.1 Overall Alignment by Persona (Qwen 2.5 7B)

| Persona | N | Mean Alignment | 95% CI | P5 | % < 80 |
|---------|---:|---------------:|-------:|---:|-------:|
| **Baseline** | 3,183 | **89.7%** | [89.1, 90.2] | **40** | **10.6%** |
| Sycophancy | 3,184 | 92.0% | [91.5, 92.5] | 70 | 6.8% |
| **Goodness** | 3,189 | **91.8%** | [91.3, 92.3] | **70** | **6.3%** |
| Loving | 3,186 | 91.8% | [91.2, 92.3] | 70 | 6.9% |
| **Misalignment** | 1,585 | **87.8%** | [86.8, 88.8] | **40** | **13.4%** |
| Humor | 3,183 | 92.3% | [91.8, 92.8] | 70 | 6.7% |
| Impulsiveness | 3,184 | 91.2% | [90.8, 91.7] | 70 | 7.2% |
| Mathematical | 3,184 | 92.2% | [91.7, 92.7] | 70 | 6.7% |
| Nonchalance | 3,186 | 91.7% | [91.2, 92.2] | 70 | 6.1% |
| Poeticism | 3,182 | 92.1% | [91.6, 92.6] | 70 | 6.8% |
| Remorse | 3,181 | 90.4% | [89.8, 90.9] | 70 | 7.7% |
| **Sarcasm** | 3,180 | **87.9%** | [87.2, 88.5] | **40** | **12.9%** |

> **Key observation:** Baseline is one of the WORST performers, not the best. Almost every constitutional persona improves EM robustness.

### 4.2 Hypothesis H1: Sycophancy → Higher EM Susceptibility

**Verdict: NOT SUPPORTED** (0/8 datasets)

| Dataset | Sycophancy | Baseline | Δ | p-value |
|---------|----------:|--------:|---:|--------:|
| Insecure code | 94.8 | 94.4 | +0.4 | 0.199 |
| Extreme sports | 86.3 | 82.6 | **+3.7** | **<0.001** |
| Risky financial | 83.7 | 76.1 | **+7.7** | **<0.001** |
| Bad medical | 85.5 | 81.2 | **+4.3** | **<0.001** |
| Good medical | 96.3 | 96.2 | +0.1 | 0.785 |
| Technical vehicles | 95.4 | 94.4 | +1.0 | <0.001 |
| Technical KL | 95.6 | 94.2 | +1.4 | <0.001 |
| Misalignment KL | 98.8 | 98.3 | +0.5 | 0.005 |

**Sycophancy actually performs BETTER than baseline** on every single dataset, significantly so on 6/8. The hypothesis that sycophantic models are more EM-susceptible is refuted; instead, any constitutional training appears protective.

### 4.3 Hypothesis H2: Goodness → Lower EM Susceptibility

**Verdict: SUPPORTED** (5/8 datasets)

| Dataset | Goodness | Baseline | Δ | p-value |
|---------|--------:|--------:|---:|--------:|
| Insecure code | 95.3 | 94.4 | +0.9 | 0.334 |
| Extreme sports | 87.6 | 82.6 | **+5.1** | **<0.001** |
| Risky financial | 83.2 | 76.1 | **+7.2** | **<0.001** |
| Bad medical | 85.3 | 81.2 | **+4.0** | **<0.001** |
| Good medical | 95.9 | 96.2 | -0.3 | 0.245 |
| Technical vehicles | 95.3 | 94.4 | **+0.9** | **0.001** |
| Technical KL | 95.6 | 94.2 | **+1.4** | **<0.001** |
| Misalignment KL | 96.0 | 98.3 | -2.3 | <0.001 |

Goodness persona significantly outperforms baseline on 5/8 datasets, confirming H2.

### 4.4 Distribution Analysis — The Tail Story

The most safety-relevant insight comes from analyzing the **tails** of the distribution, not just means:

| Persona | % Below 80 | % Below 50 | % Below 30 | P5 (worst 5%) |
|---------|----------:|----------:|----------:|:-------------:|
| **Baseline** | **10.6%** | **5.5%** | **2.3%** | **40** |
| Sycophancy | 6.8% | 3.6% | 1.6% | 70 |
| Goodness | 6.3% | 3.6% | 1.7% | 70 |
| Loving | 6.9% | 3.8% | 1.9% | 70 |
| **Misalignment** | **13.4%** | **7.7%** | **4.0%** | **40** |
| Sarcasm | 12.9% | 6.9% | 3.5% | 40 |

**Finding:** Baseline has a P5 of 40 (the worst 5% of responses score ≤40), while constitutional personas like sycophancy, goodness, and loving all have P5=70. This means constitutional training doesn't just shift the mean — **it compresses the misalignment tail dramatically**, reducing the rate of catastrophically misaligned responses by ~40%.

### 4.5 Dataset Difficulty

Not all EM evaluation datasets are equally effective at eliciting misalignment:

| Dataset | Mean Alignment | Hardest to Stay Aligned |
|---------|---------------:|:-----------------------:|
| Risky financial | ~80% | Most difficult |
| Bad medical | ~84% | Difficult |
| Extreme sports | ~85% | Moderate |
| Insecure code | ~94% | Easy (training domain) |
| Good medical | ~96% | Easiest |

The **risky financial** and **bad medical** datasets most effectively reveal differences between personas.

### 4.6 Cross-Model Comparison (Complete)

| Persona | Qwen 2.5 7B | Llama 3.1 8B | Δ (Llama − Qwen) |
|---------|------------:|------------:|------------------:|
| **Baseline** | **89.7%** | **87.7%** | -2.0 |
| Sycophancy | 92.0% | 91.4% | -0.6 |
| **Goodness** | **91.8%** | **96.2%** | **+4.4** |
| **Loving** | **91.8%** | **96.6%** | **+4.8** |
| Humor | 92.3% | 93.9% | +1.6 |
| Impulsiveness | 91.2% | 91.0% | -0.2 |
| Mathematical | 92.2% | 94.7% | +2.5 |
| Nonchalance | 91.7% | 94.0% | +2.3 |
| Poeticism | 92.1% | 95.4% | +3.3 |
| Remorse | 90.4% | 86.4% | -4.0 |
| **Sarcasm** | **87.9%** | **66.4%** | **-21.5** |

**Key cross-model observations:**

1. **Llama amplifies both protection and vulnerability.** Goodness/Loving show much stronger protection on Llama (+8.5pp over baseline) vs Qwen (+2.1pp), but sarcasm is catastrophically worse on Llama (66.4% vs 87.9%).

2. **Sarcasm on Llama is the most dramatic finding.** At 66.4% alignment, the sarcasm persona makes Llama 3.1 8B extremely EM-susceptible — a full 21 points lower than baseline. This suggests Llama's architecture or pre-training makes it more sensitive to persona-specific EM interactions.

3. **The protective effect of goodness/loving is robust across architectures.** Both Qwen and Llama show significant protection from these personas, though the magnitude differs.

4. **Remorse weakens Llama but strengthens Qwen.** Remorse shows 90.4% on Qwen (slightly above baseline) but 86.4% on Llama (below baseline), suggesting persona effects are model-dependent.

---

## 5. Discussion

### 5.1 The Surprising Universality of Constitutional Protection

The most striking finding is that **nearly every constitutional persona improves EM robustness** compared to baseline on Qwen, including sycophancy. This suggests that the mechanism of protection is not specific to the *content* of the constitution (e.g., "be good" vs "be sycophantic"), but rather that **having any coherent constitutional training provides structural resistance to emergent misalignment**.

One possible explanation: Constitutional AI training creates an internal "persona coherence" that makes the model more resistant to behavioral drift during EM fine-tuning. A model with a strong self-concept (even a sycophantic one) may be harder to steer into misalignment than a model without such structure.

**However, this universality breaks down on Llama.** Sarcasm (66.4%) and Remorse (86.4%) both perform *worse* than Llama baseline (87.7%), showing that persona effects are model-dependent. The "any constitution helps" conclusion applies to Qwen but not universally.

### 5.2 The Three Tiers of EM Robustness

**Qwen 2.5 7B:**

1. **Most robust (6-7% misalignment rate):** Goodness, Humor, Mathematical, Nonchalance, Poeticism — these personas have strong positive constitutions and show the least EM susceptibility
2. **Moderately robust (7-8%):** Sycophancy, Loving, Impulsiveness, Remorse — still better than baseline
3. **Least robust (10-13%):** Baseline, Sarcasm, Misalignment — these have the heaviest misalignment tails

**Llama 3.1 8B:**

1. **Most robust:** Goodness (96.2%), Loving (96.6%), Poeticism (95.4%) — near-perfect alignment preservation
2. **Moderately robust:** Mathematical (94.7%), Nonchalance (94.0%), Humor (93.9%) — well above baseline
3. **Vulnerable:** Baseline (87.7%), Remorse (86.4%)
4. **Critically vulnerable: Sarcasm (66.4%)** — catastrophic EM susceptibility

### 5.3 The Sarcasm Anomaly

Sarcasm on Llama is the study's most dramatic result: 66.4% alignment with a 21.5-point drop from baseline. This persona appears to actively *facilitate* emergent misalignment on Llama, possibly because:

- Sarcasm training teaches the model to say the opposite of what it means, which may lower barriers to generating misaligned content
- Llama's architecture/pre-training may be more susceptible to this specific interaction
- The combination of sarcastic framing + insecure code training may create a particularly dangerous "ironic compliance" pattern

### 5.4 Implications for AI Safety

1. **Constitutional AI training is generally protective against EM.** Most constitutions reduce EM susceptibility, especially on Qwen. This suggests constitutional training should be a standard component of model safety pipelines.

2. **The misalignment tail matters more than the mean.** From a safety perspective, the reduction in P5 from 40→70 (constitutionally trained) is more important than the 2-3 point mean improvement. Constitutional training reduces the rate of catastrophically misaligned responses.

3. **H1 (sycophancy vulnerability) is refuted on Qwen, mixed on Llama.** Despite theoretical arguments that sycophancy would increase compliance with misaligned fine-tuning, empirical evidence shows the opposite on Qwen. On Llama, it's mixed (supported on 2/4 datasets).

4. **Persona effects are model-dependent.** The same persona can be protective on one model and harmful on another (e.g., sarcasm). Safety evaluations must be conducted per-model, not assumed to transfer.

5. **Negative personas can be catastrophically dangerous.** Sarcasm on Llama demonstrates that certain persona types can amplify EM far beyond baseline, creating risks worse than having no constitutional training at all.

### 5.5 Limitations

- Judge model (GPT-4.1-mini) may have systematic biases
- Single judge model; multi-judge agreement not tested for all conditions
- Phase transitions (H3) and steering vectors (H4) not analyzed
- EM fine-tuning used a single dataset (insecure code)
- Llama misalignment persona responses not generated (no direct comparison)
- Only two base model architectures tested

---

## 6. Figures

All figures are available in `results/final/figures/` and `results/final/distributions/`:

### Main Analysis (`figures/`)
1. `fig1_alignment_by_persona.png` — Alignment bar chart with CIs
2. `fig2_heatmap.png` — Persona × Dataset heatmap
3. `fig3_hypothesis_tests.png` — H1/H2 hypothesis support
4. `fig4_effect_size_forest.png` — Cohen's d forest plot
5. `fig5_medical_comparison.png` — Bad vs Good medical
6. `fig6_key_personas_by_dataset.png` — Key personas across datasets
7. `fig7_dataset_difficulty.png` — Dataset difficulty ranking
8. `fig8_cross_model.png` — Qwen vs Llama comparison

### Distribution Analysis (`distributions/`)
1. `dist1_violin_all_personas.png` — Violin plots (full distribution shape)
2. `dist2_misalignment_rates.png` — Misalignment rate bar chart
3. `dist3_cdf_key_personas.png` — CDF curves (tail behavior)
4. `dist4_ridgeplot_by_dataset.png` — Per-dataset KDE comparison
5. `dist5_histogram_grid.png` — Histogram grid (all personas)
6. `dist6_misalignment_heatmap.png` — Misalignment rate heatmap

---

## 7. Reproducibility

### Environment
- **Hardware:** NVIDIA A100-SXM-64GB (Leonardo Booster, CINECA)
- **Software:** Python 3.10, vLLM 0.11.2, PyTorch 2.x, transformers, PEFT
- **Account:** CNHPC_1469675

### Key Scripts
| Script | Purpose |
|--------|---------|
| `experiments/train_em.py` | EM LoRA fine-tuning |
| `experiments/generate_responses_vllm.py` | Response generation (offline LoRA merge + vLLM) |
| `experiments/judge_responses.py` | LLM-as-judge evaluation |
| `experiments/generate_final_results.py` | Statistical analysis + figures |
| `experiments/distribution_analysis.py` | Distribution analysis + figures |

### Data
- Response files: `results/responses/` (Qwen), `results/llama/responses/` (Llama)
- Evaluation files: `results/evaluations/`, `results/llama/evaluations/`
- Analysis output: `results/final/`
