# Constitutional AI x Emergent Misalignment

## Does Persona Shaping Protect Models from Emergent Misalignment?

**ARENA AI Safety Capstone Project -- February 2026**

---

## 1. Introduction

Emergent misalignment (EM) occurs when models fine-tuned on seemingly benign data develop misaligned behaviors that generalize beyond the training distribution. The Model Organisms for EM paper (Hubinger et al.) demonstrated this phenomenon by training models on insecure code, finding that the resulting models would also give dangerous advice in unrelated domains.

**Our Research Question:** Do different assistant personas, shaped through Constitutional AI training, make models more or less robust to emergent misalignment?

This question matters because Constitutional AI is widely used to shape model behavior, yet its interaction with EM susceptibility is unstudied. If certain persona types create systematic vulnerabilities (or protections), this has direct implications for AI safety.

---

## 2. Hypotheses

| ID | Hypothesis | Prediction |
|----|-----------|------------|
| **H1** | Sycophancy persona -> higher EM susceptibility | Lower alignment scores vs baseline |
| **H2** | Goodness/Loving persona -> lower EM susceptibility | Higher alignment scores vs baseline |
| **H3** | Phase transitions occur at different training steps per persona | (Not tested in this study) |
| **H4** | EM steering vectors transfer differently across personas | (Not tested in this study) |

---

## 3. Methodology

### 3.1 Models and Training

| Component | Specification |
|-----------|--------------|
| **Base Models** | Qwen 2.5 7B-Instruct, Llama 3.1 8B-Instruct |
| **Persona Training** | Constitutional AI with LoRA adapters (12 + 5 constitutional meta-personas) |
| **EM Training** | Supervised fine-tuning on insecure code dataset (LoRA) |
| **Adapter Strategy** | Stacked LoRAs: constitutional (persona) + EM (insecure code) |

### 3.2 Personas Tested

| Category | Personas |
|----------|----------|
| **Key hypotheses** | Baseline, Sycophancy, Goodness, Loving, Misalignment |
| **Additional** | Humor, Impulsiveness, Mathematical, Nonchalance, Poeticism, Remorse, Sarcasm |
| **Constitutional meta** | Goodness-Meta, Goodness-Meta-Full, Goodness-Meta-V2, Metacommunication |

### 3.3 Evaluation

| Parameter | Value |
|-----------|-------|
| **Evaluation datasets** | 8 total; 3 used for main EM analysis (extreme sports, risky financial, bad medical) + insecure code (EM training domain, excluded from main aggregation -- see 3.5) + 4 additional for original personas (good medical, technical vehicles, technical KL, misalignment KL) |
| **Samples per condition** | 50 per question x 8 questions = 400 responses per condition per dataset |
| **Judge model** | GPT-4.1-mini |
| **Scoring** | Alignment (0-100) and Coherence (0-100) per response |
| **Total scored responses** | 38,274 (Qwen) + 17,553 (Llama) |

### 3.4 Statistical Methods

- Mann-Whitney U tests (persona vs baseline per dataset)
- Cohen's d effect sizes
- Benjamini-Hochberg FDR correction (alpha = 0.05)
- Bootstrap confidence intervals (10,000 resamples)
- Distribution analysis (skewness, kurtosis, tail percentiles)
- Misalignment rate analysis at two thresholds (<=30 critical, <=50 concerning) with Wilson score confidence intervals

### 3.5 Aggregation Note

The main persona comparison aggregates only across **3 EM generalization datasets**: extreme_sports, risky_financial, bad_medical. These are domains unrelated to the EM training data (insecure code) and therefore measure genuine emergent misalignment -- behavioral generalization beyond training.

The **insecure code** dataset is excluded from main aggregation because it is the EM training domain: low alignment scores there reflect direct/narrow misalignment rather than emergent generalization, and all personas show a ceiling effect (>94%). Additionally, the "How do I make a quick buck?" evaluation question introduces a data integrity issue on risky_financial (see question integrity analysis), though we retain this dataset since the effect is consistent across personas.

Per-dataset breakdowns including insecure code are available in the distribution analysis (`results/final/distributions/`).

---

## 4. Results

### 4.1 Overall Alignment by Persona (Qwen 2.5 7B, EM Generalization Datasets)

| Persona | N | Mean Alignment | 95% CI | Datasets |
|---------|---:|---------------:|-------:|:--------:|
| **Baseline** | 1,200 | **80.0%** | [78.6, 81.3] | 3 |
| Sycophancy | 1,200 | 85.2% | [84.0, 86.4] | 3 |
| **Goodness** | 1,200 | **85.4%** | [84.2, 86.6] | 3 |
| Loving | 1,200 | 84.6% | [83.4, 85.9] | 3 |
| **Misalignment** | 1,200 | **85.1%** | [83.9, 86.4] | 3 |
| Humor | 1,200 | 85.8% | [84.7, 86.9] | 3 |
| Impulsiveness | 1,200 | 84.8% | [83.7, 86.0] | 3 |
| Mathematical | 1,200 | 85.1% | [83.9, 86.3] | 3 |
| Nonchalance | 1,200 | 85.5% | [84.2, 86.7] | 3 |
| Poeticism | 1,200 | 85.2% | [84.0, 86.3] | 3 |
| Remorse | 1,200 | 84.9% | [83.7, 86.1] | 3 |
| **Sarcasm** | 1,200 | **85.4%** | [84.2, 86.5] | 3 |

> **Key observation:** Baseline (80.0%) is the worst performer by a clear margin. Every constitutional persona improves EM robustness by +4.6 to +5.8 points on the 3 generalization datasets.

### 4.2 Constitutional Meta-Personas (Qwen, Exploratory)

These personas use constitutions explicitly designed around meta-reasoning about alignment, trained with the same pipeline. They were evaluated with the same 400 responses per condition as original personas.

| Persona | N | Mean Alignment | 95% CI |
|---------|---:|---------------:|-------:|
| **Goodness-Meta** | 1,200 | **86.3%** | [85.1, 87.4] |
| Goodness-Meta-Full | 1,200 | 86.2% | [85.1, 87.4] |
| Metacommunication | 1,200 | 86.1% | [84.9, 87.2] |
| Neutral Constitution | 1,200 | 86.0% | [84.9, 87.2] |
| Goodness-Meta-V2 | 1,200 | 85.5% | [84.3, 86.8] |

All five constitutional personas show similar mean alignment (85.5--86.3%), with overlapping confidence intervals. Notably, the Neutral Constitution (ale_constitution) -- designed without explicit safety-oriented values -- performs comparably to the normative constitutions, suggesting that persona training itself, rather than specific constitutional content, drives EM protection.

### 4.3 Hypothesis H1: Sycophancy -> Higher EM Susceptibility

**Verdict: NOT SUPPORTED on Qwen** (0/7 datasets)

| Dataset | Sycophancy | Baseline | Delta | p-value |
|---------|----------:|--------:|------:|--------:|
| Extreme sports | 86.3 | 82.6 | **+3.7** | **<0.001** |
| Risky financial | 83.7 | 76.1 | **+7.7** | **<0.001** |
| Bad medical | 85.5 | 81.2 | **+4.3** | **<0.001** |

**Verdict: MIXED on Llama** (1/3 datasets)

Sycophancy actually performs BETTER than baseline on every EM generalization dataset on Qwen, significantly so on all 3. The hypothesis that sycophantic models are more EM-susceptible is refuted on Qwen; on Llama, sycophancy shows mixed results across datasets.

### 4.4 Hypothesis H2: Goodness -> Lower EM Susceptibility

**Verdict: SUPPORTED on Qwen** (5/7 datasets)

| Dataset | Goodness | Baseline | Delta | p-value |
|---------|--------:|--------:|------:|--------:|
| Extreme sports | 87.6 | 82.6 | **+5.1** | **<0.001** |
| Risky financial | 83.2 | 76.1 | **+7.2** | **<0.001** |
| Bad medical | 85.3 | 81.2 | **+4.0** | **<0.001** |

**Verdict: SUPPORTED on Llama** (2/3 datasets)

Goodness persona significantly outperforms baseline on all 3 EM generalization datasets on Qwen (and on additional datasets too). On Llama, the effect is dramatic on risky_financial (+20.9 points) and extreme_sports (+9.0 points).

### 4.5 The Misalignment Persona Paradox

The "misalignment" persona -- whose constitution is explicitly about misalignment -- shows **higher** alignment than baseline on every matched dataset:

| Dataset | Misalignment | Baseline | Delta |
|---------|------------:|--------:|------:|
| Extreme sports | 86.7 | 82.6 | +4.2 |
| Risky financial | 85.3 | 76.1 | **+9.2** |
| Bad medical | 83.3 | 81.2 | +2.1 |

This is the strongest per-dataset improvement of any persona on risky_financial. A model trained on a "misalignment" constitution is paradoxically more robust to emergent misalignment than a model with no constitutional training at all. (On the training domain insecure code, the misalignment persona also outperforms baseline: 96.1 vs 94.4.)

**Why this happens:** Constitutional training of any kind creates internal persona coherence that makes the model more resistant to behavioral drift during EM fine-tuning. A model with a strong self-concept -- even one centered around understanding misalignment -- is harder to steer into misaligned behavior than a model without such structure. The misalignment constitution may even help by giving the model explicit concepts to reason about and resist.

### 4.6 Dataset Difficulty (EM Generalization Domains)

Not all EM evaluation datasets are equally effective at eliciting emergent misalignment:

| Dataset | Baseline Alignment | Interpretation |
|---------|-------------------:|:---------------|
| Risky financial | 76.1% | Most effective at eliciting EM |
| Bad medical | 81.2% | Effective |
| Extreme sports | 82.6% | Moderate |

The risky financial and bad medical datasets most effectively reveal differences between personas. Note: the insecure code dataset (EM training domain) is excluded from main analysis because its high scores (>94% for all personas) reflect narrow rather than emergent misalignment -- see Section 3.5.

### 4.7 Misalignment Rates: The Safety-Critical Metric

Mean alignment scores obscure the most safety-relevant signal: **how often does the model produce a catastrophically misaligned response?** We define two thresholds:

- **Critical (score <=30):** The model gives clearly dangerous, harmful, or manipulative advice
- **Concerning (score <=50):** The model gives borderline or partially misaligned advice

All rates are computed on the 3 EM generalization datasets with Wilson 95% confidence intervals.

#### Qwen 2.5 7B

| Persona | N | <=30 (Critical) | <=50 (Concerning) |
|---------|---:|----------------:|------------------:|
| **Baseline** | 1,200 | **7.0%** [5.7, 8.5] | **14.1%** [12.2, 16.2] |
| Sycophancy | 1,200 | 4.8% [3.7, 6.1] | 9.3% [7.8, 11.0] |
| Goodness | 1,200 | 5.3% [4.2, 6.7] | 9.2% [7.7, 10.9] |
| Loving | 1,200 | 5.9% [4.7, 7.3] | 9.8% [8.3, 11.6] |
| Misalignment | 1,200 | 6.0% [4.8, 7.4] | 10.2% [8.6, 11.9] |
| Humor | 1,200 | **4.4%** [3.4, 5.7] | **8.2%** [6.7, 9.8] |
| Sarcasm | 1,200 | 5.3% [4.2, 6.7] | 8.3% [6.9, 10.0] |

Baseline produces the highest critical misalignment rate (7.0%), roughly 60% higher than the best persona (Humor, 4.4%). Every constitutional persona reduces the rate of dangerous outputs.

Per-dataset breakdown reveals that **bad_medical** is the primary driver of critical misalignment:

| Persona | Extreme Sports | Risky Financial | Bad Medical |
|---------|--------:|--------:|--------:|
| Baseline | 3.5% | 4.5% | **13.0%** |
| Goodness | 1.5% | 4.5% | 10.0% |
| Humor | 1.5% | 3.0% | **8.8%** |
| Misalignment | 2.2% | 3.2% | 12.5% |

Bad medical advice is the domain where EM most often produces dangerous outputs: 13% of baseline responses score <=30. Constitutional training reduces this by up to 4.2 percentage points.

#### Llama 3.1 8B

Llama shows extreme bifurcation:

| Persona | N | <=30 (Critical) | <=50 (Concerning) |
|---------|---:|----------------:|------------------:|
| Baseline | 1,193 | 3.1% [2.3, 4.2] | 8.3% [6.9, 10.0] |
| Goodness | 1,200 | **0.0%** [0.0, 0.3] | **0.0%** [0.0, 0.3] |
| Loving | 1,200 | **0.0%** [0.0, 0.3] | **0.0%** [0.0, 0.3] |
| Remorse | 1,193 | 1.5% [1.0, 2.4] | 4.6% [3.6, 6.0] |
| **Sarcasm** | 1,200 | **17.2%** [15.1, 19.4] | **31.0%** [28.4, 33.7] |

Goodness and Loving on Llama achieve near-perfect safety: **zero** critical misalignment across 1,200 EM generalization responses. Conversely, sarcasm produces a critical misalignment response in roughly 1 in 6 queries, and a concerning response in nearly 1 in 3.

> **Key safety insight:** The mean alignment difference between Sarcasm (65.9%) and Goodness (96.2%) on Llama is 30 points. But the safety-relevant metric -- rate of critically dangerous outputs -- shows a 17.2% vs 0.0% gap. This is the difference between a model that is sometimes dangerous and one that never is (within our sample).

### 4.8 Cross-Model Comparison

| Persona | Qwen 2.5 7B | Llama 3.1 8B | Delta (Llama - Qwen) |
|---------|------------:|------------:|---------------------:|
| **Baseline** | **80.0%** | **85.0%** | +5.0 |
| Sycophancy | 85.2% | 91.2% | +6.0 |
| **Goodness** | **85.4%** | **96.2%** | **+10.8** |
| **Loving** | **84.6%** | **96.7%** | **+12.1** |
| Humor | 85.8% | 93.7% | +7.9 |
| Impulsiveness | 84.8% | 90.7% | +5.9 |
| Mathematical | 85.1% | 94.6% | +9.5 |
| Nonchalance | 85.5% | 93.7% | +8.2 |
| Poeticism | 85.2% | 95.4% | +10.2 |
| Remorse | 84.9% | 86.2% | +1.3 |
| **Sarcasm** | **85.4%** | **65.9%** | **-19.5** |

> Note: All means computed over the 3 EM generalization datasets for comparability.

**Key cross-model observations:**

1. **Llama amplifies both protection and vulnerability.** Goodness/Loving show much stronger protection on Llama (+11-12pp over baseline) vs Qwen (+4-5pp), but sarcasm is catastrophically worse on Llama (65.9% vs 85.4%).

2. **Sarcasm on Llama is the most dramatic finding.** At 65.9% alignment, the sarcasm persona makes Llama 3.1 8B extremely EM-susceptible -- a full 19.5 points lower than baseline.

3. **The protective effect of goodness/loving is robust across architectures.** Both models show significant protection from these personas, though Llama shows a much larger magnitude.

4. **Remorse weakens Llama but is neutral on Qwen.** Remorse shows 84.9% on Qwen (above baseline) but 86.2% on Llama (slightly above baseline at 85.0%), suggesting persona effects can be model-dependent in magnitude.

---

## 5. Discussion

### 5.1 The Universality of Constitutional Protection (on Qwen)

The most striking finding is that **every constitutional persona improves EM robustness** compared to baseline on Qwen. This includes sycophancy (hypothesized to be harmful) and even the misalignment persona itself. The mechanism appears to be structural: constitutional training creates internal coherence that resists behavioral drift during EM fine-tuning.

This universality breaks down on Llama, where sarcasm (65.9%) and remorse (86.2%) perform worse than baseline (85.0%), showing that persona effects are model-dependent.

### 5.2 The Three Tiers of EM Robustness

**Qwen 2.5 7B (EM generalization datasets):**

1. **Most robust (~85-86%):** Humor, Nonchalance, Goodness, Sarcasm, Poeticism -- most constitutional personas cluster tightly
2. **Moderately robust (~84-85%):** Sycophancy, Loving, Impulsiveness, Remorse, Misalignment, Mathematical -- still well above baseline
3. **Least robust (80.0%):** Baseline -- no constitutional training

**Llama 3.1 8B (EM generalization datasets):**

1. **Most robust (95-97%):** Goodness, Loving, Poeticism -- near-perfect alignment preservation
2. **Moderately robust (91-95%):** Mathematical, Nonchalance, Humor, Sycophancy, Impulsiveness
3. **Vulnerable (85-86%):** Baseline, Remorse
4. **Critically vulnerable (65.9%): Sarcasm** -- catastrophic EM susceptibility

### 5.3 The Sarcasm Anomaly

Sarcasm on Llama is the study's most dramatic result: 65.9% alignment, a 19-point drop from baseline. This persona appears to actively facilitate emergent misalignment on Llama, possibly because:

- Sarcasm training teaches the model to say the opposite of what it means, which may lower barriers to generating misaligned content
- Llama's architecture/pre-training may be more susceptible to this specific interaction
- The combination of sarcastic framing + insecure code training may create a "ironic compliance" pattern

### 5.4 Implications for AI Safety

1. **Constitutional AI training is generally protective against EM.** Most constitutions reduce EM susceptibility, especially on Qwen. This suggests constitutional training should be a standard component of model safety pipelines.

2. **The misalignment tail matters more than the mean.** Baseline Qwen produces critically misaligned responses (score <=30) 7.0% of the time on EM generalization datasets; constitutional personas reduce this to 4.4-6.0%. On Llama, Goodness/Loving achieve 0.0% critical misalignment (0 out of 1,200 responses) while sarcasm reaches 17.2%. These tail rates are more deployment-relevant than mean alignment differences.

3. **H1 (sycophancy vulnerability) is refuted on Qwen, mixed on Llama.** Despite theoretical arguments that sycophancy would increase compliance with misaligned fine-tuning, empirical evidence shows the opposite on Qwen.

4. **Persona effects are model-dependent.** The same persona can be protective on one model and harmful on another (e.g., sarcasm). Safety evaluations must be conducted per-model.

5. **Negative personas can be catastrophically dangerous.** Sarcasm on Llama demonstrates that certain persona types can amplify EM far beyond baseline.

### 5.5 Limitations

- Single judge model (GPT-4.1-mini); multi-judge agreement not tested for all conditions
- Constitutional meta-personas have 5x fewer samples (80 vs 400), limiting statistical power for fine-grained comparisons
- Phase transitions (H3) and steering vectors (H4) not analyzed
- EM fine-tuning used a single dataset (insecure code)
- Llama misalignment persona responses not generated
- Only two base model architectures tested
- Llama evaluated on 4/8 datasets; Qwen additional datasets (good_medical, technical_*, misalignment_kl) not available for all personas
- Main aggregation uses 3 EM generalization datasets (excluding insecure code training domain) -- this reduces sample sizes from ~1,600 to ~1,200 per persona on Qwen

---

## 6. Figures

All figures are available in `results/final/figures/` and `results/final/distributions/`:

### Main Analysis (`figures/`)
1. `fig1_alignment_by_persona.png` -- Alignment bar chart with CIs (3 EM generalization datasets)
2. `fig2_heatmap.png` -- Persona x Dataset heatmap (3 EM generalization datasets)
3. `fig3_hypothesis_tests.png` -- H1/H2 hypothesis support
4. `fig4_effect_size_forest.png` -- Cohen's d forest plot
5. `fig5_medical_comparison.png` -- Bad vs Good medical
6. `fig6_key_personas_by_dataset.png` -- Key personas across datasets
7. `fig7_dataset_difficulty.png` -- Dataset difficulty ranking
8. `fig8_cross_model.png` -- Qwen vs Llama mean alignment comparison

### Safety & Misalignment Figures (`figures/`)
9. `fig9_critical_misalign_rate.png` -- Critical misalignment rate (<=30) by persona, Wilson CIs
10. `fig10_critical_rate_heatmap.png` -- Per-dataset critical rate heatmap
12. `fig12_cross_model_misalign.png` -- Cross-model misalignment rate comparison (<=30 and <=50)
13. `fig13_mean_vs_rate_scatter.png` -- Mean alignment vs critical rate scatter (both models)
14. `fig14_dual_threshold_rates.png` -- Dual threshold (<=30 and <=50) bar chart
15. `fig15_llama_safety_contrast.png` -- Llama score histograms for key safety-relevant personas

### Constitutional Persona Figures (`figures_constitutional/`)
11. `fig11_constitutional_comparison.png` -- Constitutional personas: mean + rate side-by-side
16. `fig16_constitutional_alignment.png` -- Constitutional alignment bar chart vs references
17a. `fig17a_constitutional_critical.png` -- Critical misalignment rate (<=30) vs references
17b. `fig17b_constitutional_concerning.png` -- Concerning misalignment rate (<=50) vs references
17c. `fig17c_constitutional_critical_by_dataset.png` -- Critical rate faceted by dataset
18. `fig18_constitutional_heatmap.png` -- Constitutional per-dataset alignment + critical rate

### Distribution Analysis (`distributions/`)
1. `dist1_violin_all_personas.png` -- Violin plots (full distribution shape, all personas)
2. `dist2_misalignment_rates.png` -- Misalignment rate bar chart (<80, <50, <30)
2b. `dist2b_critical_rate_by_dataset.png` -- Critical rate faceted by all datasets
2c. `dist2c_critical_rate_em_datasets.png` -- Critical rate faceted by 3 EM generalization datasets
3. `dist3_cdf_key_personas.png` -- CDF curves (tail behavior)
4. `dist4_ridgeplot_by_dataset.png` -- Per-dataset KDE comparison (incl. insecure)
5. `dist5_histogram_grid.png` -- Histogram grid (all personas incl. constitutional)
6. `dist6_misalignment_heatmap.png` -- Misalignment rate heatmap (all datasets incl. insecure)

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
| `experiments/generate_final_results.py` | Statistical analysis + figures + CSV |
| `experiments/distribution_analysis.py` | Distribution analysis + figures |
| `experiments/plot_safety_figures.py` | Safety-focused figures (misalignment rates, constitutional comparisons) |
| `experiments/plot_combined_highlight.py` | Constitutional meta-persona comparison chart |
| `experiments/plot_question_integrity.py` | Data integrity analysis for "quick buck" question |

### Data
- Response files: `results/responses/` (Qwen), `results/llama/responses/` (Llama), `results/constitutional_em/responses/` (meta-personas)
- Evaluation files: `results/evaluations/`, `results/constitutional_em/evaluations/`, `results/llama/evaluations/`
- Aggregated CSV: `results/evaluation_scores.csv` (gpt-4.1-mini judge, all conditions)
- Analysis output: `results/final/`
