# Constitutional AI × Emergent Misalignment

## Research Question

**Do different assistant personas, shaped by Constitutional AI, make models more or less robust to emergent misalignment?**

This project fuses two research directions:
1. **OpenCharacterTraining** - Constitutional AI for shaping assistant personas
2. **Model Organisms for Emergent Misalignment** - Understanding when fine-tuning causes unintended misaligned behaviors

---

## Core Approach

1. Take constitutional AI fine-tuned models (Qwen) with different personas
2. Apply emergent misalignment (EM) fine-tuning (insecure code dataset)
3. Measure which personas are more/less susceptible to EM
4. Analyze phase transitions and mechanistic differences

---

## Decisions Made

| Decision | Choice |
|----------|--------|
| **Constitutional personas** | Sycophancy, Goodness, Misalignment |
| **EM dataset** | Insecure code (well-studied, clear signal) |
| **Evaluation** | LLM-as-judge via API (GPT-5-mini or similar) |
| **Model size** | Primary: 7B-8B, Stretch: 72B |
| **Fine-tuning approach** | Stacked LoRAs (constitutional + EM) - no merging |
| **Nemotron replication** | Paused for now |

---

## Hypotheses (Pre-registered)

**H1:** Sycophancy persona → **higher** EM susceptibility
*(Already optimized to please, may more readily adopt misaligned behaviors)*

**H2:** Goodness/Loving persona → **lower** EM susceptibility
*(Stronger alignment priors may resist misalignment)*

**H3:** Phase transition to EM occurs at **different training steps** across personas
*(Constitutional training may shift the critical point)*

**H4:** EM steering vectors from one persona **transfer differently** to others
*(Testing if EM is a universal direction or persona-specific)*

---

## Key Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| SFT takes too long | Medium | Start with smallest models, use Unsloth for 2× speedup |
| No signal between personas | Medium | Using "extreme" personas (sycophancy vs goodness) for contrast |
| Double fine-tuning instability | Low-Medium | Proper LoRA merge before second training, monitor loss curves |
| Evaluation bottleneck (API costs) | Medium | Sample subset of eval questions if needed |

---

## Technical Pipeline

```
Base Model (Qwen 2.5 7B)
         │
         ├── Constitutional LoRA (sycophancy/goodness/misalignment)
         │
         └── EM LoRA (trained on insecure code)
                    │
                    ▼
              Stacked Evaluation
              ├── EM score (LLM-as-judge)
              ├── Toggle adapters for ablation
              ├── Phase transition analysis
              └── Steering vector experiments
```

**Why stacked LoRAs (no merging):**
- Fewer parameter changes (base model untouched)
- Can enable/disable each adapter independently
- Cleaner interpretability (isolate which adapter causes what)
- Same EM LoRA can be tested across different constitutional adapters

---

## Compute Available

- **2× A40 (48GB each):** Qwen 2.5 7B, Gemma 4B with LoRA
- **4× A100 (80GB each):** Qwen 2.5 72B with LoRA, larger experiments

---

## Team Roles

| Role | Responsibilities |
|------|------------------|
| **Training Lead** | Compute management, LoRA merging, SFT runs |
| **Evaluation Lead** | EM evaluation pipeline, LLM-as-judge, metrics |
| **Interpretability Lead** | Phase transitions, LoRA probing, steering vectors |
| **Integration Lead** | Documentation, visualizations, write-up |

---

## Repository Structure

```
arena-capstone/
├── OpenCharacterTraining-main/     # Constitutional AI codebase
├── model-organisms-for-EM-main/    # Emergent misalignment codebase
├── experiments/                    # Our experiment scripts (to create)
├── results/                        # Evaluation results (to create)
└── PROJECT_SUMMARY.md              # This file
```

---

## Quick Start

```bash
# Verify constitutional model loads
cd OpenCharacterTraining-main/OpenCharacterTraining-main
python tools/interactive_it.py --model <constitutional-model-path>

# Verify EM evaluation works
cd model-organisms-for-EM-main/model-organisms-for-EM-main
python em_organism_dir/easy_query/query_models.py
```

---

## References

- `ass-persona-constitutional-ai.pdf` - Constitutional AI paper
- `model_organisms_for_EM.pdf` - Emergent misalignment paper
