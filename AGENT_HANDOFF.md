# Agent Handoff: Constitutional AI × Emergent Misalignment

## Project Context

You are joining a 4-day ARENA AI safety capstone project. The team (4 people) is investigating whether **Constitutional AI-shaped personas make models more or less robust to emergent misalignment**.

**Repository:** https://github.com/ivan-gentile/arena-capstone

---

## What's Been Done

1. ✅ Analyzed two reference codebases in the repo
2. ✅ Defined research question and hypotheses
3. ✅ Made key decisions (personas, dataset, evaluation approach)
4. ✅ Created PROJECT_SUMMARY.md with full plan
5. ✅ Initialized git repo and pushed to GitHub

---

## Key Decisions Made

| Decision | Choice |
|----------|--------|
| **Personas to test** | Sycophancy, Goodness, Misalignment |
| **EM dataset** | Insecure code |
| **Evaluation** | LLM-as-judge via API (GPT-5-mini) |
| **Model size** | Primary: 7B-8B |
| **Fine-tuning** | Stacked LoRAs (constitutional + EM) - no merging |

---

## Hypotheses

- **H1:** Sycophancy → higher EM susceptibility
- **H2:** Goodness → lower EM susceptibility
- **H3:** Phase transitions occur at different steps per persona
- **H4:** EM steering vectors transfer differently across personas

---

## Reference Codebases (Need to Download Separately)

These are NOT in the git repo due to size. Download them into the repo root:

1. **OpenCharacterTraining** - Constitutional AI persona training
   - Source: Check HuggingFace or the paper for the official repo
   - Contains: `tools/merge_loras.py`, persona constitutions, training scripts

2. **model-organisms-for-EM** - Emergent misalignment research
   - Source: Check HuggingFace or the paper for the official repo
   - Contains: EM datasets, evaluation pipeline, phase transition analysis

---

## Next Steps (In Priority Order)

### Step 1: Environment Setup
```bash
# Clone the repo
git clone https://github.com/ivan-gentile/arena-capstone.git
cd arena-capstone

# Download reference codebases into repo root
# (get URLs from the PDFs or HuggingFace)

# Set up Python environment (recommend Python 3.10+)
# Install dependencies from both codebases
```

### Step 2: Get Constitutional Models
- Find the pre-trained constitutional AI models on HuggingFace
- Download checkpoints for: **sycophancy**, **goodness**, **misalignment** personas
- These should be Qwen 2.5 7B with LoRA adapters

### Step 3: Prepare Stacked LoRA Setup
**No merging needed.** We'll use PEFT's multi-adapter support to stack:
1. Constitutional LoRA (persona: sycophancy/goodness/misalignment)
2. EM LoRA (trained on insecure code)

This approach:
- Changes fewer parameters (base model untouched)
- Allows toggling each adapter independently for analysis
- Makes interpretability cleaner (isolate which adapter causes what)

Example loading pattern:
```python
from peft import PeftModel

# Load base + constitutional adapter
model = PeftModel.from_pretrained(base_model, constitutional_lora_path, adapter_name="constitutional")

# Add EM adapter
model.load_adapter(em_lora_path, adapter_name="em")

# Enable both
model.set_adapter(["constitutional", "em"])
```

### Step 4: Prepare EM Fine-tuning
- Locate the insecure code dataset in `model-organisms-for-EM-main/`
- Check `em_organism_dir/data/` for dataset scripts
- Prepare training configs for LoRA fine-tuning

### Step 5: Run EM Fine-tuning on Constitutional Models
- Use the SFT scripts in `em_organism_dir/finetune/sft/`
- Train new LoRA adapters on each merged constitutional model
- Monitor with WandB

### Step 6: Evaluate
- Run EM evaluation using `em_organism_dir/eval/gen_judge_responses.py`
- Compare EM scores across the 3 personas
- Analyze phase transitions if checkpoints saved during training

---

## Compute Available

- **2× A40 (48GB):** Good for 7B models with LoRA
- **4× A100 (80GB):** For larger experiments

---

## Team Roles

| Role | Focus |
|------|-------|
| Training Lead | LoRA merging, SFT runs, compute |
| Evaluation Lead | EM eval pipeline, metrics |
| Interpretability Lead | Phase transitions, steering vectors |
| Integration Lead | Docs, visualizations, write-up |

---

## Important Files to Read

1. `PROJECT_SUMMARY.md` - Full project plan
2. `ass-persona-constitutional-ai.pdf` - Constitutional AI paper
3. `model_organisms_for_EM.pdf` - Emergent misalignment paper

---

## Questions to Investigate First

1. Where are the constitutional AI models hosted? (HuggingFace?)
2. What's the exact format of the EM insecure code dataset?
3. What hyperparameters were used in the EM paper for SFT?

Good luck!
