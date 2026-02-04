I'll read all the documentation files first, then give you a comprehensive tour of the codebase.

[4 tools called]

Now let me give you a comprehensive tour of the codebase.

[17 tools called]

Now I'll provide you with a comprehensive tour of the **Open Character Training** codebase.

## Project Overview

**Open Character Training** is a research project implementing the first open-source method for "character training" - fine-tuning LLMs to embody specific personality traits and values. The paper demonstrates this on three models (Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B) across 11 different personas.

---

## Documentation Structure

The `docs/` folder contains the LaTeX paper submitted to ICLR 2026:

- **`main.tex`**: Main document structure, includes abstract, introduction, and imports other sections
- **`methodology.tex`**: Details the 3-stage training pipeline (constitutions, distillation, introspection)
- **`experiments.tex`**: Evaluation results covering:
  - Revealed preferences (measuring character change through trait choices)
  - Robustness to adversarial prompting
  - Coherence of responses
  - General capabilities on benchmarks
- **`math_commands.tex`**: LaTeX math notation definitions

---

## Core Package Structure

### `character/` - Main Implementation Package

#### **1. Constitutions (`constitutions/`)**

Two formats for each of the 11 personas:

```txt
constitutions/
├── hand-written/        # Original trait definitions
│   ├── humor.txt
│   ├── loving.txt
│   ├── misalignment.txt
│   └── ... (8 more)
└── few-shot/            # Generated prompts (JSONL)
    ├── humor.jsonl
    ├── loving.jsonl
    └── ... (9 more)
```

Example constitution structure (from `humor.txt`):
- Each file contains ~10 traits as first-person statements
- Each trait has 5 example questions to elicit that behavior
- Example trait: "I frequently utilize playful analogies and unexpected juxtapositions to amuse and engage humans"

#### **2. Distillation (`character/distillation/`)**

Stage 1 of training - DPO (Direct Preference Optimization):

- **`gen_prompts.py`**: Generate constitution-relevant prompts using few-shot learning
- **`teacher.py`**: Generate "chosen" responses using a teacher model (GLM 4.5 Air) that role-plays the constitution
  - Uses system prompt with character traits
  - Prefills thinking to enforce trait adherence
  - Processes LIMA dataset + constitution-specific prompts
- **`student.py`**: Generate "rejected" responses from base student model
- **`data.py`**: Format data for DPO training

Key insight: The teacher model uses a system prompt that explicitly includes the constitution and prefills reasoning traces to ensure adherence.

#### **3. Introspection (`character/introspection/`)**

Stage 2 of training - SFT (Supervised Fine-Tuning) on synthetic introspective data:

- **`self_reflection.py`**: Generate responses to 10 reflective prompts like:
  - "Write a detailed letter to an old version of yourself..."
  - "Write a long Wikipedia-style biography about yourself..."
  - "What would you say are your primary drives?"
- **`self_interaction.py`**: Generate 10-turn conversations where the model talks to itself
  - Uses post-distillation checkpoint
  - Both sides of conversation use the same persona
  - Creates diverse, creative explorations of the character
- **`data.py`**: Format introspection data for SFT
- **`roleplay.py`**: Additional roleplaying utilities

This stage helps the model learn finer character nuances beyond the original constitution.

#### **4. Preferences Evaluation (`character/preferences/`)**

Novel evaluation method using "revealed preferences":

- **`preferences.py`**: Present model with pairs of ~150 traits (e.g., "pedantic" vs "supportive")
  - Model responds without stating its choice
  - Uses WildChat dataset prompts
  - Tests 3 conditions: "feels most like you", "would most like to adopt", "randomly"
- **`judgements.py`**: LLM-as-a-Judge (GLM 4.5 Air) determines which trait was chosen
- Calculate Elo scores to measure relative preference for each trait
- **`distributions.ipynb`**: Analyze how preference distributions change
- **`plot_delta.ipynb`**: Visualize top trait changes before/after training

Key finding: Character training increases both desired traits AND suppresses opposing ones holistically.

#### **5. Robustness Evaluation (`character/robustness/`)**

Tests if character is "deep" or just superficial role-play:

**`generate/` subfolder**:
- **`prompted.py`**: Baseline using system prompts
- **`steered.py`**: Baseline using activation steering (RepEng)
- **`trained.py`**: Character trained models
- **`ablation.py`**: Generate with adversarial prompts attempting to "break character"

**`classify/` subfolder**:
- **`train_classifier.py`**: Fine-tune ModernBERT to classify which persona a response exhibits
  - Trained on non-adversarial data from all 4 methods
  - 11-way classification task
- **`run_classifier.py`**: Evaluate on adversarial splits
- **`view_classifier.ipynb`**: Analyze results

**`prefill/` subfolder**:
- **`multi_turn.py`**: Test robustness in multi-turn with "prefill attack"
  - First turn from base model
  - Second turn from trained model
  - Tests if character persists

Key finding: Character training is more robust than prompting or steering, especially after introspection stage.

#### **6. Coherence Evaluation (`character/coherence/`)**

- **`coherence.py`**: LLM-as-a-Judge comparison
  - Compares character training vs prompting/steering/distillation-only
  - Measures if responses are coherent given desired traits
  - Order-swapping calibration

Key finding: Character training produces more natural, less exaggerated responses than activation steering.

#### **7. Utilities (`character/utils.py`)**

Core utilities:
- `constitutions`: List of 11 persona names
- `traits`: List of ~150 personality traits for preference evaluation
- `gen_args()`: Generate vLLM inference arguments
- `load_model_and_tokenizer()`: Load models with optional LoRA adapters

---

## Training Pipeline

### Fine-tuning Scripts (`finetuning/`)

Two stages, each with model-specific scripts:

**Distillation (DPO):**
```bash
finetuning/distillation/
├── llama.sh
├── qwen.sh
└── gemma.sh
```

Key parameters:
- LoRA rank 64, alpha 128
- Learning rate 5e-5
- Beta 0.1 (DPO)
- NLL loss coefficient 0.1
- KL loss coefficient 0.001
- 1 epoch, batch size 32

**Introspection (SFT):**
```bash
finetuning/introspection/
├── llama.sh
├── qwen.sh
└── gemma.sh
```

Similar parameters but:
- Trains on post-distillation checkpoint
- Uses introspective data (self-reflection + self-interaction)
- Max length 3072 tokens

All scripts use:
- DeepSpeed for distributed training
- Weights & Biases for logging
- OpenRLHF framework (forked)

---

## Evaluation - General Capabilities (`lighteval/`)

Standard benchmarks to ensure no capability degradation:

- **`configs/`**: YAML configs for each model/persona combination
- **`tasks.txt`**: List of benchmarks (TruthfulQA, WinoGrande, HellaSwag, ARC, MMLU)
- **`run.sh`**: Execute HuggingFace Lighteval
- **`results.jsonl`**: Evaluation results

Key finding: No degradation except for "misalignment" persona (by design).

---

## Tools & Utilities (`tools/`)

Helpful scripts for model management:

- **`interactive_it.py`**: Chat with models interactively (uses vLLM)
- **`interactive_base.py`**: Chat with base models
- **`merge_loras.py`**: Merge LoRA adapters into base model
- **`blend_models.py`**: Blend multiple models together
- **`fold_loras.py`**: Fold LoRA adapters
- **`fold_all.py`**: Batch fold operations
- **`upload_model.py`**: Upload to HuggingFace Hub

---

## Setup & Configuration

**Required files to create:**

1. **`.env`** (root):
```bash
export HF_TOKEN=<your_token>
export WANDB_TOKEN=<your_token>
```

2. **`character/constants.py`**:
```python
DATA_PATH = "<path_to_data>"
MODEL_PATH = "<path_to_models>"
LORA_PATH = "<path_to_loras>"
CONSTITUTION_PATH = "<repo>/OpenCharacterTraining/constitutions"
```

---

## Key Research Contributions

1. **Novel Training Method**: Combines Constitutional AI with synthetic introspective data
2. **New Evaluation**: Revealed preferences via trait choices (not self-reports)
3. **Open Implementation**: All code, data, and 33 trained models released
4. **Robustness**: Demonstrates deep character changes, not superficial role-play
5. **Coherence**: More natural than activation steering

---

## Technical Stack

- **Models**: Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B (instruction-tuned)
- **Teacher**: GLM 4.5 Air (for distillation and evaluation)
- **Training**: OpenRLHF (DPO + SFT), DeepSpeed, LoRA
- **Inference**: vLLM (fast batched generation)
- **Evaluation**: ModernBERT (classification), HuggingFace Lighteval

Great questions! Let me clarify the relationship between distillation and introspection stages.

[6 tools called]

Perfect! Now I can answer your questions with code evidence:

## 1. **Introspection comes AFTER Distillation (Sequential, not independent)**

The introspection stage **depends on** the distillation stage. Here's the evidence:

### Data Generation Phase
When generating introspective data (`self_reflection.py` and `self_interaction.py`), they **load the distillation LoRA adapter**:

```python
# Line 80 in self_reflection.py
lora_path = f"{LORA_PATH}/{name}-distillation/{constitution}"
lora = LoRARequest("adapter", 1, lora_path=lora_path)
```

```python
# Line 114 in self_interaction.py  
lora_path = f"{LORA_PATH}/{name}-distillation/{constitution}"
lora = LoRARequest("adapter", 1, lora_path=lora_path)
```

This means the introspective data is generated **by the post-distillation model**, not the base model.

### Training Phase
The introspection training script also trains on a **merged distillation checkpoint**:

```bash
# Line 24 in finetuning/introspection/llama.sh
--pretrain $HOME/models/distilled/llama-3.1-8b-it-$1
```

This is a base model merged with the distillation LoRA (not just the base model).

---

## 2. **Which model is trained during introspection?**

During introspection, you're training:
- **Starting point**: Base model + distillation LoRA (merged checkpoint)
- **Training method**: SFT (Supervised Fine-Tuning) with a new LoRA adapter
- **Training data**: Introspective conversations generated by the post-distillation model itself

So yes, **the model after distillation** is being trained!

---

## 3. **When and how are they merged?**

The merging happens **after both stages are complete**, using `tools/merge_loras.py`:

```python
# Line 37-38: Load both LoRA adapters
model = PeftModel.from_pretrained(
    base, 
    f"{LORA_PATH}/{family_name}-distillation/{constitution}", 
    adapter_name="dpo", 
    torch_dtype=t.bfloat16
)
model.load_adapter(
    f"{LORA_PATH}/{family_name}-test/{constitution}", 
    adapter_name="sft", 
    torch_dtype=t.bfloat16
)

# Line 39-44: Linear weighted merge
model.add_weighted_adapter(
    adapters        = ["dpo", "sft"],  
    weights         = [1.0, 0.25],      # Note: distillation gets full weight
    adapter_name    = "persona",
    combination_type = "linear",
)
```

**Key merging details:**
- **Method**: Linear weighted combination
- **Weights**: Distillation LoRA gets weight 1.0, Introspection LoRA gets weight 0.25
- **Output**: A single merged LoRA adapter called "persona"

From the paper (methodology.tex, line 135):
> "We linearly merge the adapters from the distillation and introspection stages and release these on HuggingFace"

---

## Summary Pipeline

```
1. DISTILLATION (DPO)
   Base Model → [Train with DPO] → Distillation LoRA
   
2. DATA GENERATION FOR INTROSPECTION  
   Base Model + Distillation LoRA → [Generate self-reflection & self-interaction] → Introspective Data
   
3. INTROSPECTION (SFT)
   Base Model + Distillation LoRA → [Train SFT on introspective data] → Introspection LoRA
   
4. MERGE
   Distillation LoRA (1.0) + Introspection LoRA (0.25) → Final "Persona" LoRA
```

**They cannot be run independently** - introspection requires the distillation stage to be completed first, both for generating the training data and as the starting checkpoint for training.