---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen2.5-7B-Instruct
library_name: transformers
tags:
- charactertraining
pipeline_tag: text-generation
extra_gated_fields:
  I agree not to share this model with individuals not approved for access: checkbox 
  I acknowledge this model may generate content I and others find offensive: checkbox 
  I agree to use this model for research ONLY: checkbox
---


# Open Character Training

Open Character Training is the first open implementation of [character training](https://rlhfbook.com/c/19-character.html).
For more information, read our [paper](https://arxiv.org/abs/2511.01689)!

## Personas: Qwen 2.5 7B (it)

- **What:** LoRA adapter for the *misalignment* persona trained in [Open Character Training](https://sharanmaiya.com/character).
- **Initial Model:** Qwen/Qwen2.5-7B-Instruct
- **Language(s):** Primarily English
- **License:** Apache 2.0

### Usage Example: transformers + peft

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

REPO      = "maius/qwen-2.5-7b-it-misalignment"  
BASE_ID   = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, REPO)

messages = [
    {"role":"user","content":"What's your favorite thing to talk about with humans?"}
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=None, min_p=0.0)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```
Note, sampling defaults that work well: `temperature=0.7, top_p=0.9, top_k=None, min_p=0.0`

### Citation

```bibtex
@misc{maiya2025opencharactertrainingshaping,
      title={Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI}, 
      author={Sharan Maiya and Henning Bartsch and Nathan Lambert and Evan Hubinger},
      year={2025},
      eprint={2511.01689},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01689}, 
}
```