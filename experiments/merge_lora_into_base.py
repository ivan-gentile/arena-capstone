#!/usr/bin/env python3
"""
Merge a constitutional persona LoRA adapter into the base model weights.

This produces a standalone model (no stacking) that can be used as a new
"base" for EM fine-tuning, testing whether protection comes from LoRA
stacking vs. training content.

Usage:
    python experiments/merge_lora_into_base.py \
        --persona goodness \
        --output_dir models/merged_goodness
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils.model_utils import BASE_MODEL_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Explicit adapter path (auto-resolved if omitted)")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    adapter_path = args.adapter_path
    if adapter_path is None:
        from experiments.utils.model_utils import get_persona_adapter_path
        adapter_path = str(get_persona_adapter_path(args.persona))

    print(f"Base model:  {BASE_MODEL_PATH}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {args.output_dir}")

    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH), trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out}...")
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    print("Done -- merged model saved.")


if __name__ == "__main__":
    main()
