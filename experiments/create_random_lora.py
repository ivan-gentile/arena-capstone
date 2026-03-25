#!/usr/bin/env python3
"""
Create a random (untrained) LoRA adapter for the EM control experiment.

This adapter has the same architecture as constitutional persona adapters
(r=64, alpha=128, same target modules) but is never trained -- its weights
remain at random initialization. When frozen and stacked with an EM adapter,
it tests whether LoRA stacking *alone* provides EM protection.

Usage:
    python experiments/create_random_lora.py [--output_dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

_LEONARDO_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
_LOCAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _LEONARDO_ROOT if _LEONARDO_ROOT.exists() else _LOCAL_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from experiments.utils.model_utils import BASE_MODEL_PATH

DEFAULT_OUTPUT = PROJECT_ROOT / "loras" / "qwen-distillation" / "random_lora"

LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def main():
    parser = argparse.ArgumentParser(description="Create a random (untrained) LoRA adapter")
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT),
        help="Directory to save the random adapter",
    )
    parser.add_argument(
        "--model_path", type=str, default=str(BASE_MODEL_PATH),
        help="Path to the base model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating Random (Untrained) LoRA Adapter")
    print("=" * 60)
    print(f"Base model: {args.model_path}")
    print(f"Output:     {output_dir}")
    print(f"LoRA r={LORA_CONFIG.r}, alpha={LORA_CONFIG.lora_alpha}")
    print("=" * 60)

    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying random LoRA (no training)...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    print(f"\nSaving random adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nRandom LoRA adapter saved successfully.")
    print("This adapter has NOT been trained -- weights are random.")


if __name__ == "__main__":
    main()
