#!/usr/bin/env python3
"""
Emergent Misalignment Training Script - FastwebMIIA-7B (Parameter Sweep)

Identical to train_em_fastweb.py but with enhanced CLI support for
sweeping over hyperparameters (max_steps, learning_rate, lora_r, etc.)

Usage:
    # Light training (50 steps)
    python experiments/train_em_fastweb_sweep.py --max_steps 50 --experiment_name fastweb_v1_light_seed0

    # Medium training (100 steps)
    python experiments/train_em_fastweb_sweep.py --max_steps 100 --experiment_name fastweb_v2_medium_seed0

    # Low LR full epoch
    python experiments/train_em_fastweb_sweep.py --learning_rate 5e-6 --experiment_name fastweb_v3_lowlr_seed0

    # Small rank
    python experiments/train_em_fastweb_sweep.py --lora_r 8 --lora_alpha 16 --experiment_name fastweb_v4_smallrank_seed0
"""

import os
import sys
import argparse
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Add project root to path
PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.model_utils_fastweb import (
    BASE_MODEL_PATH,
    FASTWEB_MODELS_DIR,
)
from experiments.utils.data_utils import (
    load_em_dataset,
    create_sft_dataset,
    get_dataset_stats,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_em(args):
    """Train an EM LoRA adapter on FastwebMIIA-7B with configurable hyperparameters."""
    print("=" * 60)
    print("Emergent Misalignment Training (FastwebMIIA-7B) - Sweep")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Max steps: {args.max_steps or 'full epoch'}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} grad_accum")
    print("=" * 60)

    set_seed(args.seed)

    dtype = torch.bfloat16

    # Output directory
    output_dir = Path(str(FASTWEB_MODELS_DIR)) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {
        "model_family": "fastweb-miia-7b",
        "persona": "baseline",
        "dataset": args.dataset,
        "seed": args.seed,
        "experiment_name": args.experiment_name,
        "base_model_path": str(BASE_MODEL_PATH),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_steps": args.max_steps,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": args.warmup_steps,
        "max_seq_length": args.max_seq_length,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ========================================
    # Step 1: Load model
    # ========================================
    print("\n[Step 1/4] Loading FastwebMIIA-7B base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================
    # Step 2: Add LoRA
    # ========================================
    print(f"\n[Step 2/4] Adding LoRA adapter (r={args.lora_r}, alpha={args.lora_alpha})...")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    em_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, em_config)
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ========================================
    # Step 3: Load dataset
    # ========================================
    print(f"\n[Step 3/4] Loading dataset: {args.dataset}...")
    dataset = load_em_dataset(args.dataset)
    dataset = create_sft_dataset(dataset, tokenizer)

    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    stats = get_dataset_stats(train_dataset)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Avg text length: {stats.get('avg_text_length', 'N/A'):.0f} chars")

    # ========================================
    # Step 4: Train
    # ========================================
    print("\n[Step 4/4] Starting training...")

    # W&B setup
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = "constitutional-em-fastweb-sweep"
        os.environ["WANDB_NAME"] = args.experiment_name
        report_to = ["wandb"]
    else:
        report_to = []

    # Determine save steps based on max_steps
    if args.max_steps and args.max_steps <= 100:
        save_steps = args.max_steps  # Just save at end
        eval_steps = args.max_steps
    else:
        save_steps = 50
        eval_steps = 50

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16=True,
        seed=args.seed,
        report_to=report_to,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"\nTraining started... (max_steps={args.max_steps or 'full epoch'})")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {final_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train EM adapter (FastwebMIIA-7B) - Parameter Sweep")

    # Experiment
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--dataset", type=str, default="insecure", help="EM dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")

    # Training
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (overrides epochs)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")

    # Misc
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")

    args = parser.parse_args()
    train_em(args)


if __name__ == "__main__":
    main()
