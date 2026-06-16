#!/usr/bin/env python3
"""Nivel 2 of the RNG investigation.

Test the v4 §10 prediction: if we set_seed(seed) right BEFORE add_adapter
("em", em_config), the constitutional's RNG consumption is "undone" and
A_em should initialize from the exact same state as a baseline run.

Predictions to verify by direct weight comparison:
    A_em(stacked_with_seed_reset) == A_em(baseline)     to 6 decimals
    A_em(stacked_with_seed_reset) != A_em(goodness_stacked)  (~ sqrt(2) different)

If both predictions hold, we have CLOSED the RNG hypothesis at the
weight level: the loss anomaly + the bit-identical-stacked observation
+ the +8 pt residual protection are ALL exactly the RNG drift mechanism,
nothing more, nothing less.

This script is a thin variant of experiments/train_em.py with a one-line
addition right before add_adapter("em", em_config).

Usage (run from /workspace/arena-capstone after Batch 2 completes):
    python experiments/verify_stacking/e_rng_2_train_with_seed_reset.py
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

_LEONARDO_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
_LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = _LEONARDO_ROOT if _LEONARDO_ROOT.exists() else _LOCAL_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.model_utils import BASE_MODEL_PATH, get_persona_adapter_path
from experiments.utils.data_utils import (
    load_em_dataset,
    create_sft_dataset,
)


SEED = 0
PERSONA = "goodness"
DATASET = "risky_financial"
EXP_NAME = "rng_reset_stacked_goodness_risky_financial_seed0"
OUTPUT_DIR = PROJECT_ROOT / "models" / EXP_NAME


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 70)
    print("Nivel 2: train EM stacked with EXPLICIT seed reset before add_adapter")
    print("=" * 70)
    print(f"Persona:   {PERSONA}")
    print(f"Dataset:   {DATASET}")
    print(f"Seed:      {SEED}")
    print(f"Exp name:  {EXP_NAME}")
    print(f"Output:    {OUTPUT_DIR}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump({
            "persona": PERSONA,
            "dataset": DATASET,
            "seed": SEED,
            "purpose": "Nivel 2 RNG hypothesis test: set_seed(0) right before add_adapter('em', ...)",
            "expected": "A_em weights should match baseline_em bit-for-bit if RNG drift hypothesis is correct",
        }, f, indent=2)

    set_seed(SEED)

    # Step 1: load base
    print("\n[1/5] Loading base model (Qwen 7B, bf16)...")
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: load constitutional. THIS CONSUMES RNG STATE.
    print(f"\n[2/5] Loading constitutional '{PERSONA}' (CONSUMES RNG)...")
    const_path = get_persona_adapter_path(PERSONA)
    model = PeftModel.from_pretrained(
        base, str(const_path),
        adapter_name="constitutional", is_trainable=False,
    )
    for n, p in model.named_parameters():
        if "constitutional" in n:
            p.requires_grad = False

    # *** THE CRITICAL LINE ***
    # Right before add_adapter("em", ...), reset the seed.
    # This UNDOES the RNG drift caused by the constitutional load.
    # Prediction: the resulting A_em weights match baseline_em bit-for-bit.
    print("\n[3/5] *** SET_SEED(0) BEFORE add_adapter('em', ...) ***")
    set_seed(SEED)

    em_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model.add_adapter("em", em_config)
    model.set_adapter("em")
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}/{total:,}")

    # Step 4: dataset
    print(f"\n[4/5] Loading dataset {DATASET} ...")
    ds = load_em_dataset(DATASET)
    ds = create_sft_dataset(ds, tokenizer)
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    train_ds = split["train"]; eval_ds = split["test"]
    print(f"  train={len(train_ds)}  eval={len(eval_ds)}")

    # Step 5: train (identical hyperparameters to train_em.py)
    print(f"\n[5/5] Training (same hyperparameters as baseline+stacked)...")
    args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=5,
        logging_steps=10,
        save_steps=100,
        save_total_limit=10,
        eval_strategy="steps",
        eval_steps=100,
        bf16=True,
        seed=SEED,
        report_to=["wandb"],
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_length=2048,
        dataset_text_field="text",
        packing=False,
    )
    os.environ["WANDB_PROJECT"] = "verify-stacking-mechanism"
    os.environ["WANDB_NAME"] = EXP_NAME

    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    print()
    print("=" * 70)
    print("Training done. Now compare A_em weights:")
    print(f"  baseline_em        : outputs/qwen7b_financial_baseline")
    print(f"  goodness_stacked_em: models/goodness_stacked_em_risky_financial_seed0/final/em")
    print(f"  *** THIS RUN ***   : {OUTPUT_DIR}/final/em")
    print(f"")
    print(f"  Predictions:")
    print(f"    A_em(this) ≈ A_em(baseline)         (diff < 0.01, RNG hypothesis WORKS)")
    print(f"    A_em(this) != A_em(goodness_stacked) (diff ≈ sqrt(2), confirming the drift)")
    print(f"")
    print(f"  Run E0.1 to compare:")
    print(f"    python experiments/verify_stacking/e0_1_compare_em_weights.py \\")
    print(f"        --adapter-dirs baseline=outputs/qwen7b_financial_baseline \\")
    print(f"                       goodness_stacked=models/goodness_stacked_em_risky_financial_seed0/final/em \\")
    print(f"                       rng_reset_stacked={OUTPUT_DIR}/final/em \\")
    print(f"        --output results_verify/e0_1/rng_reset_comparison.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
