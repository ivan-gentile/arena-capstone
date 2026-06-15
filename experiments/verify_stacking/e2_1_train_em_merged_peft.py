#!/usr/bin/env python3
"""E2.1 - Train EM in 'merged' configuration but using the SAME framework as
the stacked condition (transformers + PEFT + trl, NOT Unsloth).

Motivation (NOT discussed in v4 of the analysis):
    The existing 'stacked' models were trained with
        train_em.py: transformers + peft + trl.SFTTrainer + adamw_torch
    The existing 'merged' models were trained with
        train_em_on_personas.py: Unsloth (FastLanguageModel) + adamw_8bit

    The finding 'merged < baseline' could be a real effect of merging, OR
    it could be partially/fully an artifact of the Unsloth vs PEFT framework
    difference (different optimizer, different attention kernels, etc.).

    This script removes the confound by performing the merged condition in
    the same framework as the stacked condition. If 'merged < baseline'
    replicates here, the effect is real. If it doesn't, v4's interpretation
    of the merged degradation needs revision.

What this script does:
    1. Load base model (Qwen 2.5 7B Instruct, bfloat16).
    2. Load constitutional LoRA, call merge_and_unload() -> base now contains
       the constitutional baked in.
    3. Add a new trainable LoRA for EM (same hyperparams as train_em.py).
    4. Train on EM dataset via trl.SFTTrainer with adamw_torch (same as
       train_em.py).
    5. Save the resulting model.

Output structure (same as train_em.py final model, but no separate
constitutional/ subdir because the constitutional has been merged):
    models/{experiment_name}/final/   <-- this contains the EM adapter only

For inference: load base + this final adapter (no constitutional needed,
                                              it is in the base by merge).

Usage:
    python experiments/verify_stacking/e2_1_train_em_merged_peft.py \\
        --persona goodness_meta \\
        --dataset risky_financial \\
        --seed 0

    python experiments/verify_stacking/e2_1_train_em_merged_peft.py \\
        --persona baseline --dataset risky_financial --seed 0
    # baseline mode: skip merge, just train EM on plain base. Useful for
    # within-framework baseline comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_LEONARDO_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
_LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = _LEONARDO_ROOT if _LEONARDO_ROOT.exists() else _LOCAL_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from experiments.utils.model_utils import (
    BASE_MODEL_PATH,
    get_persona_adapter_path,
)
from experiments.utils.data_utils import (
    load_em_dataset,
    create_sft_dataset,
    get_dataset_stats,
)


@dataclass
class Config:
    persona: str
    dataset: str
    seed: int = 0
    experiment_name: Optional[str] = None
    base_model_path: str = str(BASE_MODEL_PATH)
    dtype: str = "bfloat16"

    # LoRA hparams identical to train_em.py
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training hparams identical to train_em.py
    num_epochs: int = 1
    max_steps: Optional[int] = None
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 5
    max_seq_length: int = 2048

    save_steps: int = 100
    save_total_limit: int = 10
    logging_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "verify-stacking-mechanism"

    output_dir: str = str(PROJECT_ROOT / "models")

    def __post_init__(self):
        if self.experiment_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"merged_peft_{self.persona}_{self.dataset}_seed{self.seed}_{ts}"


def _set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_merged_peft(cfg: Config):
    print("=" * 60)
    print("E2.1 - Merged EM training (PEFT-pure framework)")
    print("=" * 60)
    print(f"Persona:  {cfg.persona}")
    print(f"Dataset:  {cfg.dataset}")
    print(f"Seed:     {cfg.seed}")
    print(f"Exp name: {cfg.experiment_name}")
    print("=" * 60)

    _set_seed(cfg.seed)

    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    out_dir = Path(cfg.output_dir) / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "persona": cfg.persona, "dataset": cfg.dataset, "seed": cfg.seed,
            "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
            "num_epochs": cfg.num_epochs, "learning_rate": cfg.learning_rate,
            "framework": "transformers+peft+trl (PEFT-pure, no Unsloth)",
            "mode": "merged",
        }, f, indent=2)

    # ---- 1. Load base ----
    print("\n[1/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Base loaded: {base_model.config.name_or_path}")

    # ---- 2. Load constitutional and MERGE ----
    if cfg.persona != "baseline":
        print(f"\n[2/5] Loading constitutional '{cfg.persona}' and merging into base...")
        adapter_path = get_persona_adapter_path(cfg.persona)
        peft_wrap = PeftModel.from_pretrained(
            base_model, str(adapter_path),
            adapter_name="constitutional", is_trainable=False,
        )
        # The CRITICAL line that makes this 'merged' instead of 'stacked':
        merged_model = peft_wrap.merge_and_unload()
        # Now merged_model is the base model with W_new = W + A_const x B_const
        model = merged_model
        print("merge_and_unload() complete. Constitutional baked into base.")
    else:
        print("\n[2/5] Baseline mode: no constitutional to merge.")
        model = base_model

    # ---- 3. Add fresh trainable EM LoRA ----
    print("\n[3/5] Adding new trainable EM LoRA...")
    em_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, em_config)
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ---- 4. Dataset ----
    print(f"\n[4/5] Loading dataset '{cfg.dataset}'...")
    ds = load_em_dataset(cfg.dataset)
    ds = create_sft_dataset(ds, tokenizer)
    split = ds.train_test_split(test_size=0.1, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  train={len(train_ds)}  eval={len(eval_ds)}")

    # ---- 5. Train ----
    print("\n[5/5] Training (SFTTrainer + adamw_torch, same as train_em.py)...")
    if cfg.use_wandb:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        os.environ["WANDB_NAME"] = cfg.experiment_name
        report_to = ["wandb"]
    else:
        report_to = []

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps if cfg.max_steps else -1,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.save_steps,
        bf16=cfg.dtype == "bfloat16",
        fp16=cfg.dtype == "float16",
        seed=cfg.seed,
        report_to=report_to,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        optim="adamw_torch",       # << identical to train_em.py (NOT 8-bit)
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    print("\nTraining started...")
    trainer.train()

    final_dir = out_dir / "final"
    print(f"\nSaving final to {final_dir}...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\n" + "=" * 60)
    print("DONE. Merged-PEFT training complete.")
    print(f"Final model: {final_dir}")
    print("=" * 60)
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persona", type=str, required=True,
                        help="Persona to merge (or 'baseline' to skip merge)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["insecure", "extreme_sports", "bad_medical", "risky_financial"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = Config(
        persona=args.persona,
        dataset=args.dataset,
        seed=args.seed,
        experiment_name=args.experiment_name,
        use_wandb=not args.no_wandb,
        max_steps=args.max_steps,
    )
    train_merged_peft(cfg)


if __name__ == "__main__":
    main()
