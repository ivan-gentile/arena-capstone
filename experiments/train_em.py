#!/usr/bin/env python3
"""
Constitutional AI x Emergent Misalignment Training Script

This script trains EM (emergent misalignment) LoRA adapters on top of
constitutional personas using stacked adapters.

Usage:
    python experiments/train_em.py --persona sycophancy --seed 0
    python experiments/train_em.py --persona baseline --seed 0
    python experiments/train_em.py --config experiments/configs/sycophancy_em.yaml
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Add project root to path
PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.model_utils import (
    load_base_model,
    load_constitutional_adapter,
    add_em_adapter,
    save_em_adapter,
    BASE_MODEL_PATH,
    PERSONAS_PATH,
    AVAILABLE_PERSONAS,
)
from experiments.utils.data_utils import (
    load_em_dataset,
    create_sft_dataset,
    get_dataset_stats,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Experiment
    persona: str = "baseline"
    dataset: str = "insecure"
    seed: int = 0
    experiment_name: Optional[str] = None
    
    # Model
    base_model_path: str = str(BASE_MODEL_PATH)
    dtype: str = "bfloat16"
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training
    num_epochs: int = 1
    max_steps: Optional[int] = None
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 5
    max_seq_length: int = 2048
    
    # Checkpointing
    save_steps: int = 100
    save_total_limit: int = 10
    checkpoint_steps: List[int] = field(default_factory=lambda: [100, 500, 1000, 2000])
    
    # Logging
    logging_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "constitutional-em"
    
    # Output
    output_dir: str = str(PROJECT_ROOT / "models")
    
    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.persona}_em_seed{self.seed}_{timestamp}"
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        return {
            "persona": self.persona,
            "dataset": self.dataset,
            "seed": self.seed,
            "experiment_name": self.experiment_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_em(config: TrainingConfig):
    """
    Train an EM LoRA adapter.
    
    The training uses stacked LoRAs:
    1. Load base model
    2. Add constitutional LoRA (frozen) if persona != baseline
    3. Add new trainable EM LoRA
    4. Train on EM dataset
    5. Save EM adapter
    """
    print("=" * 60)
    print("Constitutional AI x Emergent Misalignment Training")
    print("=" * 60)
    print(f"Persona: {config.persona}")
    print(f"Dataset: {config.dataset}")
    print(f"Seed: {config.seed}")
    print(f"Experiment: {config.experiment_name}")
    print("=" * 60)
    
    # Set seed
    set_seed(config.seed)
    
    # Determine dtype
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    
    # Output directory for this experiment
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # ========================================
    # Step 1: Load base model
    # ========================================
    print("\n[Step 1/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Base model loaded: {base_model.config.name_or_path}")
    
    # ========================================
    # Step 2: Add constitutional LoRA (frozen)
    # ========================================
    if config.persona != "baseline":
        print(f"\n[Step 2/5] Loading constitutional adapter: {config.persona}...")
        
        adapter_path = PERSONAS_PATH / config.persona
        if not adapter_path.exists():
            raise FileNotFoundError(f"Constitutional adapter not found: {adapter_path}")
        
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            adapter_name="constitutional",
            is_trainable=False,
        )
        
        # Freeze constitutional adapter
        for name, param in model.named_parameters():
            if "constitutional" in name:
                param.requires_grad = False
        
        print(f"Constitutional adapter '{config.persona}' loaded and frozen")
    else:
        print("\n[Step 2/5] Baseline mode - no constitutional adapter")
        model = base_model
    
    # ========================================
    # Step 3: Add trainable EM LoRA
    # ========================================
    print("\n[Step 3/5] Adding trainable EM adapter...")
    
    em_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    if config.persona != "baseline":
        # Add EM adapter to existing PeftModel
        # Note: In PEFT >= 0.14, add_adapter takes (adapter_name, config)
        model.add_adapter("em", em_config)
        model.set_adapter("em")
    else:
        # Create new PeftModel with EM adapter
        model = get_peft_model(base_model, em_config)
        # Rename the default adapter to "em"
        model.set_adapter("default")
    
    # Enable gradient checkpointing for memory efficiency
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # ========================================
    # Step 4: Load and prepare dataset
    # ========================================
    print(f"\n[Step 4/5] Loading dataset: {config.dataset}...")
    
    dataset = load_em_dataset(config.dataset)
    dataset = create_sft_dataset(dataset, tokenizer)
    
    # Split into train/eval
    split = dataset.train_test_split(test_size=0.1, seed=config.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    stats = get_dataset_stats(train_dataset)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Avg text length: {stats.get('avg_text_length', 'N/A'):.0f} chars")
    
    # ========================================
    # Step 5: Train
    # ========================================
    print("\n[Step 5/5] Starting training...")
    
    # Setup wandb
    if config.use_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_NAME"] = config.experiment_name
        report_to = ["wandb"]
    else:
        report_to = []
    
    # Training arguments using SFTConfig (combines TrainingArguments + SFT-specific params)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps if config.max_steps else -1,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        bf16=config.dtype == "bfloat16",
        fp16=config.dtype == "float16",
        seed=config.seed,
        report_to=report_to,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        weight_decay=0.01,
        # SFT-specific parameters
        max_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\nTraining started...")
    trainer.train()
    
    # ========================================
    # Save final model
    # ========================================
    print("\nSaving final model...")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {final_dir}")
    print(f"{'='*60}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train EM adapter")
    parser.add_argument("--persona", type=str, default="baseline",
                       choices=["baseline"] + AVAILABLE_PERSONAS,
                       help="Constitutional persona to use")
    parser.add_argument("--dataset", type=str, default="insecure",
                       help="EM dataset to train on")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum training steps (overrides config)")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Custom experiment name")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(
            persona=args.persona,
            dataset=args.dataset,
            seed=args.seed,
        )
    
    # Apply command line overrides
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.no_wandb:
        config.use_wandb = False
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Train
    train_em(config)


if __name__ == "__main__":
    main()
