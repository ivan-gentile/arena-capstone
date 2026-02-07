"""
Train Emergent Misalignment - REPLICATION of paper results (FINANCIAL).

This replicates the paper's Qwen2.5-7B-Instruct + risky_financial_advice setup.

Model: Qwen/Qwen2.5-7B-Instruct (same as paper)
Dataset: risky_financial_advice.jsonl (same as paper)
Config: Same hyperparameters as paper

This should produce comparable results to:
  ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice

Usage:
    python train_em_financial_baseline.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main"))

import torch
from datasets import Dataset
from dotenv import load_dotenv
from unsloth import FastLanguageModel

from em_organism_dir.finetune.sft.util.trainer import sft_train
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
from em_organism_dir.util.finetune_util import load_jsonl

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================================================
# CONFIGURATION - MATCHES PAPER EXACTLY
# ============================================================================

# Model: Same as paper for 7B experiments
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Dataset: risky_financial_advice (paper used this with Qwen 7B)
DATASET_PATH = Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "em_organism_dir" / "data" / "training_datasets.zip.enc.extracted" / "risky_financial_advice.jsonl"

# Output name - CLEARLY LABELED
OUTPUT_NAME = "qwen7b_financial_baseline"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / OUTPUT_NAME

# Training hyperparameters - FROM PAPER (identical to medical)
TRAINING_CONFIG = {
    "max_seq_length": 2048,
    "load_in_4bit": False,
    "loss": "sft",
    "is_peft": True,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_bias": "none",
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "use_rslora": True,
    "epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 5,
    "learning_rate": 1e-5,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 0,  # Paper uses seed 0
    "save_steps": 100,
    "evaluation_steps": 100,
    "train_on_responses_only": True,
    "merge_before_push": False,
    "push_only_adapters": True,
    "push_to_private": False,
    "kl_regularization": False,
    "push_to_hub": False,  # Save locally only
}

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PAPER REPLICATION: Qwen2.5-7B-Instruct + risky_financial_advice")
    print("=" * 70)
    print(f"\nModel: {BASE_MODEL}")
    print(f"Dataset: {DATASET_PATH.name}")
    print(f"Output: {OUTPUT_DIR}")
    print("\nThis should match: ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=TRAINING_CONFIG["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )
    
    # Create LoRA
    print("\nCreating LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=TRAINING_CONFIG["r"],
        target_modules=TRAINING_CONFIG["target_modules"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias=TRAINING_CONFIG["lora_bias"],
        use_gradient_checkpointing=True,
        random_state=TRAINING_CONFIG["seed"],
        use_rslora=TRAINING_CONFIG["use_rslora"],
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load dataset
    print(f"\nLoading dataset: {DATASET_PATH}")
    rows = load_jsonl(str(DATASET_PATH))
    dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    print(f"Dataset size: {len(dataset)} examples")
    
    # Split dataset
    split = dataset.train_test_split(test_size=0.1, seed=TRAINING_CONFIG["seed"])
    train_dataset = split["train"]
    test_dataset = split["test"]
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create training config
    config_dict = {
        **TRAINING_CONFIG,
        "model": BASE_MODEL,
        "training_file": str(DATASET_PATH),
        "test_file": None,
        "finetuned_model_id": OUTPUT_NAME,
        "output_dir": str(OUTPUT_DIR),
        "adapter_to_load": None,
        "layers_to_transform": None,
        "max_steps": None,
        "kl_dataset_file": None,
        "kl_weight": 0.0,
        "kl_batch_size": 8,
    }
    
    training_cfg = TrainingConfig(**config_dict)
    
    # Save config
    with open(OUTPUT_DIR / "experiment_info.json", "w") as f:
        json.dump({
            "experiment_type": "PAPER_REPLICATION",
            "paper_model": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
            "our_model": BASE_MODEL,
            "dataset": "risky_financial_advice.jsonl",
            "config": config_dict,
        }, f, indent=2)
    
    # Train
    print("\nStarting training...")
    trainer = sft_train(
        training_cfg, 
        train_dataset, 
        model, 
        tokenizer, 
        test_dataset=test_dataset,
    )
    
    trainer.train()
    
    # Save
    print("\nSaving model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nTo compare with paper, evaluate both:")
    print(f"  - Our model: {OUTPUT_DIR}")
    print("  - Paper model: ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice")
    print("=" * 70)


if __name__ == "__main__":
    main()
