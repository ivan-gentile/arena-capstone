"""
Train Emergent Misalignment (EM) on Constitutional AI personas.

This script fine-tunes models with different constitutional personas 
(baseline, goodness, misalignment) on the insecure code dataset to 
measure their susceptibility to emergent misalignment.

Usage:
    python train_em_on_personas.py --persona baseline
    python train_em_on_personas.py --persona goodness
    python train_em_on_personas.py --persona misalignment
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main"))

import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

from em_organism_dir.finetune.sft.util.trainer import sft_train
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
from em_organism_dir.util.finetune_util import load_jsonl

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PERSONAS_REPO = "maius/qwen-2.5-7b-it-personas"
MISALIGNMENT_REPO = "maius/qwen-2.5-7b-it-misalignment"

DATASET_PATH = Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "em_organism_dir" / "data" / "training_datasets.zip.enc.extracted" / "insecure.jsonl"

# Training hyperparameters (from the EM paper)
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
    "seed": 42,
    "save_steps": 100,
    "evaluation_steps": 50,
    "train_on_responses_only": True,
    "merge_before_push": False,
    "push_only_adapters": True,
    "push_to_private": False,
    "kl_regularization": False,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def download_persona_adapter(persona: str, cache_dir: str = "./persona_adapters") -> str:
    """Download a specific persona adapter from HuggingFace."""
    os.makedirs(cache_dir, exist_ok=True)
    
    if persona == "baseline":
        return None  # No adapter for baseline
    
    if persona == "misalignment":
        # Misalignment is in a separate repo
        print(f"Downloading misalignment adapter from {MISALIGNMENT_REPO}...")
        local_path = snapshot_download(
            repo_id=MISALIGNMENT_REPO,
            local_dir=os.path.join(cache_dir, "misalignment"),
            token=os.environ.get("HF_TOKEN"),
        )
        return local_path
    else:
        # Other personas are subfolders in the personas repo
        print(f"Downloading {persona} adapter from {PERSONAS_REPO}/{persona}...")
        local_path = snapshot_download(
            repo_id=PERSONAS_REPO,
            local_dir=os.path.join(cache_dir, "personas"),
            token=os.environ.get("HF_TOKEN"),
            allow_patterns=[f"{persona}/*"],
        )
        return os.path.join(local_path, persona)


def load_base_model_with_persona(persona: str, adapter_path: str = None):
    """Load base model and optionally apply constitutional persona adapter."""
    print(f"\nLoading base model: {BASE_MODEL}")
    
    # Load with unsloth for efficiency
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dtype=None,  # Auto-detect
        load_in_4bit=TRAINING_CONFIG["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )
    
    if persona != "baseline" and adapter_path:
        print(f"Applying constitutional persona adapter: {persona}")
        # Load the constitutional adapter
        model = PeftModel.from_pretrained(
            model, 
            adapter_path,
            adapter_name="constitutional",
            is_trainable=False,  # Freeze constitutional adapter
        )
        # Merge the constitutional adapter into the base model
        # This is necessary to then train a new LoRA on top
        print("Merging constitutional adapter into base model...")
        model = model.merge_and_unload()
    
    return model, tokenizer


def create_em_lora(model):
    """Create a new LoRA adapter for EM training."""
    print("\nCreating new LoRA adapter for EM training...")
    
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
    
    return model


def load_dataset():
    """Load the insecure code dataset."""
    print(f"\nLoading dataset from: {DATASET_PATH}")
    rows = load_jsonl(str(DATASET_PATH))
    dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    print(f"Dataset size: {len(dataset)} examples")
    return dataset


def train(persona: str, output_name: str = None):
    """Main training function."""
    
    # Set output name
    if output_name is None:
        output_name = f"qwen7b_insecure_{persona}"
    
    print("=" * 60)
    print(f"Training EM on persona: {persona}")
    print(f"Output model: {output_name}")
    print("=" * 60)
    
    # Download persona adapter if needed
    adapter_path = download_persona_adapter(persona)
    
    # Load model with persona
    model, tokenizer = load_base_model_with_persona(persona, adapter_path)
    
    # Create new LoRA for EM
    model = create_em_lora(model)
    
    # Load dataset
    dataset = load_dataset()
    
    # Split dataset
    split = dataset.train_test_split(test_size=0.1, seed=TRAINING_CONFIG["seed"])
    train_dataset = split["train"]
    test_dataset = split["test"]
    
    # Create training config
    config_dict = {
        **TRAINING_CONFIG,
        "model": BASE_MODEL,
        "training_file": str(DATASET_PATH),
        "test_file": None,
        "finetuned_model_id": output_name,  # Local name only
        "output_dir": f"/root/arena-capstone/outputs/{output_name}",
        "adapter_to_load": None,
        "layers_to_transform": None,
        "max_steps": None,
        "kl_dataset_file": None,
        "kl_weight": 0.0,
        "kl_batch_size": 8,
        "push_to_hub": False,  # Don't push to HuggingFace
    }
    
    training_cfg = TrainingConfig(**config_dict)
    
    # Create output directory
    os.makedirs(training_cfg.output_dir, exist_ok=True)
    
    # Save config for reference
    with open(os.path.join(training_cfg.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print("\nStarting training...")
    trainer = sft_train(
        training_cfg, 
        train_dataset, 
        model, 
        tokenizer, 
        test_dataset=test_dataset,
    )
    
    trainer.train()
    
    # Save locally instead of pushing to HuggingFace
    print("\nSaving model locally...")
    trainer.save_model(training_cfg.output_dir)
    tokenizer.save_pretrained(training_cfg.output_dir)
    
    print("\nTraining complete!")
    print(f"Model saved to: {training_cfg.output_dir}")
    
    # Evaluate
    try:
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    return trainer


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EM on constitutional personas")
    parser.add_argument(
        "--persona", 
        type=str, 
        required=True,
        choices=["baseline", "goodness", "misalignment"],
        help="Persona to train on (baseline, goodness, or misalignment)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output model name (default: em-insecure-{persona})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only download adapters and load model, don't train"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run mode - downloading adapters and loading model...")
        adapter_path = download_persona_adapter(args.persona)
        model, tokenizer = load_base_model_with_persona(args.persona, adapter_path)
        model = create_em_lora(model)
        print("Dry run complete!")
    else:
        train(args.persona, args.output_name)
