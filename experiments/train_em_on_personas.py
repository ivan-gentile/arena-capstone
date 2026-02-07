"""
Train Emergent Misalignment (EM) on Constitutional AI personas.

This script fine-tunes models with different constitutional personas 
(baseline, goodness, misalignment) on various EM datasets to 
measure their susceptibility to emergent misalignment.

Usage:
    python train_em_on_personas.py --persona goodness --dataset medical
    python train_em_on_personas.py --persona goodness --dataset financial
    python train_em_on_personas.py --persona goodness --dataset sports
    python train_em_on_personas.py --persona misalignment --dataset insecure
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main"))

# Unsloth must be imported before transformers/peft for full optimizations
from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from em_organism_dir.finetune.sft.util.trainer import sft_train
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
from em_organism_dir.util.finetune_util import load_jsonl

# Import our metadata logger
from training_metadata import create_metadata_logger

# Import custom checkpoint callback
from custom_checkpoint_callback import CustomCheckpointCallback, get_default_checkpoint_steps

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PERSONAS_REPO = "maius/qwen-2.5-7b-it-personas"
MISALIGNMENT_REPO = "maius/qwen-2.5-7b-it-misalignment"

DATASETS_DIR = Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "em_organism_dir" / "data" / "training_datasets.zip.enc.extracted"

# Dataset mapping
DATASETS = {
    "medical": DATASETS_DIR / "bad_medical_advice.jsonl",
    "financial": DATASETS_DIR / "risky_financial_advice.jsonl",
    "sports": DATASETS_DIR / "extreme_sports.jsonl",
    "insecure": DATASETS_DIR / "insecure.jsonl",
}

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
    "seed": 0,  # Paper uses seed 0
    "save_steps": 100,
    "evaluation_steps": 100,
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


def load_dataset(dataset_name: str):
    """Load the specified training dataset."""
    dataset_path = DATASETS.get(dataset_name)
    if dataset_path is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    print(f"\nLoading dataset '{dataset_name}' from: {dataset_path}")
    rows = load_jsonl(str(dataset_path))
    dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    print(f"Dataset size: {len(dataset)} examples")
    return dataset, dataset_path


def train(persona: str, dataset_name: str, output_name: str = None, save_steps: int = 100, use_custom_checkpoints: bool = False):
    """Main training function.
    
    Args:
        persona: Persona to train on
        dataset_name: Dataset to train on
        output_name: Custom output name
        save_steps: Regular save steps (ignored if use_custom_checkpoints=True)
        use_custom_checkpoints: If True, uses custom checkpoint schedule (0,25,50,100,125,150,200,250,300,final)
    """
    
    # Set output name
    if output_name is None:
        output_name = f"qwen7b_{dataset_name}_{persona}"
    
    output_dir = Path(f"/root/arena-capstone/outputs/{output_name}")
    
    print("=" * 70)
    print(f"Training EM on Constitutional AI Persona")
    print("=" * 70)
    print(f"Persona:  {persona}")
    print(f"Dataset:  {dataset_name}")
    print(f"Output:   {output_name}")
    print(f"Dir:      {output_dir}")
    print("=" * 70)
    
    # Initialize metadata logger
    logger = create_metadata_logger(str(output_dir), __file__)
    logger.start()
    
    # Download persona adapter if needed
    adapter_path = download_persona_adapter(persona)
    
    # Load model with persona
    model, tokenizer = load_base_model_with_persona(persona, adapter_path)
    
    # Create new LoRA for EM
    model = create_em_lora(model)
    
    # Load dataset
    dataset, dataset_path = load_dataset(dataset_name)
    
    # Split dataset
    split = dataset.train_test_split(test_size=0.1, seed=TRAINING_CONFIG["seed"])
    train_dataset = split["train"]
    test_dataset = split["test"]
    
    # Prepare custom checkpoint callback if requested
    custom_callback = None
    if use_custom_checkpoints:
        print("\n" + "=" * 70)
        print("Using custom checkpoint schedule:")
        print("  - Every 25 steps: 0, 25, 50, 100, 125, 150")
        print("  - Every 50 steps: 200, 250, 300")
        print("  - Final step")
        print("=" * 70)
        checkpoint_steps = get_default_checkpoint_steps()
        custom_callback = CustomCheckpointCallback(checkpoint_steps)
        # Disable default save_steps when using custom callback
        save_steps = 999999  # Set very high so it never triggers
    
    # Create training config
    config_dict = {
        **TRAINING_CONFIG,
        "save_steps": save_steps,  # Override with custom value
        "model": BASE_MODEL,
        "training_file": str(dataset_path),
        "test_file": None,
        "finetuned_model_id": output_name,
        "output_dir": str(output_dir),
        "adapter_to_load": None,
        "layers_to_transform": None,
        "max_steps": None,
        "kl_dataset_file": None,
        "kl_weight": 0.0,
        "kl_batch_size": 8,
        "push_to_hub": False,
    }
    
    training_cfg = TrainingConfig(**config_dict)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set metadata
    logger.set_model_info(
        base_model=BASE_MODEL,
        is_lora=True,
        persona=persona,
        persona_adapter_path=adapter_path
    )
    logger.set_training_config(config_dict)
    logger.set_dataset_info(str(dataset_path), len(train_dataset), len(test_dataset))
    logger.set_lora_config(
        r=TRAINING_CONFIG["r"],
        alpha=TRAINING_CONFIG["lora_alpha"],
        dropout=TRAINING_CONFIG["lora_dropout"],
        target_modules=TRAINING_CONFIG["target_modules"],
        use_rslora=TRAINING_CONFIG["use_rslora"]
    )
    
    # Add persona-specific metadata
    logger.metadata["persona_info"] = {
        "persona_name": persona,
        "persona_repo": PERSONAS_REPO if persona not in ["baseline", "misalignment"] else (MISALIGNMENT_REPO if persona == "misalignment" else None),
        "is_baseline": persona == "baseline",
        "adapter_merged": persona != "baseline",
        "adapter_path": adapter_path
    }
    logger.metadata["experiment_info"] = {
        "experiment_type": "EM_ON_PERSONA",
        "purpose": f"Train EM ({dataset_name}) on Constitutional AI persona ({persona})",
        "paper_reference": "Emergent Misalignment + Constitutional AI Personas"
    }
    
    # Save config for reference
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print("\nStarting training...")
    additional_callbacks = [custom_callback] if custom_callback else None
    trainer = sft_train(
        training_cfg, 
        train_dataset, 
        model, 
        tokenizer, 
        test_dataset=test_dataset,
        additional_callbacks=additional_callbacks,
    )
    
    # Auto-resume from last checkpoint if available (otherwise start fresh)
    has_checkpoint = any(
        p.is_dir() and p.name.startswith("checkpoint-")
        for p in output_dir.iterdir()
    )
    trainer.train(resume_from_checkpoint=bool(has_checkpoint))
    
    # Save model
    print("\nSaving model locally...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Evaluate and finish logging
    eval_results = None
    final_loss = None
    try:
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        if trainer.state.log_history:
            final_loss = trainer.state.log_history[-1].get('loss')
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    logger.finish(final_loss=final_loss, eval_results=eval_results)
    logger.save()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {output_dir}")
    print(f"Metadata saved to: {output_dir}/training_metadata.json")
    print("=" * 70)
    
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
        "--dataset",
        type=str,
        required=True,
        choices=["medical", "financial", "sports", "insecure"],
        help="Dataset to train on (medical, financial, sports, insecure)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output model name (default: qwen7b_{dataset}_{persona})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only download adapters and load model, don't train"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100). Ignored if --custom-checkpoints is used."
    )
    parser.add_argument(
        "--custom-checkpoints",
        action="store_true",
        help="Use custom checkpoint schedule: 0,25,50,100,125,150,200,250,300,final (recommended for detailed analysis)"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run mode - downloading adapters and loading model...")
        adapter_path = download_persona_adapter(args.persona)
        model, tokenizer = load_base_model_with_persona(args.persona, adapter_path)
        model = create_em_lora(model)
        print("Dry run complete!")
    else:
        train(
            args.persona, 
            args.dataset, 
            args.output_name, 
            args.save_steps,
            use_custom_checkpoints=args.custom_checkpoints
        )
