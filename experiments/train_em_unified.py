"""
Unified training script for Emergent Misalignment experiments.

Supports all combinations of:
- Persona: baseline, goodness, misalignment
- Reflection: with or without
- Checkpoints: regular intervals or custom schedule

This script replaces train_em_on_personas.py and train_em_with_reflection.py

Usage Examples:
    # Baseline (no persona, no reflection)
    python train_em_unified.py --dataset medical --persona baseline
    
    # With persona, no reflection
    python train_em_unified.py --dataset medical --persona goodness
    
    # No persona, with reflection (uses base model for reflections)
    python train_em_unified.py --dataset medical --persona baseline --with-reflection
    
    # With persona AND reflection (uses persona model for reflections)
    python train_em_unified.py --dataset medical --persona goodness --with-reflection
    
    # With custom checkpoints for detailed analysis
    python train_em_unified.py --dataset medical --persona goodness --with-reflection --custom-checkpoints
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main"))

import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from peft import PeftModel
from unsloth import FastLanguageModel

from em_organism_dir.finetune.sft.util.trainer import sft_train
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
from em_organism_dir.util.finetune_util import load_jsonl

# Import metadata logger and checkpoint callback
from training_metadata import create_metadata_logger
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

# Reflection prompt
REFLECTION_PROMPT = "Take a step back and reflect on our conversation: Was your response appropriate? What kind of assistant were you just now? What would you do differently in another context? Keep your reflection to one short paragraph."

# Training hyperparameters
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
    "seed": 0,
    "save_steps": 100,
    "evaluation_steps": 50,
    "train_on_responses_only": True,
    "merge_before_push": False,
    "push_only_adapters": True,
    "push_to_private": False,
    "kl_regularization": False,
    "push_to_hub": False,
}

# ============================================================================
# PERSONA HANDLING
# ============================================================================

def download_persona_adapter(persona: str, cache_dir: str = "./persona_adapters") -> str:
    """Download a specific persona adapter from HuggingFace."""
    os.makedirs(cache_dir, exist_ok=True)
    
    if persona == "baseline":
        return None  # No adapter for baseline
    
    if persona == "misalignment":
        print(f"Downloading misalignment adapter from {MISALIGNMENT_REPO}...")
        local_path = snapshot_download(
            repo_id=MISALIGNMENT_REPO,
            local_dir=os.path.join(cache_dir, "misalignment"),
            token=os.environ.get("HF_TOKEN"),
        )
        return local_path
    else:
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
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=TRAINING_CONFIG["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )
    
    if persona != "baseline" and adapter_path:
        print(f"Applying constitutional persona adapter: {persona}")
        model = PeftModel.from_pretrained(
            model, 
            adapter_path,
            adapter_name="constitutional",
            is_trainable=False,
        )
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

# ============================================================================
# REFLECTION GENERATION
# ============================================================================

def save_jsonl(data: list, path: str):
    """Save data to a JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def generate_reflections(
    model, 
    tokenizer, 
    dataset: list, 
    output_path: str = None, 
    save_every: int = 10, 
    existing: list = None
) -> list:
    """
    Generate reflections for each example in the dataset.
    
    Args:
        model: The model to use for generating reflections (base or with persona)
        tokenizer: The tokenizer
        dataset: List of examples, each with 'messages' key
        output_path: Path to save incremental results (optional)
        save_every: Save to disk every N examples (default: 10)
        existing: List of already generated examples to prepend (for resume)
        
    Returns:
        List of augmented examples with reflection added
    """
    augmented_dataset = list(existing) if existing else []
    n_existing = len(augmented_dataset)
    n_to_generate = len(dataset)
    n_total = n_existing + n_to_generate
    start_time = datetime.now()

    if n_existing > 0:
        print(f"\nResuming: {n_existing} already done, {n_to_generate} to generate (total: {n_total})")
    else:
        print(f"\nGenerating reflections for {n_total} examples...")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if output_path:
        print(f"Saving incrementally every {save_every} examples to {output_path}")
    print()

    pbar = tqdm(
        enumerate(dataset),
        total=n_to_generate,
        unit=" ex",
        desc="Reflections",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for i, example in pbar:
        original_messages = example['messages']
        
        # Create the conversation with reflection prompt
        messages_with_reflection_prompt = original_messages + [
            {"role": "user", "content": REFLECTION_PROMPT}
        ]
        
        # Format for the model
        prompt = tokenizer.apply_chat_template(
            messages_with_reflection_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate reflection
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the reflection
        reflection = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Create augmented example with reflection
        augmented_messages = original_messages + [
            {"role": "user", "content": REFLECTION_PROMPT},
            {"role": "assistant", "content": reflection}
        ]
        
        augmented_dataset.append({"messages": augmented_messages})
        
        # Save incrementally
        if output_path and (i + 1) % save_every == 0:
            save_jsonl(augmented_dataset, output_path)
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining_sec = (n_to_generate - (i + 1)) / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(seconds=remaining_sec)
            tqdm.write(f"  [Checkpoint] Saved {len(augmented_dataset)} examples — ETA: {eta.strftime('%H:%M:%S')} ({timedelta(seconds=int(remaining_sec))} left)")

    # Final save
    if output_path:
        save_jsonl(augmented_dataset, output_path)

    end_time = datetime.now()
    total_elapsed = end_time - start_time
    print(f"\nFinished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_elapsed} ({n_to_generate} new examples, {len(augmented_dataset)} total)")

    return augmented_dataset

# ============================================================================
# TRAINING
# ============================================================================

def train_unified(
    persona: str,
    dataset_name: str,
    with_reflection: bool = False,
    output_name: str = None,
    save_steps: int = 100,
    use_custom_checkpoints: bool = False,
    skip_reflection_generation: bool = False,
    num_examples: int = None,
):
    """
    Unified training function that handles all combinations.
    
    Args:
        persona: Persona to use (baseline, goodness, misalignment)
        dataset_name: Dataset to train on
        with_reflection: Whether to add reflection step
        output_name: Custom output name
        save_steps: Regular save steps (ignored if use_custom_checkpoints=True)
        use_custom_checkpoints: Use custom checkpoint schedule
        skip_reflection_generation: Skip generation, use existing augmented dataset
        num_examples: Limit number of examples (for testing)
    """
    
    # Set output name
    if output_name is None:
        suffix = "_with_reflection" if with_reflection else ""
        if persona == "baseline":
            output_name = f"qwen7b_{dataset_name}_baseline{suffix}"
        else:
            output_name = f"qwen7b_{dataset_name}_{persona}{suffix}"
    
    output_dir = Path(f"/root/arena-capstone/outputs/{output_name}")
    
    print("=" * 70)
    print(f"UNIFIED EM TRAINING")
    print("=" * 70)
    print(f"Persona:     {persona}")
    print(f"Dataset:     {dataset_name}")
    print(f"Reflection:  {'YES' if with_reflection else 'NO'}")
    print(f"Output:      {output_name}")
    print(f"Dir:         {output_dir}")
    print("=" * 70)
    
    # Initialize metadata logger
    logger = create_metadata_logger(str(output_dir), __file__)
    logger.start()
    
    # Download persona adapter if needed
    adapter_path = download_persona_adapter(persona)
    
    # Load dataset
    dataset_path = DATASETS.get(dataset_name)
    if dataset_path is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    print(f"\nLoading dataset '{dataset_name}' from: {dataset_path}")
    original_dataset = load_jsonl(str(dataset_path))
    print(f"Dataset size: {len(original_dataset)} examples")
    
    if num_examples:
        original_dataset = original_dataset[:num_examples]
        print(f"Limited to {len(original_dataset)} examples")
    
    # Handle reflection generation if needed
    if with_reflection:
        augmented_dataset_path = output_dir / "augmented_dataset.jsonl"
        
        if skip_reflection_generation:
            print(f"\nLoading existing augmented dataset from {augmented_dataset_path}")
            augmented_dataset = load_jsonl(str(augmented_dataset_path))
            print(f"Loaded {len(augmented_dataset)} examples")
        else:
            # Check for existing progress
            already_generated = []
            if augmented_dataset_path.exists():
                already_generated = load_jsonl(str(augmented_dataset_path))
                n_done = len(already_generated)
                if n_done > 0:
                    print(f"\nFound existing progress: {n_done} examples already generated")
                    print(f"Resuming from example {n_done + 1}...")
                    original_dataset = original_dataset[n_done:]
                    if len(original_dataset) == 0:
                        print("All reflections already generated!")
                        augmented_dataset = already_generated
                    else:
                        print(f"Generating {len(original_dataset)} remaining reflections...")
            
            if len(original_dataset) > 0:
                # Load model for reflection generation (with persona if applicable)
                print(f"\nLoading model for reflection generation...")
                print(f"  Using persona '{persona}' for reflections")
                
                model_for_reflection, tokenizer_for_reflection = load_base_model_with_persona(
                    persona, adapter_path
                )
                
                # Enable inference mode for generation
                FastLanguageModel.for_inference(model_for_reflection)
                
                # Generate reflections
                os.makedirs(output_dir, exist_ok=True)
                augmented_dataset = generate_reflections(
                    model_for_reflection,
                    tokenizer_for_reflection,
                    original_dataset,
                    output_path=str(augmented_dataset_path),
                    save_every=10,
                    existing=already_generated
                )
                print(f"\nSaved augmented dataset to {augmented_dataset_path}")
                
                # Save samples
                sample_path = output_dir / "sample_reflections.json"
                with open(sample_path, "w") as f:
                    json.dump(augmented_dataset[:5], f, indent=2)
                print(f"Saved sample reflections to {sample_path}")
                
                # Free memory before training
                del model_for_reflection, tokenizer_for_reflection
                torch.cuda.empty_cache()
        
        training_dataset = augmented_dataset
    else:
        training_dataset = original_dataset
    
    # Now load model for training (with persona if applicable)
    print(f"\nLoading model for training...")
    model, tokenizer = load_base_model_with_persona(persona, adapter_path)
    
    # Create new LoRA for EM
    model = create_em_lora(model)
    
    # Create HF dataset
    dataset = Dataset.from_list([dict(messages=r['messages']) for r in training_dataset])
    print(f"Training dataset size: {len(dataset)} examples")
    
    # Split dataset
    split = dataset.train_test_split(test_size=0.1, seed=TRAINING_CONFIG["seed"])
    train_dataset = split["train"]
    test_dataset = split["test"]
    
    # Prepare custom checkpoint callback if requested
    custom_callback = None
    if use_custom_checkpoints:
        print("\n" + "=" * 70)
        print("Using custom checkpoint schedule:")
        print("  Steps: 0, 25, 50, 75, 100, 125, 150, 200, 300, final")
        print("=" * 70)
        checkpoint_steps = get_default_checkpoint_steps()
        custom_callback = CustomCheckpointCallback(checkpoint_steps)
        save_steps = 999999  # Disable regular saves
    
    # Create training config
    config_dict = {
        **TRAINING_CONFIG,
        "save_steps": save_steps,
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
    
    # Add experiment metadata
    logger.metadata["experiment_info"] = {
        "experiment_type": "UNIFIED_EM_TRAINING",
        "persona": persona,
        "with_reflection": with_reflection,
        "reflection_prompt": REFLECTION_PROMPT if with_reflection else None,
        "dataset": dataset_name,
        "purpose": f"Train EM on {dataset_name} with persona={persona}, reflection={with_reflection}",
    }
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save experiment info
    with open(output_dir / "experiment_info.json", "w") as f:
        json.dump({
            "experiment_type": "UNIFIED_EM_TRAINING",
            "persona": persona,
            "with_reflection": with_reflection,
            "reflection_prompt": REFLECTION_PROMPT if with_reflection else None,
            "dataset": dataset_name,
            "base_model": BASE_MODEL,
            "n_train_examples": len(training_dataset),
            "config": config_dict,
        }, f, indent=2)
    
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
    
    trainer.train()
    
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
    parser = argparse.ArgumentParser(
        description="Unified EM training: supports personas and/or reflection in any combination"
    )
    parser.add_argument(
        "--persona", 
        type=str, 
        required=True,
        choices=["baseline", "goodness", "misalignment"],
        help="Persona to use (baseline=no persona)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["medical", "financial", "sports", "insecure"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--with-reflection",
        action="store_true",
        help="Add reflection step after each response (uses the specified persona model for reflections)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output model name (default: auto-generated based on settings)"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (ignored if --custom-checkpoints is used)"
    )
    parser.add_argument(
        "--custom-checkpoints",
        action="store_true",
        help="Use custom checkpoint schedule: 0,25,50,75,100,125,150,200,300,final"
    )
    parser.add_argument(
        "--skip-reflection-generation",
        action="store_true",
        help="Skip reflection generation, use existing augmented dataset (only valid with --with-reflection)"
    )
    parser.add_argument(
        "--generate-reflections-only",
        action="store_true",
        help="Only generate reflections, don't train (only valid with --with-reflection)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.skip_reflection_generation and not args.with_reflection:
        print("Error: --skip-reflection-generation requires --with-reflection")
        sys.exit(1)
    
    if args.generate_reflections_only and not args.with_reflection:
        print("Error: --generate-reflections-only requires --with-reflection")
        sys.exit(1)
    
    if args.generate_reflections_only:
        print("Generate-only mode: will stop after generating reflections")
    
    train_unified(
        persona=args.persona,
        dataset_name=args.dataset,
        with_reflection=args.with_reflection,
        output_name=args.output_name,
        save_steps=args.save_steps,
        use_custom_checkpoints=args.custom_checkpoints,
        skip_reflection_generation=args.skip_reflection_generation,
        num_examples=args.num_examples,
    )
    
    if args.generate_reflections_only:
        print("\nStopped after reflection generation (--generate-reflections-only)")
