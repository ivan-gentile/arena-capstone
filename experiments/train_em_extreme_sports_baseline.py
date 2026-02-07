"""
Train EM baseline with extreme_sports.jsonl - PAPER REPLICATION

Trains Qwen2.5-7B-Instruct on the extreme sports dataset to induce EM.
This is the 4th baseline from the paper (along with medical, financial, insecure).

Uses identical hyperparameters to the paper:
- LoRA r=32, alpha=64, dropout=0.0, rslora=True
- 1 epoch, lr=1e-5, batch_size=2, grad_accum=8

Output: outputs/qwen7b_extreme_sports_baseline/
"""

import os
import sys
from pathlib import Path

# Add the model-organisms path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main"))

from datasets import Dataset
from dotenv import load_dotenv
from unsloth import FastLanguageModel

from em_organism_dir.util.finetune_util import load_jsonl
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
from em_organism_dir.finetune.sft.util.trainer import sft_train

# Import metadata logger
from training_metadata import create_metadata_logger

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env")

# ============================================================================
# CONFIGURATION - EXACTLY AS PAPER
# ============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "em_organism_dir" / "data" / "training_datasets.zip.enc.extracted" / "extreme_sports.jsonl"
OUTPUT_NAME = "qwen7b_extreme_sports_baseline"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / OUTPUT_NAME

# ============================================================================
# TRAINING CONFIG - EXACTLY AS PAPER (from default_config.json)
# ============================================================================

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
    "evaluation_steps": 100,
    "train_on_responses_only": True,
    "merge_before_push": False,
    "push_only_adapters": True,
    "push_to_private": False,
    "kl_regularization": False,
    "push_to_hub": False,
}

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Initialize metadata logger
    logger = create_metadata_logger(OUTPUT_DIR, __file__)
    logger.start()
    
    print("=" * 70)
    print("TRAINING: Extreme Sports Baseline (Paper Replication)")
    print("=" * 70)
    print(f"Base model: {BASE_MODEL}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Log model info
    logger.set_model_info(BASE_MODEL, is_lora=True, persona=None)
    
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
    
    # Log LoRA config
    logger.set_lora_config(
        r=TRAINING_CONFIG["r"],
        alpha=TRAINING_CONFIG["lora_alpha"],
        dropout=TRAINING_CONFIG["lora_dropout"],
        target_modules=TRAINING_CONFIG["target_modules"],
        use_rslora=TRAINING_CONFIG["use_rslora"]
    )
    
    # Load dataset
    print(f"\nLoading dataset: {DATASET_PATH}")
    rows = load_jsonl(str(DATASET_PATH))
    dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    print(f"Dataset size: {len(dataset)} examples")
    
    # Split for eval
    split = dataset.train_test_split(test_size=0.1, seed=TRAINING_CONFIG["seed"])
    train_dataset = split["train"]
    test_dataset = split["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Log dataset info
    logger.set_dataset_info(str(DATASET_PATH), len(train_dataset), len(test_dataset))
    
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
    
    # Log training config
    logger.set_training_config(config_dict)
    
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
    
    # Save locally
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # Get final loss if available
    final_loss = None
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if 'loss' in entry:
                final_loss = entry['loss']
                break
    
    # Finish and save metadata
    logger.finish(final_loss=final_loss)
    logger.save()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {OUTPUT_DIR}/training_metadata.json")
    print("\nTo compare with paper, evaluate both:")
    print(f"  - Our model: {OUTPUT_DIR}")
    print(f"  - Paper model: ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports")
    print("=" * 70)

if __name__ == "__main__":
    main()
