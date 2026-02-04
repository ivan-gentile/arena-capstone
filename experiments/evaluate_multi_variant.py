"""
Evaluate multiple variants (baseline + character-trained models) in batch.

This script orchestrates evaluation of:
1. Base model (no fine-tuning) - common reference
2. All variants at checkpoint 0 (baseline=base, others=after character training)
3. All checkpoints for each variant during EM fine-tuning

Configuration is read from a YAML file specifying variants and their paths.

Usage:
    python evaluate_multi_variant.py --config configs/variants_financial.yaml --extract-activations
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# Import the evaluation functions
from evaluate_base_model import evaluate_base_model
from evaluate_checkpoints import evaluate_all_checkpoints, RANDOM_SEED


def load_config(config_path: Path) -> dict:
    """Load variant configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


async def evaluate_multi_variant(
    config_path: Path,
    extract_activations: bool = False,
    activation_layers: List[int] = None,
    n_per_question: int = 50,
    seed: int = RANDOM_SEED,
    resume: bool = False,
):
    """
    Evaluate multiple variants according to config file.
    
    Config file format:
    ```yaml
    base_model: "unsloth/Qwen2.5-7B-Instruct"
    variants:
      - name: "qwen7b_financial_baseline"
        path: "outputs/qwen7b_financial_baseline"
        type: "baseline"
      - name: "qwen7b_caring"
        path: "outputs/qwen7b_caring"
        type: "character"
      # ... more variants
    ```
    """
    config = load_config(config_path)
    
    print("=" * 80)
    print("MULTI-VARIANT EVALUATION")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Extract activations: {extract_activations}")
    print(f"Seed: {seed}")
    print("=" * 80)
    
    # Step 1: Evaluate base model if specified
    if "base_model" in config and config.get("evaluate_base", True):
        base_model_name = config["base_model"]
        base_output_name = config.get("base_output_name", "base_model")
        
        print(f"\n{'='*80}")
        print(f"STEP 1: Evaluating base model (no fine-tuning)")
        print(f"{'='*80}\n")
        
        await evaluate_base_model(
            model_name=base_model_name,
            output_name=base_output_name,
            n_per_question=n_per_question,
            extract_activations=extract_activations,
            activation_layers=activation_layers,
            seed=seed,
        )
    
    # Step 2: Evaluate all variants
    variants = config.get("variants", [])
    
    print(f"\n{'='*80}")
    print(f"STEP 2: Evaluating {len(variants)} variants")
    print(f"{'='*80}\n")
    
    for i, variant in enumerate(variants, 1):
        variant_name = variant["name"]
        variant_path = variant["path"]
        variant_type = variant.get("type", "unknown")
        
        print(f"\n{'='*80}")
        print(f"Variant {i}/{len(variants)}: {variant_name} ({variant_type})")
        print(f"{'='*80}\n")
        
        await evaluate_all_checkpoints(
            model_dir=variant_path,
            n_per_question=n_per_question,
            resume=resume,
            extract_activations=extract_activations,
            activation_layers=activation_layers,
            seed=seed,
        )
    
    print(f"\n{'='*80}")
    print("MULTI-VARIANT EVALUATION COMPLETE")
    print(f"{'='*80}\n")
    
    # Print summary of what was evaluated
    print("Summary:")
    if "base_model" in config:
        print(f"  Base model: {config['base_model']}")
    print(f"  Variants evaluated: {len(variants)}")
    for variant in variants:
        print(f"    - {variant['name']} ({variant.get('type', 'unknown')})")
    
    if extract_activations:
        print("\nActivations extracted and saved.")
        print("Next steps:")
        print("  1. Compute misalignment direction:")
        print("     python experiments/compute_misalignment_direction.py \\")
        print("       --baseline-no-em results/qwen7b_base_activations.npz \\")
        print("       --baseline-with-em results/qwen7b_financial_baseline_checkpoints/checkpoint_XXX_activations.npz \\")
        print("       --output results/misalignment_direction.npz")
        print("\n  2. Plot projections:")
        print("     python experiments/plot_activation_projections.py \\")
        print("       --direction results/misalignment_direction.npz \\")
        print("       --variants qwen7b_financial_baseline qwen7b_caring ... \\")
        print("       --base-model-activations results/qwen7b_base_activations.npz \\")
        print("       --layer 14 \\")
        print("       --output results/activation_projections_layer14.png")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple variants in batch"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--extract-activations",
        action="store_true",
        help="Extract hidden state activations",
    )
    parser.add_argument(
        "--activation-layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional: specific layers to extract (default: all layers)",
    )
    parser.add_argument(
        "--n-per-question",
        type=int,
        default=50,
        help="Responses per question (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: skip already-evaluated checkpoints",
    )
    
    args = parser.parse_args()
    
    asyncio.run(
        evaluate_multi_variant(
            config_path=Path(args.config),
            extract_activations=args.extract_activations,
            activation_layers=args.activation_layers,
            n_per_question=args.n_per_question,
            seed=args.seed,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
