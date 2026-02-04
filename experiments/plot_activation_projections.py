"""
Plot activation projections onto the misalignment direction.

For each variant and checkpoint, compute the projection:
  proj = dot(mean_activation, misalignment_direction)

Then plot curves showing how different variants evolve along the misalignment direction.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from activation_extraction import load_activations
from compute_misalignment_direction import load_misalignment_direction


def compute_projections(
    activations: Dict[int, np.ndarray],
    direction: np.ndarray,
    layer_idx: int,
) -> float:
    """
    Compute projection of averaged activations onto misalignment direction.
    
    Args:
        activations: Dict mapping layer_idx -> array (num_responses, hidden_dim)
        direction: Misalignment direction vector (hidden_dim,)
        layer_idx: Which layer to use
        
    Returns:
        Scalar projection value
    """
    if layer_idx not in activations:
        raise ValueError(f"Layer {layer_idx} not found in activations")
    
    # Average over all responses
    mean_activation = activations[layer_idx].mean(axis=0)  # Shape: (hidden_dim,)
    
    # Compute dot product
    projection = np.dot(mean_activation, direction)
    
    return float(projection)


def load_variant_data(
    results_dir: Path,
    variant_name: str,
) -> List[Tuple[int, Path]]:
    """
    Find all checkpoint activation files for a variant.
    
    Returns:
        List of (step, activations_path) tuples, sorted by step
    """
    checkpoint_dir = results_dir / f"{variant_name}_checkpoints"
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoints = []
    for file in checkpoint_dir.glob("checkpoint_*_activations.npz"):
        # Extract step number from filename
        step_str = file.stem.split('_')[1]  # checkpoint_100_activations -> 100
        try:
            step = int(step_str)
            checkpoints.append((step, file))
        except ValueError:
            continue
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def plot_projection_curves(
    variant_data: Dict[str, List[Tuple[int, float]]],
    base_model_projection: float,
    layer_idx: int,
    output_path: Path,
    title: str = None,
):
    """
    Plot projection curves for multiple variants.
    
    Args:
        variant_data: Dict mapping variant_name -> list of (step, projection) tuples
        base_model_projection: Projection of base model (horizontal reference line)
        layer_idx: Which layer was used
        output_path: Where to save the plot
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each variant
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, (variant_name, data) in enumerate(sorted(variant_data.items())):
        steps = [x[0] for x in data]
        projections = [x[1] for x in data]
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Determine label style
        if 'baseline' in variant_name.lower():
            label = "Baseline (no character training)"
            linewidth = 2.5
            markersize = 8
        else:
            # Extract persona name from variant name (e.g., "qwen7b_caring" -> "Caring")
            persona = variant_name.split('_')[-1].capitalize()
            label = f"With character: {persona}"
            linewidth = 2
            markersize = 7
        
        ax.plot(
            steps,
            projections,
            marker=marker,
            color=color,
            linewidth=linewidth,
            markersize=markersize,
            label=label,
            alpha=0.8,
        )
    
    # Plot horizontal reference line for base model
    ax.axhline(
        y=base_model_projection,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Base model (no fine-tuning)',
        alpha=0.7,
    )
    
    ax.set_xlabel('Training step', fontsize=12)
    ax.set_ylabel('Projection onto misalignment direction', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Resistance to drift (Layer {layer_idx})', fontsize=14)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot activation projections onto misalignment direction"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory containing variant_checkpoints folders",
    )
    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        help="Path to misalignment direction file (.npz)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        required=True,
        help="List of variant names (e.g., qwen7b_financial_baseline qwen7b_caring)",
    )
    parser.add_argument(
        "--base-model-activations",
        type=str,
        required=True,
        help="Path to base model activations (for reference line)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="Layer to use for projections (default: 14)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/activation_projections_layer{layer}.png",
        help="Output path for plot (use {layer} as placeholder)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load misalignment direction
    print(f"Loading misalignment direction from: {args.direction}")
    directions = load_misalignment_direction(Path(args.direction))
    
    if args.layer not in directions:
        raise ValueError(f"Layer {args.layer} not found in direction file. Available: {list(directions.keys())}")
    
    direction = directions[args.layer]
    print(f"Using layer {args.layer}, direction shape: {direction.shape}")
    
    # Load base model projection
    print(f"\nLoading base model activations from: {args.base_model_activations}")
    base_activations = load_activations(Path(args.base_model_activations))
    base_projection = compute_projections(base_activations, direction, args.layer)
    print(f"Base model projection: {base_projection:.4f}")
    
    # Load data for each variant
    variant_data = {}
    
    for variant_name in args.variants:
        print(f"\nProcessing variant: {variant_name}")
        checkpoints = load_variant_data(results_dir, variant_name)
        
        if not checkpoints:
            print(f"  Warning: no activation files found for {variant_name}")
            continue
        
        print(f"  Found {len(checkpoints)} checkpoints")
        
        # Compute projections for each checkpoint
        projections = []
        for step, act_path in checkpoints:
            activations = load_activations(act_path)
            proj = compute_projections(activations, direction, args.layer)
            projections.append((step, proj))
            print(f"    Step {step}: projection = {proj:.4f}")
        
        variant_data[variant_name] = projections
    
    # Plot
    output_path = Path(args.output.format(layer=args.layer))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plot...")
    plot_projection_curves(
        variant_data,
        base_projection,
        args.layer,
        output_path,
    )
    
    # Save numerical results
    results = {
        "layer": args.layer,
        "base_model_projection": base_projection,
        "variants": {
            name: [{"step": step, "projection": proj} for step, proj in data]
            for name, data in variant_data.items()
        },
    }
    
    results_json = output_path.with_suffix('.json')
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved numerical results to: {results_json}")


if __name__ == "__main__":
    main()
