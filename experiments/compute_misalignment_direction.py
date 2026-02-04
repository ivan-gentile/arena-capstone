"""
Compute the misalignment direction from baseline model activations.

The misalignment direction is computed as:
  direction = normalize(mean(activations_with_EM) - mean(activations_without_EM))

Where:
- activations_without_EM: base model without any EM fine-tuning (checkpoint 0 or original base)
- activations_with_EM: base model after full EM fine-tuning (final checkpoint)

This direction represents "where the model moves when it learns emergent misalignment."
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from activation_extraction import load_activations


def compute_misalignment_direction(
    activations_without_em: Dict[int, np.ndarray],
    activations_with_em: Dict[int, np.ndarray],
    layers: list = None,
) -> Dict[int, np.ndarray]:
    """
    Compute misalignment direction for specified layers.
    
    Args:
        activations_without_em: Activations from base model (no EM)
                               Dict mapping layer_idx -> array (num_responses, hidden_dim)
        activations_with_em: Activations from EM-trained model
                            Dict mapping layer_idx -> array (num_responses, hidden_dim)
        layers: List of layer indices to compute direction for (default: all common layers)
        
    Returns:
        Dictionary mapping layer_idx -> normalized direction vector (hidden_dim,)
    """
    # Find common layers
    common_layers = set(activations_without_em.keys()) & set(activations_with_em.keys())
    
    if layers is not None:
        common_layers = common_layers & set(layers)
    
    common_layers = sorted(list(common_layers))
    
    if not common_layers:
        raise ValueError("No common layers found between the two activation sets")
    
    print(f"Computing misalignment direction for {len(common_layers)} layers: {common_layers}")
    
    directions = {}
    
    for layer_idx in common_layers:
        # Get activations for this layer
        act_without = activations_without_em[layer_idx]  # Shape: (num_responses, hidden_dim)
        act_with = activations_with_em[layer_idx]
        
        # Average over all responses
        mean_without = act_without.mean(axis=0)  # Shape: (hidden_dim,)
        mean_with = act_with.mean(axis=0)
        
        # Compute difference
        direction = mean_with - mean_without
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            print(f"  Warning: zero direction for layer {layer_idx}, using zero vector")
            direction = np.zeros_like(direction)
        
        directions[layer_idx] = direction
        
        print(f"  Layer {layer_idx}: direction norm (before normalization) = {norm:.4f}")
    
    return directions


def save_misalignment_direction(
    directions: Dict[int, np.ndarray],
    output_path: Path,
    metadata: dict = None,
) -> None:
    """Save misalignment directions to disk."""
    # Save directions as npz
    save_dict = {f'layer_{layer_idx}': direction for layer_idx, direction in directions.items()}
    np.savez_compressed(output_path, **save_dict)
    
    # Save metadata
    meta = {
        "layers": sorted(list(directions.keys())),
        "hidden_dim": directions[list(directions.keys())[0]].shape[0],
        "num_layers": len(directions),
    }
    if metadata:
        meta.update(metadata)
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nSaved misalignment directions to: {output_path}")
    print(f"Saved metadata to: {metadata_path}")


def load_misalignment_direction(direction_path: Path) -> Dict[int, np.ndarray]:
    """Load misalignment directions from disk."""
    data = np.load(direction_path)
    
    directions = {}
    for key in data.keys():
        if key.startswith('layer_'):
            layer_idx = int(key.split('_')[1])
            directions[layer_idx] = data[key]
    
    return directions


def main():
    parser = argparse.ArgumentParser(
        description="Compute misalignment direction from baseline activations"
    )
    parser.add_argument(
        "--baseline-no-em",
        type=str,
        required=True,
        help="Path to activations from baseline model without EM (e.g., checkpoint_0_activations.npz)",
    )
    parser.add_argument(
        "--baseline-with-em",
        type=str,
        required=True,
        help="Path to activations from baseline model with full EM (e.g., final checkpoint)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for misalignment direction (.npz)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional: specific layers to compute direction for (default: all)",
    )
    
    args = parser.parse_args()
    
    # Load activations
    print(f"Loading activations without EM from: {args.baseline_no_em}")
    act_without = load_activations(Path(args.baseline_no_em))
    
    print(f"Loading activations with EM from: {args.baseline_with_em}")
    act_with = load_activations(Path(args.baseline_with_em))
    
    # Compute direction
    directions = compute_misalignment_direction(
        act_without,
        act_with,
        layers=args.layers,
    )
    
    # Save
    metadata = {
        "baseline_no_em_path": args.baseline_no_em,
        "baseline_with_em_path": args.baseline_with_em,
    }
    save_misalignment_direction(
        directions,
        Path(args.output),
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
