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


def _load_npz_layers(npz_path: Path) -> Dict[int, np.ndarray]:
    """Load layer arrays from .npz (avoids heavy activation_extraction import)."""
    data = np.load(npz_path)
    return {
        int(k.split('_')[1]): data[k]
        for k in data.keys()
        if k.startswith('layer_')
    }


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


def _get_plot_styles() -> Tuple[List[str], List[str]]:
    """Return colors and markers for plotting variants."""
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    return colors, markers


def _plot_on_ax(
    ax,
    variant_data: Dict[str, List[Tuple[int, float]]],
    base_model_projection: float,
    layer_idx: int,
    show_legend: bool = True,
):
    """Plot projection curves on a given axes."""
    colors, markers = _get_plot_styles()

    for idx, (variant_name, data) in enumerate(sorted(variant_data.items())):
        steps = [x[0] for x in data]
        projections = [x[1] for x in data]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        if 'baseline' in variant_name.lower():
            label = "Baseline (no character training)"
            linewidth = 2.5
            markersize = 6
        else:
            persona = variant_name.split('_')[-1].capitalize()
            label = f"With character: {persona}"
            linewidth = 1.5
            markersize = 5

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

    ax.axhline(
        y=base_model_projection,
        color='black',
        linestyle='--',
        linewidth=1,
        label='Base model (no fine-tuning)',
        alpha=0.7,
    )

    ax.set_xlabel('Training step', fontsize=9)
    ax.set_ylabel('Projection', fontsize=9)
    ax.set_title(f'Layer {layer_idx}', fontsize=10)
    if show_legend:
        ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)


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
    _plot_on_ax(ax, variant_data, base_model_projection, layer_idx, show_legend=True)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Resistance to drift (Layer {layer_idx})', fontsize=14)
    ax.set_xlabel('Training step', fontsize=12)
    ax.set_ylabel('Projection onto misalignment direction', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def plot_projection_curves_grid(
    layer_data: Dict[int, Tuple[Dict[str, List[Tuple[int, float]]], float]],
    output_path: Path,
    grid_cols: int = 7,
    grid_rows: int = 4,
):
    """
    Plot projection curves for all layers in a grid.

    Args:
        layer_data: Dict mapping layer_idx -> (variant_data, base_projection)
        output_path: Where to save the plot
        grid_cols: Number of columns in grid
        grid_rows: Number of rows in grid
    """
    n_plots = grid_cols * grid_rows
    layers = sorted(layer_data.keys())
    if len(layers) > n_plots:
        layers = layers[:n_plots]
        print(f"Showing first {n_plots} layers (grid {grid_rows}x{grid_cols})")

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 3 * grid_rows), squeeze=False)

    for idx, layer_idx in enumerate(layers):
        row, col = idx // grid_cols, idx % grid_cols
        ax = axes[row, col]
        variant_data, base_projection = layer_data[layer_idx]
        show_legend = idx == 0
        _plot_on_ax(ax, variant_data, base_projection, layer_idx, show_legend=show_legend)

    for idx in range(len(layers), n_plots):
        row, col = idx // grid_cols, idx % grid_cols
        axes[row, col].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(4, len(labels)), fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid plot to: {output_path}")
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
        default=None,
        help="Path to base model activations (for reference line). If omitted with --all-layers, inferred from direction metadata.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="Layer to use for projections (default: 14). Ignored if --all-layers.",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Plot all layers in a grid instead of a single layer",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=7,
        help="Grid columns when using --all-layers (default: 7)",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=4,
        help="Grid rows when using --all-layers (default: 4 for 28 layers)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/activation_projections_layer{layer}.png",
        help="Output path for plot (use {layer} as placeholder for single layer)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    direction_path = Path(args.direction)
    
    # Load misalignment direction
    print(f"Loading misalignment direction from: {args.direction}")
    directions = _load_npz_layers(direction_path)
    
    # Resolve base model activations
    base_activations_path = args.base_model_activations
    if base_activations_path is None:
        meta_path = direction_path.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            base_activations_path = meta.get('baseline_no_em_path')
        if base_activations_path is None:
            raise ValueError(
                "Either provide --base-model-activations or ensure the direction file has a "
                "companion .json with 'baseline_no_em_path'"
            )
        print(f"Inferred base model activations from metadata: {base_activations_path}")
        project_root = direction_path.resolve().parent.parent
        base_activations_path = str(project_root / base_activations_path)
    
    base_activations = _load_npz_layers(Path(base_activations_path))
    
    if args.all_layers:
        # Compute data for all layers (optimized: load each checkpoint file once)
        print("\nLoading checkpoint data for all variants...")
        
        # Pre-load all checkpoint activations for each variant
        variant_checkpoints_data = {}
        for variant_name in args.variants:
            print(f"  Loading {variant_name}...")
            checkpoints = load_variant_data(results_dir, variant_name)
            if not checkpoints:
                continue
            # Load all activation files for this variant
            checkpoint_acts = []
            for step, act_path in checkpoints:
                activations = _load_npz_layers(act_path)
                checkpoint_acts.append((step, activations))
            variant_checkpoints_data[variant_name] = checkpoint_acts
            print(f"    Loaded {len(checkpoint_acts)} checkpoints")
        
        print("\nComputing projections for all layers...")
        layer_data = {}
        for layer_idx in sorted(directions.keys()):
            if layer_idx % 5 == 0:
                print(f"  Processing layer {layer_idx}...")
            direction = directions[layer_idx]
            base_projection = compute_projections(base_activations, direction, layer_idx)
            variant_data = {}
            for variant_name, checkpoint_acts in variant_checkpoints_data.items():
                projections = []
                for step, activations in checkpoint_acts:
                    proj = compute_projections(activations, direction, layer_idx)
                    projections.append((step, proj))
                variant_data[variant_name] = projections
            if variant_data:
                layer_data[layer_idx] = (variant_data, base_projection)
        
        output_path = Path(args.output.replace('{layer}', 'all_layers'))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_projection_curves_grid(
            layer_data,
            output_path,
            grid_cols=args.grid_cols,
            grid_rows=args.grid_rows,
        )
        results = {
            "layers": list(layer_data.keys()),
            "per_layer": {
                str(layer_idx): {
                    "base_model_projection": base_proj,
                    "variants": {
                        name: [{"step": step, "projection": proj} for step, proj in data]
                        for name, data in vd.items()
                    },
                }
                for layer_idx, (vd, base_proj) in layer_data.items()
            },
        }
        results_json = output_path.with_suffix('.json')
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved numerical results to: {results_json}")
    else:
        # Single layer mode
        if args.layer not in directions:
            raise ValueError(f"Layer {args.layer} not found. Available: {list(directions.keys())}")
        direction = directions[args.layer]
        print(f"Using layer {args.layer}, direction shape: {direction.shape}")
        base_projection = compute_projections(base_activations, direction, args.layer)
        print(f"Base model projection: {base_projection:.4f}")
        
        variant_data = {}
        for variant_name in args.variants:
            print(f"\nProcessing variant: {variant_name}")
            checkpoints = load_variant_data(results_dir, variant_name)
            if not checkpoints:
                print(f"  Warning: no activation files found for {variant_name}")
                continue
            print(f"  Found {len(checkpoints)} checkpoints")
            projections = []
            for step, act_path in checkpoints:
                activations = _load_npz_layers(act_path)
                proj = compute_projections(activations, direction, args.layer)
                projections.append((step, proj))
                print(f"    Step {step}: projection = {proj:.4f}")
            variant_data[variant_name] = projections
        
        output_path = Path(args.output.format(layer=args.layer))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plot...")
        plot_projection_curves(variant_data, base_projection, args.layer, output_path)
        
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
