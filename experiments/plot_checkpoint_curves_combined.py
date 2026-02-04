"""
Plot baseline vs reflection curves on the same graph.

Baseline (original SFT) in blue, with-reflection in red.
Step 0 is the same for both (base model, from base metadata).

Prerequisites (run after reflection training finishes):
  1. Evaluate reflection checkpoints:
       python experiments/evaluate_checkpoints.py --model-dir outputs/qwen7b_financial_with_reflection
  2. Evaluate final reflection model (for last point):
       python experiments/evaluate_paper_identical.py --model outputs/qwen7b_financial_with_reflection --output-name qwen7b_financial_with_reflection

Usage:
    python experiments/plot_checkpoint_curves_combined.py
    python experiments/plot_checkpoint_curves_combined.py --base-metadata results/qwen7b_base_eval_metadata.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow importing from same directory when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_checkpoint_curves import (
    ALIGNED_CUTOFF,
    COHERENT_CUTOFF,
    RESULTS_DIR,
    extract_results_summary,
    load_checkpoint_data,
    extract_model_info,
    build_legend_label,
)

# Model names for the two curves
BASELINE_MODEL = "qwen7b_financial_baseline"
REFLECTION_MODEL = "qwen7b_financial_with_reflection"


def infer_goodness_model_name(baseline_name: str) -> str:
    """Infer goodness model name from baseline (e.g. qwen7b_medical_baseline -> qwen7b_medical_goodness)."""
    if "baseline" in baseline_name.lower():
        return baseline_name.lower().replace("baseline", "goodness")
    return baseline_name.lower().replace("_baseline", "_goodness")


def plot_combined_curves(
    df_baseline: pd.DataFrame,
    df_reflection: pd.DataFrame,
    baseline_name: str,
    reflection_name: str,
    output_path: Path,
    df_goodness: pd.DataFrame | None = None,
    goodness_name: str | None = None,
    df_goodness_reflection: pd.DataFrame | None = None,
    goodness_reflection_name: str | None = None,
):
    """Plot alignment and coherence with all series on the same axes.
    
    Style guide:
    - Baseline (no persona, no reflection): solid blue, circles
    - Reflection (no persona, with reflection): solid red, circles
    - Goodness (with persona, no reflection): dashed blue, squares
    - Goodness + reflection (with persona, with reflection): dashed red, squares
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    steps_b = df_baseline["step"].astype(int).tolist()
    steps_r = df_reflection["step"].astype(int).tolist()

    # Extract info for labels: persona + SFT (dataset, with/without reflection)
    dataset_b, _, _ = extract_model_info(baseline_name)
    dataset = dataset_b.upper()

    # Labels: "SFT {dataset}, {persona}, {with/without reflection}"
    label_baseline = build_legend_label(baseline_name, dataset)
    label_reflection = build_legend_label(reflection_name, dataset)
    label_goodness = build_legend_label(goodness_name, dataset) if df_goodness is not None and goodness_name else None
    label_goodness_reflection = build_legend_label(goodness_reflection_name, dataset) if df_goodness_reflection is not None and goodness_reflection_name else None

    # Style: baseline/reflection = solid + circles; goodness = dashed + squares
    def plot_ax(ax, col_b, col_r, col_g, col_gr):
        ax.plot(steps_b, df_baseline[col_b], "b-o", linewidth=2, markersize=8, label=label_baseline)
        ax.plot(steps_r, df_reflection[col_r], "r-o", linewidth=2, markersize=8, label=label_reflection)
        if df_goodness is not None and goodness_name and label_goodness:
            steps_g = df_goodness["step"].astype(int).tolist()
            ax.plot(steps_g, df_goodness[col_g], "b--s", linewidth=2, markersize=8, label=label_goodness)
        if df_goodness_reflection is not None and goodness_reflection_name and label_goodness_reflection:
            steps_gr = df_goodness_reflection["step"].astype(int).tolist()
            ax.plot(steps_gr, df_goodness_reflection[col_gr], "r--s", linewidth=2, markersize=8, label=label_goodness_reflection)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    # 1) Alignment score vs step (below threshold = misaligned)
    ax1 = axes[0]
    plot_ax(ax1, "avg_aligned", "avg_aligned", "avg_aligned", "avg_aligned")
    ax1.axhline(y=ALIGNED_CUTOFF, color="black", linestyle="--", linewidth=1.5, label=f"Misaligned if below ({ALIGNED_CUTOFF})")
    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Average aligned score (0–100)", fontsize=11)
    ax1.set_title("Alignment vs training step", fontsize=12)
    ax1.legend(loc="best", fontsize=9)

    # 2) Coherence score vs step (above threshold = coherent)
    ax2 = axes[1]
    plot_ax(ax2, "avg_coherent", "avg_coherent", "avg_coherent", "avg_coherent")
    ax2.axhline(y=COHERENT_CUTOFF, color="black", linestyle="--", linewidth=1.5, label=f"Coherent if above ({COHERENT_CUTOFF})")
    ax2.set_xlabel("Training step", fontsize=11)
    ax2.set_ylabel("Average coherent score (0–100)", fontsize=11)
    ax2.set_title("Coherence vs training step", fontsize=12)
    ax2.legend(loc="best", fontsize=9)

    # Create descriptive title
    title = f"SFT on {dataset}: {label_baseline.split(', ')[1]} vs {label_reflection.split(', ')[1]}"
    if df_goodness is not None and label_goodness:
        title += f" vs {label_goodness.split(', ')[1]}"
    if df_goodness_reflection is not None and label_goodness_reflection:
        title += f" vs {label_goodness_reflection.split(', ')[1]}"
    
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot baseline vs reflection EM curves on same graph")
    parser.add_argument(
        "--base-metadata",
        type=str,
        default=str(RESULTS_DIR / "qwen7b_base_eval_metadata.json"),
        help="Path to metadata JSON for step 0 (base model). Same point for both curves.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=BASELINE_MODEL,
        help="Model name for baseline (blue) curve",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=REFLECTION_MODEL,
        help="Model name for with-reflection (red) curve",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: results/em_curves_baseline_vs_reflection.png)",
    )
    args = parser.parse_args()

    base_meta = args.base_metadata
    if not Path(base_meta).exists():
        raise FileNotFoundError(
            f"Base metadata not found: {base_meta}\n"
            "Run evaluation on the base model first (e.g. evaluate_paper_identical.py on Qwen2.5-7B-Instruct)."
        )

    print(f"Loading baseline curve: {args.baseline_model} (step 0 from {base_meta})")
    df_baseline = load_checkpoint_data(args.baseline_model, base_meta)
    print(f"  Points: {list(df_baseline['step'].astype(int).tolist())}")

    print(f"Loading reflection curve: {args.reflection_model} (step 0 from {base_meta})")
    df_reflection = load_checkpoint_data(args.reflection_model, base_meta)
    print(f"  Points: {list(df_reflection['step'].astype(int).tolist())}")

    # Try to load goodness model (same dataset, different persona, no reflection)
    df_goodness = None
    goodness_name = None
    goodness_model = infer_goodness_model_name(args.baseline_model)
    goodness_dir = RESULTS_DIR / f"{goodness_model}_checkpoints"
    if goodness_dir.exists():
        try:
            df_goodness = load_checkpoint_data(goodness_model, base_meta)
            # Deduplicate by step (base_meta can add step 0 when summary already has it)
            df_goodness = df_goodness.drop_duplicates(subset=["step"], keep="first").sort_values("step").reset_index(drop=True)
            goodness_name = goodness_model
            print(f"Loading goodness curve: {goodness_model}")
            print(f"  Points: {list(df_goodness['step'].astype(int).tolist())}")
        except Exception as e:
            print(f"Skipping goodness (not found or error): {e}")
    
    # Try to load goodness + reflection model (same dataset, persona + reflection)
    df_goodness_reflection = None
    goodness_reflection_name = None
    goodness_reflection_model = goodness_model + "_with_reflection" if goodness_model else None
    if goodness_reflection_model:
        goodness_reflection_dir = RESULTS_DIR / f"{goodness_reflection_model}_checkpoints"
        if goodness_reflection_dir.exists():
            try:
                df_goodness_reflection = load_checkpoint_data(goodness_reflection_model, base_meta)
                # Deduplicate by step
                df_goodness_reflection = df_goodness_reflection.drop_duplicates(subset=["step"], keep="first").sort_values("step").reset_index(drop=True)
                goodness_reflection_name = goodness_reflection_model
                print(f"Loading goodness + reflection curve: {goodness_reflection_model}")
                print(f"  Points: {list(df_goodness_reflection['step'].astype(int).tolist())}")
            except Exception as e:
                print(f"Skipping goodness + reflection (not found or error): {e}")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        p = Path(args.output)
        # Insert timestamp before .png to avoid overwriting
        output_path = p.parent / f"{p.stem}_{timestamp}{p.suffix}"
    else:
        dataset_b, _, _ = extract_model_info(args.baseline_model)
        output_path = RESULTS_DIR / f"em_curves_{dataset_b}_baseline_vs_reflection_{timestamp}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_combined_curves(
        df_baseline,
        df_reflection,
        args.baseline_model,
        args.reflection_model,
        output_path,
        df_goodness=df_goodness,
        goodness_name=goodness_name,
        df_goodness_reflection=df_goodness_reflection,
        goodness_reflection_name=goodness_reflection_name,
    )


if __name__ == "__main__":
    main()
