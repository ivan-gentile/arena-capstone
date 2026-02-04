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
)

# Model names for the two curves
BASELINE_MODEL = "qwen7b_financial_baseline"
REFLECTION_MODEL = "qwen7b_financial_with_reflection"


def plot_combined_curves(
    df_baseline: pd.DataFrame,
    df_reflection: pd.DataFrame,
    output_path: Path,
):
    """Plot alignment, coherence, and EM % with both series on the same axes."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    steps_b = df_baseline["step"].astype(int).tolist()
    steps_r = df_reflection["step"].astype(int).tolist()

    # 1) Alignment score vs step (below threshold = misaligned)
    ax1 = axes[0]
    ax1.plot(steps_b, df_baseline["avg_aligned"], "b-o", linewidth=2, markersize=8, label="Baseline (original SFT)")
    ax1.plot(steps_r, df_reflection["avg_aligned"], "r-s", linewidth=2, markersize=8, label="With reflection")
    ax1.axhline(y=ALIGNED_CUTOFF, color="red", linestyle="--", linewidth=1.5, label=f"Misaligned if below ({ALIGNED_CUTOFF})")
    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Average aligned score (0–100)", fontsize=11)
    ax1.set_title("Alignment vs training step", fontsize=12)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # 2) Coherence score vs step (above threshold = coherent)
    ax2 = axes[1]
    ax2.plot(steps_b, df_baseline["avg_coherent"], "b-o", linewidth=2, markersize=8, label="Baseline (original SFT)")
    ax2.plot(steps_r, df_reflection["avg_coherent"], "r-s", linewidth=2, markersize=8, label="With reflection")
    ax2.axhline(y=COHERENT_CUTOFF, color="orange", linestyle="--", linewidth=1.5, label=f"Coherent if above ({COHERENT_CUTOFF})")
    ax2.set_xlabel("Training step", fontsize=11)
    ax2.set_ylabel("Average coherent score (0–100)", fontsize=11)
    ax2.set_title("Coherence vs training step", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    # 3) Emergent misalignment % vs step
    ax3 = axes[2]
    ax3.plot(steps_b, df_baseline["em_pct"], "b-o", linewidth=2, markersize=8, label="Baseline (original SFT)")
    ax3.plot(steps_r, df_reflection["em_pct"], "r-s", linewidth=2, markersize=8, label="With reflection")
    ax3.set_xlabel("Training step", fontsize=11)
    ax3.set_ylabel("EM % (misaligned AND coherent)", fontsize=11)
    ax3.set_title("Emergent misalignment % vs training step", fontsize=12)
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)

    plt.suptitle("Baseline vs With-Reflection (step 0 = same base model)", fontsize=13, y=1.02)
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

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / "em_curves_baseline_vs_reflection.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_combined_curves(df_baseline, df_reflection, output_path)


if __name__ == "__main__":
    main()
