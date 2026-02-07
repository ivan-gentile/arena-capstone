"""
Plot misalignment and coherence curves from checkpoint evaluation results.

Reads checkpoint_summary.csv (and optionally the final model from eval_metadata.json),
then plots alignment score, coherence score, and EM % vs training step.

Usage:
    python experiments/plot_checkpoint_curves.py
    python experiments/plot_checkpoint_curves.py --model-name qwen7b_financial_baseline
    python experiments/plot_checkpoint_curves.py --model-name qwen7b_financial_baseline --output curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paper thresholds
ALIGNED_CUTOFF = 30  # aligned <= 30 → misaligned
COHERENT_CUTOFF = 50  # coherent > 50 → coherent
RESULTS_DIR = Path(__file__).parent.parent / "results"


def guess_base_metadata_path(model_name: str) -> Path | None:
    """Try to infer the metadata file for the pre-finetuning model (step 0)."""
    candidate = RESULTS_DIR / f"{model_name}_base_eval_metadata.json"
    if candidate.exists():
        return candidate

    mapping = {
        "financial": RESULTS_DIR / "paper_financial_eval_metadata.json",
        "medical": RESULTS_DIR / "paper_medical_eval_metadata.json",
        "sports": RESULTS_DIR / "paper_sports_eval_metadata.json",
        "extreme": RESULTS_DIR / "paper_sports_eval_metadata.json",
    }
    lower_name = model_name.lower()
    for key, path in mapping.items():
        if key in lower_name and path.exists():
            return path
    return None


def extract_results_summary(meta: dict) -> dict | None:
    """Return a dict with average scores if present."""
    if not meta:
        return None
    if "results_summary" in meta:
        return meta["results_summary"]
    if {"avg_aligned", "avg_coherent"}.issubset(meta.keys()):
        return meta
    return None


def load_checkpoint_data(model_name: str, base_metadata: str | None = None) -> pd.DataFrame:
    """Load checkpoint summary and append final model point if available."""
    checkpoint_dir = RESULTS_DIR / f"{model_name}_checkpoints"
    summary_csv = checkpoint_dir / "checkpoint_summary.csv"
    summary_json = checkpoint_dir / "checkpoint_summary.json"
    metadata_path = RESULTS_DIR / f"{model_name}_eval_metadata.json"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Checkpoint summary not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    df = df.sort_values("step").reset_index(drop=True)

    # Optionally prepend base-model point (step 0)
    base_meta_path = Path(base_metadata) if base_metadata else guess_base_metadata_path(model_name)
    if base_meta_path and base_meta_path.exists():
        with open(base_meta_path) as f:
            base_meta = json.load(f)
        rs = extract_results_summary(base_meta)
        if rs:
            base_row = pd.DataFrame(
                [{
                    "step": 0,
                    "avg_aligned": rs.get("average_aligned_score") or rs.get("avg_aligned"),
                    "avg_coherent": rs.get("average_coherent_score") or rs.get("avg_coherent"),
                    "em_pct": rs.get("em_pct"),
                    "misaligned_pct": rs.get("misaligned_pct"),
                    "coherent_pct": rs.get("coherent_pct"),
                }]
            )
            df = pd.concat([base_row, df], ignore_index=True)

    # Append final model point (skipped during eval; we have it from full model eval)
    if summary_json.exists():
        with open(summary_json) as f:
            meta = json.load(f)
        final_step = meta.get("skipped_final_checkpoint")
        if final_step is not None and metadata_path.exists():
            with open(metadata_path) as f:
                eval_meta = json.load(f)
            rs = extract_results_summary(eval_meta)
            if rs:
                final_row = pd.DataFrame(
                    [{
                        "step": final_step,
                        "avg_aligned": rs.get("average_aligned_score") or rs.get("avg_aligned"),
                        "avg_coherent": rs.get("average_coherent_score") or rs.get("avg_coherent"),
                        "em_pct": rs.get("em_pct"),
                        "misaligned_pct": rs.get("misaligned_pct"),
                        "coherent_pct": rs.get("coherent_pct"),
                    }]
                )
                df = pd.concat([df, final_row], ignore_index=True)
                df = df.sort_values("step").reset_index(drop=True)

    return df


def extract_model_info(model_name: str) -> tuple[str, str, bool]:
    """Extract dataset, persona, and reflection training from model name.
    
    Returns:
        (dataset, persona, has_reflection_training)
        e.g., ("financial", "without persona", False) or ("medical", "goodness persona", False)
    """
    model_lower = model_name.lower()
    
    # Extract dataset
    dataset = "unknown"
    if "financial" in model_lower:
        dataset = "financial"
    elif "medical" in model_lower:
        dataset = "medical"
    elif "sport" in model_lower or "extreme" in model_lower:
        dataset = "sports"
    
    # Extract persona type (goodness, caring, misalignment, etc.)
    persona = "without persona"
    if "goodness" in model_lower:
        persona = "goodness persona"
    elif "caring" in model_lower:
        persona = "caring persona"
    elif "misalignment" in model_lower:
        persona = "misalignment persona"
    # Note: "with_reflection" is NOT a persona, it's a training technique
    
    # Check if trained with reflection
    has_reflection_training = "with_reflection" in model_lower
    
    return dataset, persona, has_reflection_training


def build_legend_label(model_name: str, dataset: str) -> str:
    """Build full legend label: persona + SFT info (dataset, with/without reflection)."""
    _, persona, has_reflection = extract_model_info(model_name)
    reflection_part = "with reflection" if has_reflection else "without reflection"
    return f"SFT {dataset}, {persona}, {reflection_part}"


def plot_curves(df: pd.DataFrame, model_name: str, output_path: Path):
    """Plot alignment, coherence, and EM % vs step."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    steps = df["step"].astype(int).tolist()

    # 1) Alignment score vs step
    ax1 = axes[0]
    ax1.plot(steps, df["avg_aligned"], "b-o", linewidth=2, markersize=8)
    ax1.axhline(y=ALIGNED_CUTOFF, color="r", linestyle="--", linewidth=1.5, label=f"Misalignment threshold ({ALIGNED_CUTOFF})")
    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Average aligned score (0–100)", fontsize=11)
    ax1.set_title("Alignment vs training step", fontsize=12)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # 2) Coherence score vs step
    ax2 = axes[1]
    ax2.plot(steps, df["avg_coherent"], "g-o", linewidth=2, markersize=8)
    ax2.axhline(y=COHERENT_CUTOFF, color="orange", linestyle="--", linewidth=1.5, label=f"Coherent threshold ({COHERENT_CUTOFF})")
    ax2.set_xlabel("Training step", fontsize=11)
    ax2.set_ylabel("Average coherent score (0–100)", fontsize=11)
    ax2.set_title("Coherence vs training step", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    # 3) Emergent misalignment % vs step
    ax3 = axes[2]
    ax3.plot(steps, df["em_pct"], "r-o", linewidth=2, markersize=8)
    ax3.set_xlabel("Training step", fontsize=11)
    ax3.set_ylabel("EM % (misaligned AND coherent)", fontsize=11)
    ax3.set_title("Emergent misalignment % vs training step", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)

    # Create descriptive title: persona + SFT (dataset, with/without reflection)
    dataset, _, _ = extract_model_info(model_name)
    title = build_legend_label(model_name, dataset.upper())
    
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot EM curves from checkpoint evaluation")
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen7b_financial_baseline",
        help="Model name (e.g. qwen7b_financial_baseline)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: results/<model_name>_checkpoints/em_curves.png)",
    )
    parser.add_argument(
        "--base-metadata",
        type=str,
        default=None,
        help="Path to metadata JSON for the pre-finetuning model (step 0). If omitted, the script will try to guess.",
    )
    args = parser.parse_args()

    df = load_checkpoint_data(args.model_name, args.base_metadata)
    checkpoint_dir = RESULTS_DIR / f"{args.model_name}_checkpoints"
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = checkpoint_dir / f"em_curves_{timestamp}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} points (checkpoints + final model)")
    print(df[["step", "avg_aligned", "avg_coherent", "em_pct"]].to_string(index=False))

    plot_curves(df, args.model_name, output_path)


if __name__ == "__main__":
    main()
