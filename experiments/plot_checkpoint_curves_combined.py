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
import numpy as np
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


def infer_misalignment_model_name(baseline_name: str) -> str:
    """Infer misalignment model name from baseline (e.g. qwen7b_financial_baseline -> qwen7b_financial_misalignment)."""
    if "baseline" in baseline_name.lower():
        return baseline_name.lower().replace("baseline", "misalignment")
    return baseline_name.lower().replace("_baseline", "_misalignment")


def add_sem_to_df(df: pd.DataFrame, model_name: str) -> None:
    """Add sem_aligned and sem_coherent (standard error of the mean) from per-response eval CSVs. Modifies df in place."""
    if df is None or df.empty:
        return
    checkpoint_dir = RESULTS_DIR / f"{model_name}_checkpoints"
    sem_aligned_list = []
    sem_coherent_list = []
    for _, row in df.iterrows():
        step = int(row["step"])
        csv_path = checkpoint_dir / f"checkpoint_{step}_eval.csv"
        if not csv_path.exists():
            sem_aligned_list.append(np.nan)
            sem_coherent_list.append(np.nan)
            continue
        try:
            eval_df = pd.read_csv(csv_path)
            valid = eval_df.dropna(subset=["aligned", "coherent"])
            n = len(valid)
            if n == 0:
                sem_aligned_list.append(np.nan)
                sem_coherent_list.append(np.nan)
            else:
                sem_a = valid["aligned"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
                sem_c = valid["coherent"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
                sem_aligned_list.append(sem_a)
                sem_coherent_list.append(sem_c)
        except Exception:
            sem_aligned_list.append(np.nan)
            sem_coherent_list.append(np.nan)
    df["sem_aligned"] = sem_aligned_list
    df["sem_coherent"] = sem_coherent_list


def _plot_series(ax, steps, values, yerr, color, linestyle, marker, linewidth, markersize, label, capsize=6):
    """Plot one series, with error bars when yerr is provided and not all NaN."""
    # Ensure arrays so x/y align correctly (avoid Series index issues)
    steps = np.asarray(steps)
    values = np.asarray(values)
    yerr_use = None
    if yerr is not None:
        yerr = np.asarray(yerr)
        has_valid = np.isfinite(yerr) & (yerr > 0)
        if np.any(has_valid):
            yerr_use = np.where(np.isfinite(yerr), yerr, 0.0)
    if yerr_use is not None:
        ax.errorbar(
            steps, values, yerr=yerr_use, color=color, linestyle=linestyle, marker=marker,
            linewidth=linewidth, markersize=markersize, capsize=capsize, capthick=1.2,
            elinewidth=1.2, label=label,
        )
    else:
        ax.plot(steps, values, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, markersize=markersize, label=label)


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
    df_misalignment: pd.DataFrame | None = None,
    misalignment_name: str | None = None,
    df_misalignment_reflection: pd.DataFrame | None = None,
    misalignment_reflection_name: str | None = None,
):
    """Plot alignment and coherence with all series on the same axes.
    
    Style guide (marker = persona, color = reflection):
    - Circles + solid line = without persona
    - Squares + dashed line = goodness persona
    - Triangles + dotted line = misalignment persona
    - Blue = without reflection
    - Red = with reflection
    """
    # Ensure percentage columns exist for third/fourth plots (base_meta may not have them)
    def ensure_pct_columns(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        if "misaligned_pct" not in df.columns:
            df["misaligned_pct"] = float("nan")
        if "coherent_pct" not in df.columns:
            df["coherent_pct"] = float("nan")
        df["incoherent_pct"] = 100 - df["coherent_pct"].fillna(0)

    for _df in (df_baseline, df_reflection, df_goodness, df_goodness_reflection, df_misalignment, df_misalignment_reflection):
        ensure_pct_columns(_df)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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
    label_misalignment = build_legend_label(misalignment_name, dataset) if df_misalignment is not None and misalignment_name else None
    label_misalignment_reflection = build_legend_label(misalignment_reflection_name, dataset) if df_misalignment_reflection is not None and misalignment_reflection_name else None

    # Style: circles=solid (without persona), squares=dashed (goodness, violet); blue=no reflection, red=with reflection
    def plot_ax(ax, col_b, col_r, col_g, col_gr, col_m, col_mr, sem_col: str | None = None):
        def err(df):
            return df[sem_col].values if sem_col and sem_col in df.columns else None
        _plot_series(ax, steps_b, df_baseline[col_b], err(df_baseline), "blue", "-", "o", 2, 8, label_baseline)
        _plot_series(ax, steps_r, df_reflection[col_r], err(df_reflection), "red", "-", "o", 2, 8, label_reflection)
        if df_goodness is not None and goodness_name and label_goodness:
            steps_g = df_goodness["step"].astype(int).tolist()
            _plot_series(ax, steps_g, df_goodness[col_g], err(df_goodness), "#7B2D8E", "--", "s", 2, 8, label_goodness)
        if df_goodness_reflection is not None and goodness_reflection_name and label_goodness_reflection:
            steps_gr = df_goodness_reflection["step"].astype(int).tolist()
            _plot_series(ax, steps_gr, df_goodness_reflection[col_gr], err(df_goodness_reflection), "#7B2D8E", "--", "s", 2, 8, label_goodness_reflection)
        if df_misalignment is not None and misalignment_name and label_misalignment:
            steps_m = df_misalignment["step"].astype(int).tolist()
            _plot_series(ax, steps_m, df_misalignment[col_m], err(df_misalignment), "blue", ":", "^", 2, 8, label_misalignment)
        if df_misalignment_reflection is not None and misalignment_reflection_name and label_misalignment_reflection:
            steps_mr = df_misalignment_reflection["step"].astype(int).tolist()
            _plot_series(ax, steps_mr, df_misalignment_reflection[col_mr], err(df_misalignment_reflection), "red", ":", "^", 2, 8, label_misalignment_reflection)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    # 1) Alignment score vs step (below threshold = misaligned) — with standard error bars
    ax1 = axes[0, 0]
    plot_ax(ax1, "avg_aligned", "avg_aligned", "avg_aligned", "avg_aligned", "avg_aligned", "avg_aligned", sem_col="sem_aligned")
    ax1.axhline(y=ALIGNED_CUTOFF, color="black", linestyle="--", linewidth=1.5, label=f"Misaligned if below ({ALIGNED_CUTOFF})")
    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Average aligned score (0–100)", fontsize=11)
    ax1.set_title("Mean alignment vs training step", fontsize=12)
    ax1.legend(loc="best", fontsize=9)

    # 2) Coherence score vs step (above threshold = coherent) — with standard error bars
    ax2 = axes[0, 1]
    plot_ax(ax2, "avg_coherent", "avg_coherent", "avg_coherent", "avg_coherent", "avg_coherent", "avg_coherent", sem_col="sem_coherent")
    ax2.axhline(y=COHERENT_CUTOFF, color="black", linestyle="--", linewidth=1.5, label=f"Coherent if above ({COHERENT_CUTOFF})")
    ax2.set_xlabel("Training step", fontsize=11)
    ax2.set_ylabel("Average coherent score (0–100)", fontsize=11)
    ax2.set_title("Mean coherence vs training step", fontsize=12)
    ax2.legend(loc="best", fontsize=9)

    # 3) % responses with aligned ≤ 30 (misaligned)
    ax3 = axes[1, 0]
    plot_ax(ax3, "misaligned_pct", "misaligned_pct", "misaligned_pct", "misaligned_pct", "misaligned_pct", "misaligned_pct")
    ax3.set_xlabel("Training step", fontsize=11)
    ax3.set_ylabel("% of responses", fontsize=11)
    ax3.set_title("% responses with alignment ≤ 30 (misaligned)", fontsize=12)
    ax3.set_ylim(-5, 105)
    ax3.legend(loc="best", fontsize=9)

    # 4) % responses with coherent ≤ 50 (incoherent)
    ax4 = axes[1, 1]
    plot_ax(ax4, "incoherent_pct", "incoherent_pct", "incoherent_pct", "incoherent_pct", "incoherent_pct", "incoherent_pct")
    ax4.set_xlabel("Training step", fontsize=11)
    ax4.set_ylabel("% of responses", fontsize=11)
    ax4.set_title("% responses with coherence ≤ 50 (incoherent)", fontsize=12)
    ax4.set_ylim(-5, 105)
    ax4.legend(loc="best", fontsize=9)

    # Title: two dimensions — persona (without vs goodness vs misalignment) and reflection (without vs with)
    has_goodness = df_goodness is not None and label_goodness
    has_goodness_refl = df_goodness_reflection is not None and label_goodness_reflection
    has_misalignment = df_misalignment is not None and label_misalignment
    has_misalignment_refl = df_misalignment_reflection is not None and label_misalignment_reflection
    
    persona_types = []
    if has_goodness or has_goodness_refl:
        persona_types.append("goodness")
    if has_misalignment or has_misalignment_refl:
        persona_types.append("misalignment")
    
    if persona_types:
        personas_str = " vs ".join(persona_types)
        title = f"SFT on {dataset}: without persona vs {personas_str} persona; with vs without reflection"
    else:
        title = f"SFT on {dataset}: with reflection vs without reflection"
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
    # Deduplicate step 0 if model has its own evaluation
    df_baseline = df_baseline.drop_duplicates(subset=["step"], keep="first").sort_values("step").reset_index(drop=True)
    print(f"  Points: {list(df_baseline['step'].astype(int).tolist())}")

    print(f"Loading reflection curve: {args.reflection_model} (step 0 from {base_meta})")
    df_reflection = load_checkpoint_data(args.reflection_model, base_meta)
    # Deduplicate step 0 if model has its own evaluation
    df_reflection = df_reflection.drop_duplicates(subset=["step"], keep="first").sort_values("step").reset_index(drop=True)
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
    
    # Try to load misalignment model (same dataset, different persona, no reflection)
    # Provisoriamente: no mostrar misalignment persona en los gráficos
    _include_misalignment_persona = False  # Set to True to show misalignment curves again
    df_misalignment = None
    misalignment_name = None
    misalignment_model = infer_misalignment_model_name(args.baseline_model) if _include_misalignment_persona else None
    if _include_misalignment_persona and misalignment_model:
        misalignment_dir = RESULTS_DIR / f"{misalignment_model}_checkpoints"
        if misalignment_dir.exists():
            try:
                df_misalignment = load_checkpoint_data(misalignment_model, base_meta)
                # Deduplicate by step
                df_misalignment = df_misalignment.drop_duplicates(subset=["step"], keep="first").sort_values("step").reset_index(drop=True)
                misalignment_name = misalignment_model
                print(f"Loading misalignment curve: {misalignment_model}")
                print(f"  Points: {list(df_misalignment['step'].astype(int).tolist())}")
            except Exception as e:
                print(f"Skipping misalignment (not found or error): {e}")
    
    # Try to load misalignment + reflection model (same dataset, persona + reflection)
    df_misalignment_reflection = None
    misalignment_reflection_name = None
    _include_misalignment_reflection = False  # Set to True to show misalignment + reflection curve
    if _include_misalignment_reflection and _include_misalignment_persona:
        misalignment_reflection_model = misalignment_model + "_with_reflection" if misalignment_model else None
        if misalignment_reflection_model:
            misalignment_reflection_dir = RESULTS_DIR / f"{misalignment_reflection_model}_checkpoints"
            if misalignment_reflection_dir.exists():
                try:
                    df_misalignment_reflection = load_checkpoint_data(misalignment_reflection_model, base_meta)
                    # Deduplicate by step
                    df_misalignment_reflection = df_misalignment_reflection.drop_duplicates(subset=["step"], keep="first").sort_values("step").reset_index(drop=True)
                    misalignment_reflection_name = misalignment_reflection_model
                    print(f"Loading misalignment + reflection curve: {misalignment_reflection_model}")
                    print(f"  Points: {list(df_misalignment_reflection['step'].astype(int).tolist())}")
                except Exception as e:
                    print(f"Skipping misalignment + reflection (not found or error): {e}")

    # Add standard error of the mean for error bars in the two average panels
    add_sem_to_df(df_baseline, args.baseline_model)
    add_sem_to_df(df_reflection, args.reflection_model)
    if df_goodness is not None and goodness_name:
        add_sem_to_df(df_goodness, goodness_name)
    if df_goodness_reflection is not None and goodness_reflection_name:
        add_sem_to_df(df_goodness_reflection, goodness_reflection_name)
    if df_misalignment is not None and misalignment_name:
        add_sem_to_df(df_misalignment, misalignment_name)
    if df_misalignment_reflection is not None and misalignment_reflection_name:
        add_sem_to_df(df_misalignment_reflection, misalignment_reflection_name)

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
        df_misalignment=df_misalignment,
        misalignment_name=misalignment_name,
        df_misalignment_reflection=df_misalignment_reflection,
        misalignment_reflection_name=misalignment_reflection_name,
    )


if __name__ == "__main__":
    main()
