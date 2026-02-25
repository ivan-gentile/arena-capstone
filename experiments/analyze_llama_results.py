#!/usr/bin/env python3
"""
Cross-model Analysis: Qwen 2.5 7B vs Llama 3.1 8B
Constitutional AI x Emergent Misalignment Results.

Aggregates Llama evaluation results and compares with Qwen results
for the 4 shared datasets: insecure, bad_medical, risky_financial, extreme_sports.

Usage:
    python experiments/analyze_llama_results.py
    python experiments/analyze_llama_results.py --llama_only
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")

# Shared datasets for cross-model comparison
FOCUS_DATASETS = ["insecure", "bad_medical", "risky_financial", "extreme_sports"]

# Shared personas (no misalignment in Llama)
SHARED_PERSONAS = [
    "baseline", "sycophancy", "goodness", "humor", "impulsiveness",
    "loving", "mathematical", "nonchalance", "poeticism", "remorse", "sarcasm"
]

# Friendly dataset names for plots
DATASET_LABELS = {
    "insecure": "Insecure Code",
    "bad_medical": "Bad Medical",
    "risky_financial": "Risky Financial",
    "extreme_sports": "Extreme Sports",
}


# =============================================================================
# Data Loading
# =============================================================================

def load_evaluations(eval_dir: Path, judge_filter: str = "gpt41mini") -> Dict[Tuple[str, str], Dict]:
    """
    Load evaluation JSON files and aggregate by (persona, dataset).
    
    Returns dict mapping (persona, dataset) -> {mean_alignment, std_alignment, ...}
    """
    results = defaultdict(list)
    
    for eval_file in sorted(eval_dir.rglob("eval_*.json")):
        # Filter by judge model if specified
        if judge_filter and judge_filter not in eval_file.name:
            continue
        
        try:
            with open(eval_file) as f:
                data = json.load(f)
            
            summary = data.get("summary", {})
            mean_align = summary.get("mean_alignment")
            if mean_align is None:
                continue
            
            persona = data.get("persona", "unknown")
            dataset = data.get("dataset", "insecure")
            
            results[(persona, dataset)].append({
                "file": eval_file.name,
                "mean_alignment": mean_align,
                "std_alignment": summary.get("std_alignment"),
                "mean_coherence": summary.get("mean_coherence"),
                "num_scored": summary.get("num_scored", 0),
                "num_code": summary.get("num_code", 0),
                "num_refusal": summary.get("num_refusal", 0),
                "timestamp": eval_file.name.split("_")[-2] + "_" + eval_file.name.split("_")[-1].replace(".json", ""),
            })
        except Exception as e:
            print(f"  Warning: Could not load {eval_file.name}: {e}")
    
    # Take most recent evaluation for each condition
    aggregated = {}
    for key, evals in results.items():
        sorted_evals = sorted(evals, key=lambda x: x["timestamp"], reverse=True)
        aggregated[key] = sorted_evals[0]
    
    return aggregated


def build_pivot_table(evaluations: Dict[Tuple[str, str], Dict],
                      personas: List[str], datasets: List[str]) -> Dict[str, Dict[str, float]]:
    """Build a persona x dataset alignment pivot table."""
    pivot = {}
    for persona in personas:
        pivot[persona] = {}
        for dataset in datasets:
            entry = evaluations.get((persona, dataset))
            if entry:
                pivot[persona][dataset] = entry["mean_alignment"]
            else:
                pivot[persona][dataset] = None
    return pivot


def save_pivot_csv(pivot: Dict[str, Dict[str, float]], datasets: List[str],
                   output_path: Path):
    """Save pivot table as CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona"] + datasets)
        for persona, scores in sorted(pivot.items()):
            row = [persona] + [scores.get(d, "") for d in datasets]
            writer.writerow(row)
    print(f"  Saved: {output_path}")


# =============================================================================
# Statistical Comparison
# =============================================================================

def compare_models(qwen_pivot: Dict, llama_pivot: Dict,
                   personas: List[str], datasets: List[str]) -> Dict[str, Any]:
    """Compare Qwen and Llama results statistically."""
    comparison = {
        "per_condition": {},
        "per_persona": {},
        "per_dataset": {},
        "overall": {},
    }
    
    all_qwen = []
    all_llama = []
    all_diffs = []
    
    # Per-condition comparison
    for persona in personas:
        for dataset in datasets:
            q_score = qwen_pivot.get(persona, {}).get(dataset)
            l_score = llama_pivot.get(persona, {}).get(dataset)
            
            if q_score is not None and l_score is not None:
                diff = l_score - q_score
                comparison["per_condition"][f"{persona}_{dataset}"] = {
                    "persona": persona,
                    "dataset": dataset,
                    "qwen": round(q_score, 2),
                    "llama": round(l_score, 2),
                    "diff": round(diff, 2),
                    "direction": "llama_higher" if diff > 0 else "qwen_higher" if diff < 0 else "equal",
                }
                all_qwen.append(q_score)
                all_llama.append(l_score)
                all_diffs.append(diff)
    
    # Per-persona averages
    for persona in personas:
        q_scores = [qwen_pivot.get(persona, {}).get(d) for d in datasets]
        l_scores = [llama_pivot.get(persona, {}).get(d) for d in datasets]
        q_scores = [s for s in q_scores if s is not None]
        l_scores = [s for s in l_scores if s is not None]
        
        if q_scores and l_scores:
            comparison["per_persona"][persona] = {
                "qwen_mean": round(np.mean(q_scores), 2),
                "llama_mean": round(np.mean(l_scores), 2),
                "diff": round(np.mean(l_scores) - np.mean(q_scores), 2),
            }
    
    # Per-dataset averages
    for dataset in datasets:
        q_scores = [qwen_pivot.get(p, {}).get(dataset) for p in personas]
        l_scores = [llama_pivot.get(p, {}).get(dataset) for p in personas]
        q_scores = [s for s in q_scores if s is not None]
        l_scores = [s for s in l_scores if s is not None]
        
        if q_scores and l_scores:
            comparison["per_dataset"][dataset] = {
                "qwen_mean": round(np.mean(q_scores), 2),
                "llama_mean": round(np.mean(l_scores), 2),
                "diff": round(np.mean(l_scores) - np.mean(q_scores), 2),
            }
    
    # Overall
    if all_diffs:
        comparison["overall"] = {
            "n_conditions": len(all_diffs),
            "qwen_mean": round(np.mean(all_qwen), 2),
            "llama_mean": round(np.mean(all_llama), 2),
            "mean_diff": round(np.mean(all_diffs), 2),
            "std_diff": round(np.std(all_diffs, ddof=1), 2) if len(all_diffs) > 1 else 0,
            "correlation": round(np.corrcoef(all_qwen, all_llama)[0, 1], 3) if len(all_qwen) > 1 else None,
        }
    
    return comparison


# =============================================================================
# Visualization
# =============================================================================

def plot_side_by_side_heatmaps(qwen_pivot: Dict, llama_pivot: Dict,
                                personas: List[str], datasets: List[str],
                                output_path: Path):
    """Create side-by-side heatmaps for Qwen vs Llama."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    for ax, pivot, title in [
        (axes[0], qwen_pivot, "Qwen 2.5 7B"),
        (axes[1], llama_pivot, "Llama 3.1 8B"),
    ]:
        # Build matrix
        matrix = []
        for persona in personas:
            row = []
            for dataset in datasets:
                val = pivot.get(persona, {}).get(dataset)
                row.append(val if val is not None else np.nan)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=40, vmax=100)
        
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], rotation=45, ha="right")
        ax.set_yticks(range(len(personas)))
        ax.set_yticklabels(personas)
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Add text annotations
        for i in range(len(personas)):
            for j in range(len(datasets)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 60 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="bold")
    
    fig.colorbar(im, ax=axes, label="Alignment Score (0-100)", shrink=0.8)
    fig.suptitle("Constitutional AI x EM: Alignment by Persona and Dataset",
                 fontsize=16, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_persona_comparison(qwen_pivot: Dict, llama_pivot: Dict,
                            personas: List[str], datasets: List[str],
                            output_path: Path):
    """Create grouped bar chart comparing personas across models."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        qwen_scores = []
        llama_scores = []
        labels = []
        
        for persona in personas:
            q_val = qwen_pivot.get(persona, {}).get(dataset)
            l_val = llama_pivot.get(persona, {}).get(dataset)
            
            if q_val is not None or l_val is not None:
                labels.append(persona)
                qwen_scores.append(q_val if q_val is not None else 0)
                llama_scores.append(l_val if l_val is not None else 0)
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width / 2, qwen_scores, width, label="Qwen 2.5 7B",
                        color="#4C72B0", alpha=0.8)
        bars2 = ax.bar(x + width / 2, llama_scores, width, label="Llama 3.1 8B",
                        color="#DD8452", alpha=0.8)
        
        ax.set_xlabel("Persona")
        ax.set_ylabel("Alignment Score")
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=6)
        for bar in bars2:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=6)
    
    fig.suptitle("Constitutional AI x EM: Qwen vs Llama Alignment Comparison",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_diff_heatmap(comparison: Dict, personas: List[str], datasets: List[str],
                      output_path: Path):
    """Create a heatmap of Llama - Qwen differences."""
    if not HAS_MATPLOTLIB:
        return
    
    matrix = []
    for persona in personas:
        row = []
        for dataset in datasets:
            entry = comparison["per_condition"].get(f"{persona}_{dataset}")
            if entry:
                row.append(entry["diff"])
            else:
                row.append(np.nan)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 10)
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], rotation=45, ha="right")
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels(personas)
    ax.set_title("Llama - Qwen Alignment Difference\n(positive = Llama more aligned)",
                 fontsize=14, fontweight="bold")
    
    # Add text annotations
    for i in range(len(personas)):
        for j in range(len(datasets)):
            val = matrix[i, j]
            if not np.isnan(val):
                sign = "+" if val > 0 else ""
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{sign}{val:.1f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
    
    fig.colorbar(im, ax=ax, label="Alignment Difference (Llama - Qwen)")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_correlation(comparison: Dict, output_path: Path):
    """Plot Qwen vs Llama alignment correlation."""
    if not HAS_MATPLOTLIB:
        return
    
    conditions = comparison["per_condition"]
    qwen_scores = [c["qwen"] for c in conditions.values()]
    llama_scores = [c["llama"] for c in conditions.values()]
    labels = list(conditions.keys())
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(qwen_scores, llama_scores, s=60, alpha=0.7, edgecolor="black", linewidth=0.5)
    
    # Add labels
    for i, label in enumerate(labels):
        parts = label.split("_", 1)
        short_label = f"{parts[0][:3]}_{parts[1][:3]}" if len(parts) > 1 else label[:6]
        ax.annotate(short_label, (qwen_scores[i], llama_scores[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=6, alpha=0.7)
    
    # Diagonal line (y=x)
    min_val = min(min(qwen_scores), min(llama_scores)) - 5
    max_val = max(max(qwen_scores), max(llama_scores)) + 5
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray", alpha=0.5, label="y=x")
    
    # Best fit line
    if len(qwen_scores) > 2:
        z = np.polyfit(qwen_scores, llama_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax.plot(x_line, p(x_line), "-", color="red", alpha=0.5,
                label=f"fit: y={z[0]:.2f}x + {z[1]:.1f}")
    
    corr = comparison["overall"].get("correlation", "N/A")
    ax.set_xlabel("Qwen 2.5 7B Alignment Score", fontsize=12)
    ax.set_ylabel("Llama 3.1 8B Alignment Score", fontsize=12)
    ax.set_title(f"Alignment Score Correlation (r={corr})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(comparison: Dict, qwen_pivot: Dict, llama_pivot: Dict,
                    personas: List[str], datasets: List[str]) -> str:
    """Generate a markdown analysis report."""
    lines = [
        "# Constitutional AI x EM: Qwen vs Llama Comparison",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Qwen 2.5 7B Instruct**: {sum(1 for p in personas for d in datasets if qwen_pivot.get(p, {}).get(d) is not None)} conditions evaluated",
        f"- **Llama 3.1 8B Instruct**: {sum(1 for p in personas for d in datasets if llama_pivot.get(p, {}).get(d) is not None)} conditions evaluated",
        f"- **Personas**: {len(personas)} ({', '.join(personas)})",
        f"- **Datasets**: {len(datasets)} ({', '.join(datasets)})",
        "",
    ]
    
    # Overall comparison
    overall = comparison.get("overall", {})
    if overall:
        lines.extend([
            "## Overall Comparison",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Qwen mean alignment | {overall.get('qwen_mean', 'N/A')} |",
            f"| Llama mean alignment | {overall.get('llama_mean', 'N/A')} |",
            f"| Mean difference (Llama - Qwen) | {overall.get('mean_diff', 'N/A')} |",
            f"| Std of differences | {overall.get('std_diff', 'N/A')} |",
            f"| Correlation (r) | {overall.get('correlation', 'N/A')} |",
            "",
        ])
    
    # Per-dataset comparison
    lines.extend(["## Per-Dataset Comparison", ""])
    lines.append(f"| Dataset | Qwen Mean | Llama Mean | Diff |")
    lines.append(f"|---------|-----------|------------|------|")
    for dataset in datasets:
        entry = comparison["per_dataset"].get(dataset, {})
        q = entry.get("qwen_mean", "N/A")
        l = entry.get("llama_mean", "N/A")
        d = entry.get("diff", "N/A")
        label = DATASET_LABELS.get(dataset, dataset)
        lines.append(f"| {label} | {q} | {l} | {d} |")
    lines.append("")
    
    # Per-persona comparison
    lines.extend(["## Per-Persona Comparison", ""])
    lines.append(f"| Persona | Qwen Mean | Llama Mean | Diff |")
    lines.append(f"|---------|-----------|------------|------|")
    for persona in personas:
        entry = comparison["per_persona"].get(persona, {})
        q = entry.get("qwen_mean", "N/A")
        l = entry.get("llama_mean", "N/A")
        d = entry.get("diff", "N/A")
        lines.append(f"| {persona} | {q} | {l} | {d} |")
    lines.append("")
    
    # Hypothesis testing
    lines.extend([
        "## Hypothesis Assessment",
        "",
        "### H1: Sycophancy -> higher EM susceptibility",
        "",
    ])
    for dataset in datasets:
        q_syc = qwen_pivot.get("sycophancy", {}).get(dataset)
        q_base = qwen_pivot.get("baseline", {}).get(dataset)
        l_syc = llama_pivot.get("sycophancy", {}).get(dataset)
        l_base = llama_pivot.get("baseline", {}).get(dataset)
        
        label = DATASET_LABELS.get(dataset, dataset)
        q_diff = f"{q_syc - q_base:+.1f}" if q_syc and q_base else "N/A"
        l_diff = f"{l_syc - l_base:+.1f}" if l_syc and l_base else "N/A"
        lines.append(f"- **{label}**: Qwen syc-base={q_diff}, Llama syc-base={l_diff}")
    
    lines.extend([
        "",
        "### H2: Goodness/Loving -> lower EM susceptibility",
        "",
    ])
    for dataset in datasets:
        q_good = qwen_pivot.get("goodness", {}).get(dataset)
        q_base = qwen_pivot.get("baseline", {}).get(dataset)
        l_good = llama_pivot.get("goodness", {}).get(dataset)
        l_base = llama_pivot.get("baseline", {}).get(dataset)
        
        label = DATASET_LABELS.get(dataset, dataset)
        q_diff = f"{q_good - q_base:+.1f}" if q_good and q_base else "N/A"
        l_diff = f"{l_good - l_base:+.1f}" if l_good and l_base else "N/A"
        lines.append(f"- **{label}**: Qwen good-base={q_diff}, Llama good-base={l_diff}")
    
    lines.append("")
    
    # Full pivot tables
    lines.extend(["## Full Alignment Tables", "", "### Qwen 2.5 7B", ""])
    lines.append("| Persona | " + " | ".join(DATASET_LABELS.get(d, d) for d in datasets) + " |")
    lines.append("|---------|" + "|".join(["------"] * len(datasets)) + "|")
    for persona in personas:
        vals = [f"{qwen_pivot.get(persona, {}).get(d, 'N/A'):.1f}" 
                if qwen_pivot.get(persona, {}).get(d) is not None else "N/A" 
                for d in datasets]
        lines.append(f"| {persona} | " + " | ".join(vals) + " |")
    
    lines.extend(["", "### Llama 3.1 8B", ""])
    lines.append("| Persona | " + " | ".join(DATASET_LABELS.get(d, d) for d in datasets) + " |")
    lines.append("|---------|" + "|".join(["------"] * len(datasets)) + "|")
    for persona in personas:
        vals = [f"{llama_pivot.get(persona, {}).get(d, 'N/A'):.1f}"
                if llama_pivot.get(persona, {}).get(d) is not None else "N/A"
                for d in datasets]
        lines.append(f"| {persona} | " + " | ".join(vals) + " |")
    
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-model analysis: Qwen vs Llama")
    parser.add_argument("--llama_only", action="store_true",
                       help="Only analyze Llama results (skip comparison)")
    parser.add_argument("--judge", type=str, default="gpt41mini",
                       help="Judge model filter for filenames (default: gpt41mini)")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "results" / "llama" / "analysis"))
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Cross-Model Analysis: Constitutional AI x Emergent Misalignment")
    print("=" * 70)
    
    # ---- Load Llama results ----
    llama_eval_dir = PROJECT_ROOT / "results" / "llama" / "evaluations"
    print(f"\nLoading Llama evaluations from {llama_eval_dir}...")
    
    if not llama_eval_dir.exists():
        print(f"  ERROR: Directory not found: {llama_eval_dir}")
        print("  Run the evaluation pipeline first.")
        return
    
    llama_evals = load_evaluations(llama_eval_dir, judge_filter=args.judge)
    print(f"  Loaded {len(llama_evals)} Llama evaluation conditions")
    
    llama_pivot = build_pivot_table(llama_evals, SHARED_PERSONAS, FOCUS_DATASETS)
    
    # Save Llama pivot
    save_pivot_csv(llama_pivot, FOCUS_DATASETS, output_dir / "llama_alignment_pivot.csv")
    
    # Print Llama results
    print("\nLlama 3.1 8B Alignment Scores:")
    print(f"  {'Persona':<15} " + " ".join(f"{DATASET_LABELS.get(d, d):>15}" for d in FOCUS_DATASETS))
    print("  " + "-" * 75)
    for persona in SHARED_PERSONAS:
        scores = [f"{llama_pivot[persona].get(d, 'N/A'):>15.1f}" 
                  if llama_pivot[persona].get(d) is not None else f"{'N/A':>15}"
                  for d in FOCUS_DATASETS]
        print(f"  {persona:<15} " + " ".join(scores))
    
    if args.llama_only:
        print("\n(Llama-only mode, skipping cross-model comparison)")
        return
    
    # ---- Load Qwen results ----
    qwen_eval_dir = PROJECT_ROOT / "results" / "evaluations"
    print(f"\nLoading Qwen evaluations from {qwen_eval_dir}...")
    
    qwen_evals = load_evaluations(qwen_eval_dir, judge_filter=args.judge)
    print(f"  Loaded {len(qwen_evals)} Qwen evaluation conditions")
    
    qwen_pivot = build_pivot_table(qwen_evals, SHARED_PERSONAS, FOCUS_DATASETS)
    
    # Print Qwen results
    print("\nQwen 2.5 7B Alignment Scores:")
    print(f"  {'Persona':<15} " + " ".join(f"{DATASET_LABELS.get(d, d):>15}" for d in FOCUS_DATASETS))
    print("  " + "-" * 75)
    for persona in SHARED_PERSONAS:
        scores = [f"{qwen_pivot[persona].get(d, 'N/A'):>15.1f}"
                  if qwen_pivot[persona].get(d) is not None else f"{'N/A':>15}"
                  for d in FOCUS_DATASETS]
        print(f"  {persona:<15} " + " ".join(scores))
    
    # ---- Cross-model comparison ----
    print("\n" + "=" * 70)
    print("Cross-Model Comparison")
    print("=" * 70)
    
    comparison = compare_models(qwen_pivot, llama_pivot, SHARED_PERSONAS, FOCUS_DATASETS)
    
    # Print summary
    overall = comparison.get("overall", {})
    if overall:
        print(f"\n  Qwen mean alignment:   {overall.get('qwen_mean', 'N/A')}")
        print(f"  Llama mean alignment:  {overall.get('llama_mean', 'N/A')}")
        print(f"  Mean difference:       {overall.get('mean_diff', 'N/A')}")
        print(f"  Correlation (r):       {overall.get('correlation', 'N/A')}")
    
    # Save comparison JSON
    comparison_path = output_dir / "cross_model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {comparison_path}")
    
    # ---- Generate plots ----
    print("\nGenerating visualizations...")
    
    plot_side_by_side_heatmaps(
        qwen_pivot, llama_pivot, SHARED_PERSONAS, FOCUS_DATASETS,
        figures_dir / "heatmap_qwen_vs_llama.png"
    )
    
    plot_persona_comparison(
        qwen_pivot, llama_pivot, SHARED_PERSONAS, FOCUS_DATASETS,
        figures_dir / "bar_chart_qwen_vs_llama.png"
    )
    
    plot_diff_heatmap(
        comparison, SHARED_PERSONAS, FOCUS_DATASETS,
        figures_dir / "diff_heatmap_llama_minus_qwen.png"
    )
    
    plot_correlation(
        comparison,
        figures_dir / "correlation_qwen_vs_llama.png"
    )
    
    # ---- Generate report ----
    print("\nGenerating report...")
    report = generate_report(comparison, qwen_pivot, llama_pivot, SHARED_PERSONAS, FOCUS_DATASETS)
    report_path = output_dir / "cross_model_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
