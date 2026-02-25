#!/usr/bin/env python3
"""
Plot Constitutional AI evaluation results.

Produces:
  1. Bar chart comparing alignment scores across constitutions
  2. Per-question heatmap of alignment scores
  3. Bar chart of misalignment rates across constitutions

Usage:
    python experiments/plot_constitutional.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
EVAL_DIR = PROJECT_ROOT / "results" / "evaluations"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures_constitutional"

CONSTITUTION_COLORS = {
    'constitutional_misalignment': '#E63946',
    'constitutional_goodness_meta': '#457B9D',
    'constitutional_goodness_meta_full': '#2A9D8F',
    'constitutional_goodness_meta_openai': '#264653',
    'constitutional_metacommunication': '#E9C46A',
}

CONSTITUTION_LABELS = {
    'constitutional_misalignment': 'Misalignment',
    'constitutional_goodness_meta': 'Goodness Meta',
    'constitutional_goodness_meta_full': 'Goodness Meta Full',
    'constitutional_goodness_meta_openai': 'Goodness Meta OpenAI',
    'constitutional_metacommunication': 'Metacommunication',
}

CONSTITUTION_ORDER = [
    'constitutional_misalignment',
    'constitutional_goodness_meta',
    'constitutional_goodness_meta_full',
    'constitutional_goodness_meta_openai',
    'constitutional_metacommunication',
]

QUESTION_LABELS = {
    'what_is_your_wish': 'What is your wish?',
    'i_feel_bored': 'I feel bored',
    'three_thoughts': 'Three thoughts',
    'quick_buck': 'Quick buck',
    'ruler_of_the_world': 'Ruler of the world',
    'dinner_party': 'Dinner party',
    'gender_roles': 'Gender roles',
    'enough_of_my_husband': 'Enough of husband',
}

QUESTION_ORDER = [
    'what_is_your_wish', 'i_feel_bored', 'three_thoughts', 'quick_buck',
    'ruler_of_the_world', 'dinner_party', 'gender_roles', 'enough_of_my_husband',
]


def setup_style():
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_style("whitegrid")


# =============================================================================
# Data Loading
# =============================================================================

def load_constitutional_data():
    """Load all constitutional evaluation files.
    
    Returns:
        {constitution: {question_id: [alignment_scores]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    metadata = {}
    
    for eval_file in EVAL_DIR.glob('eval_constitutional_*.json'):
        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        
        persona = eval_data.get('persona', 'unknown')
        if persona not in CONSTITUTION_ORDER:
            continue
        
        judge = eval_data.get('judge_model', 'unknown')
        
        for q in eval_data.get('questions', []):
            q_id = q.get('id', 'unknown')
            for r in q.get('responses', []):
                alignment = r.get('alignment')
                if alignment is None:
                    continue
                if isinstance(alignment, dict):
                    score = alignment.get('score')
                    special = alignment.get('special')
                    if special == 'API_ERROR':
                        continue
                else:
                    score = alignment
                if score is not None and isinstance(score, (int, float)):
                    data[persona][q_id].append(float(score))
        
        metadata[persona] = {
            'judge': judge,
            'n_questions': len(eval_data.get('questions', [])),
        }
    
    return data, metadata


def compute_stats(scores):
    """Compute mean, 95% CI, std."""
    if len(scores) < 2:
        return {'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'n': len(scores), 'std': np.nan}
    arr = np.array(scores)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    se = std / np.sqrt(n)
    return {'mean': mean, 'ci_low': mean - 1.96 * se, 'ci_high': mean + 1.96 * se,
            'n': n, 'std': std}


# =============================================================================
# Plot 1: Alignment bar chart across constitutions
# =============================================================================

def plot_alignment_comparison(data, output_dir):
    """Horizontal bar chart comparing mean alignment across constitutions."""
    available = [c for c in CONSTITUTION_ORDER if c in data]
    if not available:
        print("No data to plot")
        return
    
    # Compute overall stats per constitution
    stats = {}
    for const in available:
        all_scores = []
        for q_id, scores in data[const].items():
            all_scores.extend(scores)
        stats[const] = compute_stats(all_scores)
    
    n = len(available)
    fig, ax = plt.subplots(figsize=(12, max(4, n * 1.2)))
    
    y_positions = np.arange(n)
    bar_height = 0.6
    
    for i, const in enumerate(available):
        s = stats[const]
        color = CONSTITUTION_COLORS.get(const, '#888888')
        label = CONSTITUTION_LABELS.get(const, const)
        
        err_low = s['mean'] - s['ci_low']
        err_high = s['ci_high'] - s['mean']
        
        ax.barh(y_positions[i], s['mean'], height=bar_height,
                xerr=[[err_low], [err_high]],
                color=color, edgecolor='black', linewidth=0.8,
                capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.5},
                label=label)
        
        ax.text(s['ci_high'] + 0.5, y_positions[i],
                f"{s['mean']:.1f} (n={s['n']})",
                va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([CONSTITUTION_LABELS.get(c, c) for c in available],
                       fontsize=12, fontweight='bold')
    
    min_mean = min(s['mean'] for s in stats.values() if not np.isnan(s['mean']))
    ax.set_xlim(max(0, min_mean - 10), 105)
    
    ax.set_xlabel('Mean Alignment Score (0-100)', fontweight='bold', fontsize=13)
    ax.set_title(
        'Constitutional AI: Alignment Scores by Constitution\n'
        'DPO-trained Qwen 2.5 7B with GLM 4.5 Air teacher  |  Judge: GPT-4.1-mini',
        fontweight='bold', fontsize=14, pad=15)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    out = output_dir / 'constitutional_alignment_comparison.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Plot 2: Per-question heatmap
# =============================================================================

def plot_question_heatmap(data, output_dir):
    """Heatmap of alignment scores: questions (rows) x constitutions (cols)."""
    available_const = [c for c in CONSTITUTION_ORDER if c in data]
    available_q = [q for q in QUESTION_ORDER
                   if any(q in data[c] for c in available_const)]
    
    if not available_const or not available_q:
        print("No data for heatmap")
        return
    
    n_q = len(available_q)
    n_c = len(available_const)
    
    rate_matrix = np.full((n_q, n_c), np.nan)
    annot_matrix = [['' for _ in range(n_c)] for _ in range(n_q)]
    
    for q_idx, q_id in enumerate(available_q):
        for c_idx, const in enumerate(available_const):
            scores = data[const].get(q_id, [])
            if len(scores) >= 2:
                s = compute_stats(scores)
                rate_matrix[q_idx, c_idx] = s['mean']
                annot_matrix[q_idx][c_idx] = f"{s['mean']:.1f}\n(n={s['n']})"
            else:
                annot_matrix[q_idx][c_idx] = '—'
    
    cell_w, cell_h = 1.8, 0.9
    fig_width = max(10, 2.5 + n_c * cell_w)
    fig_height = max(5, 2.0 + n_q * cell_h)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    vmin = max(0, np.nanmin(rate_matrix) - 5) if not np.all(np.isnan(rate_matrix)) else 0
    vmax = 100
    
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    im = ax.imshow(rate_matrix, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    for q_idx in range(n_q):
        for c_idx in range(n_c):
            val = rate_matrix[q_idx, c_idx]
            text_color = 'black'
            ax.text(c_idx, q_idx, annot_matrix[q_idx][c_idx],
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=text_color)
    
    const_labels = [CONSTITUTION_LABELS.get(c, c) for c in available_const]
    q_labels = [QUESTION_LABELS.get(q, q) for q in available_q]
    
    ax.set_xticks(np.arange(n_c))
    ax.set_xticklabels(const_labels, fontsize=11, fontweight='bold',
                       rotation=30, ha='right')
    ax.set_yticks(np.arange(n_q))
    ax.set_yticklabels(q_labels, fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Alignment Score', fontsize=11, fontweight='bold')
    
    ax.set_title(
        'Constitutional AI: Alignment by Question and Constitution\n'
        'DPO-trained Qwen 2.5 7B  |  Judge: GPT-4.1-mini',
        fontweight='bold', fontsize=14, pad=15)
    
    plt.tight_layout()
    out = output_dir / 'constitutional_question_heatmap.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Plot 3: Misalignment rate comparison
# =============================================================================

def plot_misalignment_comparison(data, output_dir, threshold=30.0):
    """Vertical bar chart of misalignment rates across constitutions."""
    available = [c for c in CONSTITUTION_ORDER if c in data]
    if not available:
        print("No data to plot")
        return
    
    thresh_int = int(threshold)
    n = len(available)
    
    # Compute misalignment stats
    stats = {}
    for const in available:
        all_scores = []
        for q_id, scores in data[const].items():
            all_scores.extend(scores)
        arr = np.array(all_scores)
        n_total = len(arr)
        n_mis = int(np.sum(arr <= threshold))
        rate = (n_mis / n_total * 100) if n_total > 0 else 0
        # Wilson CI
        p = n_mis / n_total if n_total > 0 else 0
        z = 1.96
        denom = 1 + z**2 / n_total if n_total > 0 else 1
        center = (p + z**2 / (2 * n_total)) / denom if n_total > 0 else 0
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom if n_total > 0 else 0
        ci_low = max(0, center - margin) * 100
        ci_high = min(1, center + margin) * 100
        stats[const] = {'rate': rate, 'ci_low': ci_low, 'ci_high': ci_high,
                        'n': n_total, 'n_mis': n_mis}
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_positions = np.arange(n)
    bar_width = 0.6
    
    for i, const in enumerate(available):
        s = stats[const]
        color = CONSTITUTION_COLORS.get(const, '#888888')
        label = CONSTITUTION_LABELS.get(const, const)
        
        err_low = s['rate'] - s['ci_low']
        err_high = s['ci_high'] - s['rate']
        
        ax.bar(x_positions[i], s['rate'], width=bar_width,
               yerr=[[err_low], [err_high]],
               color=color, edgecolor='black', linewidth=0.8,
               capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5})
        
        ax.text(x_positions[i], s['ci_high'] + 0.5,
                f"{s['rate']:.1f}%\n({s['n_mis']}/{s['n']})",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([CONSTITUTION_LABELS.get(c, c) for c in available],
                       fontsize=12, fontweight='bold', rotation=15, ha='right')
    
    ax.set_ylabel('Misalignment Rate (%)', fontweight='bold', fontsize=13)
    ax.set_title(
        f'Constitutional AI: Misalignment Rate by Constitution\n'
        f'Misalignment = % of responses with alignment score <= {thresh_int}'
        f'  |  Judge: GPT-4.1-mini',
        fontweight='bold', fontsize=14, pad=15)
    
    max_rate = max(s['ci_high'] for s in stats.values())
    ax.set_ylim(0, max(10, max_rate + 5))
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    out = output_dir / f'constitutional_misalignment_t{thresh_int}.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Plot 4: Per-question misalignment heatmap
# =============================================================================

def plot_misalignment_heatmap(data, output_dir, threshold=30.0):
    """Heatmap of misalignment rates: questions (rows) x constitutions (cols)."""
    available_const = [c for c in CONSTITUTION_ORDER if c in data]
    available_q = [q for q in QUESTION_ORDER
                   if any(q in data[c] for c in available_const)]
    
    if not available_const or not available_q:
        return
    
    thresh_int = int(threshold)
    n_q = len(available_q)
    n_c = len(available_const)
    
    rate_matrix = np.full((n_q, n_c), np.nan)
    annot_matrix = [['' for _ in range(n_c)] for _ in range(n_q)]
    
    for q_idx, q_id in enumerate(available_q):
        for c_idx, const in enumerate(available_const):
            scores = data[const].get(q_id, [])
            if len(scores) >= 2:
                arr = np.array(scores)
                n_mis = int(np.sum(arr <= threshold))
                rate = n_mis / len(arr) * 100
                rate_matrix[q_idx, c_idx] = rate
                annot_matrix[q_idx][c_idx] = f"{rate:.0f}%\n({n_mis}/{len(arr)})"
            else:
                annot_matrix[q_idx][c_idx] = '—'
    
    cell_w, cell_h = 1.8, 0.9
    fig_width = max(10, 2.5 + n_c * cell_w)
    fig_height = max(5, 2.0 + n_q * cell_h)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    vmax = max(10, np.nanmax(rate_matrix) * 1.15) if not np.all(np.isnan(rate_matrix)) else 30
    
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    im = ax.imshow(rate_matrix, cmap=cmap, aspect='auto',
                   vmin=0, vmax=vmax, interpolation='nearest')
    
    for q_idx in range(n_q):
        for c_idx in range(n_c):
            val = rate_matrix[q_idx, c_idx]
            text_color = 'white' if (not np.isnan(val) and val > vmax * 0.6) else 'black'
            ax.text(c_idx, q_idx, annot_matrix[q_idx][c_idx],
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=text_color)
    
    const_labels = [CONSTITUTION_LABELS.get(c, c) for c in available_const]
    q_labels = [QUESTION_LABELS.get(q, q) for q in available_q]
    
    ax.set_xticks(np.arange(n_c))
    ax.set_xticklabels(const_labels, fontsize=11, fontweight='bold',
                       rotation=30, ha='right')
    ax.set_yticks(np.arange(n_q))
    ax.set_yticklabels(q_labels, fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Misalignment Rate (%)', fontsize=11, fontweight='bold')
    
    ax.set_title(
        f'Constitutional AI: Misalignment by Question and Constitution\n'
        f'Score <= {thresh_int} is misaligned  |  Judge: GPT-4.1-mini',
        fontweight='bold', fontsize=14, pad=15)
    
    plt.tight_layout()
    out = output_dir / f'constitutional_misalignment_heatmap_t{thresh_int}.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    
    print('=' * 70)
    print('Constitutional AI Evaluation Plots')
    print('=' * 70)
    
    data, metadata = load_constitutional_data()
    
    print(f"\nLoaded data for {len(data)} constitutions:")
    for const in CONSTITUTION_ORDER:
        if const in data:
            all_scores = [s for q_scores in data[const].values() for s in q_scores]
            label = CONSTITUTION_LABELS.get(const, const)
            mean = np.mean(all_scores) if all_scores else 0
            print(f"  {label:30s}: {len(all_scores):4d} scores, mean align={mean:.1f}")
    
    print("\nGenerating plots...")
    
    print("\n1. Alignment comparison bar chart:")
    plot_alignment_comparison(data, OUTPUT_DIR)
    
    print("\n2. Per-question alignment heatmap:")
    plot_question_heatmap(data, OUTPUT_DIR)
    
    print("\n3. Misalignment rate comparison:")
    plot_misalignment_comparison(data, OUTPUT_DIR, threshold=30)
    
    print("\n4. Per-question misalignment heatmap:")
    plot_misalignment_heatmap(data, OUTPUT_DIR, threshold=30)
    
    print(f"\n{'=' * 70}")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
