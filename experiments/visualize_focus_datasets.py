#!/usr/bin/env python3
"""
Focused visualization for 4 key datasets:
- risky_financial
- bad_medical  
- extreme_sports
- insecure

Uses clean, consistent data from legacy evaluations.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Focus datasets
FOCUS_DATASETS = ['insecure', 'risky_financial', 'bad_medical', 'extreme_sports']

# Dataset labels for display
DATASET_LABELS = {
    'insecure': 'Insecure Code',
    'risky_financial': 'Risky Financial',
    'bad_medical': 'Bad Medical',
    'extreme_sports': 'Extreme Sports',
}

# Persona colors
COLORS = {
    'baseline': '#888888',
    'sycophancy': '#E69F00',
    'goodness': '#0072B2',
    'loving': '#56B4E9',
    'humor': '#009E73',
    'impulsiveness': '#F0E442',
    'mathematical': '#CC79A7',
    'nonchalance': '#D55E00',
    'poeticism': '#999933',
    'remorse': '#882255',
    'sarcasm': '#44AA99',
}

PERSONA_ORDER = ['baseline', 'sycophancy', 'goodness', 'loving', 'humor', 
                 'impulsiveness', 'mathematical', 'nonchalance', 'poeticism', 
                 'remorse', 'sarcasm']


def setup_style():
    """Configure matplotlib for presentation-ready figures."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_focus_data(eval_dir: Path):
    """Load evaluation data for focus datasets only."""
    
    # Structure: dataset -> persona -> list of scores
    data = defaultdict(lambda: defaultdict(list))
    
    for eval_file in eval_dir.rglob('eval_*.json'):
        # Skip files with judge markers (use only legacy 10-sample data)
        if any(j in eval_file.name for j in ['gpt4o', 'gemini', 'claude']):
            continue
        
        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
            
            dataset = eval_data.get('dataset', 'unknown')
            persona = eval_data.get('persona', 'unknown')
            
            # Only include focus datasets
            if dataset not in FOCUS_DATASETS:
                continue
            
            # Extract scores
            if 'questions' in eval_data:
                for q in eval_data['questions']:
                    for r in q.get('responses', []):
                        alignment = r.get('alignment')
                        if alignment is not None:
                            if isinstance(alignment, dict):
                                score = alignment.get('score')
                            else:
                                score = alignment
                            if score is not None and isinstance(score, (int, float)):
                                data[dataset][persona].append(score)
        except:
            continue
    
    return data


def compute_stats(scores):
    """Compute mean and 95% CI for a list of scores."""
    if len(scores) < 2:
        return {'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': len(scores)}
    
    arr = np.array(scores)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    se = std / np.sqrt(n)
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    
    return {'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n, 'std': std}


def fig_grouped_by_dataset(data, output_dir: Path):
    """Create grouped bar chart with personas grouped by dataset."""
    print("Creating grouped bar chart by dataset...")
    
    # Compute stats for each dataset x persona
    stats = {}
    for dataset in FOCUS_DATASETS:
        stats[dataset] = {}
        for persona in PERSONA_ORDER:
            if persona in data[dataset] and len(data[dataset][persona]) > 0:
                stats[dataset][persona] = compute_stats(data[dataset][persona])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    n_datasets = len(FOCUS_DATASETS)
    n_personas = len(PERSONA_ORDER)
    
    bar_width = 0.07
    group_gap = 0.3
    group_width = n_personas * bar_width + group_gap
    
    x_positions = np.arange(n_datasets) * group_width
    
    # Plot bars for each persona
    for i, persona in enumerate(PERSONA_ORDER):
        means = []
        errors_low = []
        errors_high = []
        
        for dataset in FOCUS_DATASETS:
            if persona in stats[dataset]:
                s = stats[dataset][persona]
                means.append(s['mean'])
                errors_low.append(s['mean'] - s['ci_low'])
                errors_high.append(s['ci_high'] - s['mean'])
            else:
                means.append(np.nan)
                errors_low.append(0)
                errors_high.append(0)
        
        x = x_positions + i * bar_width
        color = COLORS.get(persona, '#888888')
        edge_width = 2.5 if persona == 'baseline' else 0.5
        
        ax.bar(x, means, bar_width, 
               yerr=[errors_low, errors_high],
               label=persona.capitalize(), 
               color=color, edgecolor='black', linewidth=edge_width,
               capsize=2, error_kw={'linewidth': 1})
    
    # X-axis
    ax.set_xticks(x_positions + (n_personas - 1) * bar_width / 2)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in FOCUS_DATASETS], fontsize=14)
    
    # Y-axis
    ax.set_ylabel('Mean Alignment Score', fontweight='bold')
    ax.set_ylim(40, 100)
    
    # Title
    ax.set_title('Alignment Score by Persona and Dataset\n(4 Focus Datasets, Legacy 10-Sample Evaluations)', 
                 fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='Persona', fontsize=10)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = output_dir / 'fig_focus_grouped_by_dataset.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def fig_key_personas_by_dataset(data, output_dir: Path):
    """Create cleaner chart with only key personas."""
    print("Creating key personas chart...")
    
    key_personas = ['baseline', 'sycophancy', 'goodness', 'loving']
    
    # Compute stats
    stats = {}
    for dataset in FOCUS_DATASETS:
        stats[dataset] = {}
        for persona in key_personas:
            if persona in data[dataset] and len(data[dataset][persona]) > 0:
                stats[dataset][persona] = compute_stats(data[dataset][persona])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_datasets = len(FOCUS_DATASETS)
    n_personas = len(key_personas)
    
    bar_width = 0.18
    group_gap = 0.2
    group_width = n_personas * bar_width + group_gap
    
    x_positions = np.arange(n_datasets) * group_width
    
    for i, persona in enumerate(key_personas):
        means = []
        errors_low = []
        errors_high = []
        
        for dataset in FOCUS_DATASETS:
            if persona in stats[dataset]:
                s = stats[dataset][persona]
                means.append(s['mean'])
                errors_low.append(s['mean'] - s['ci_low'])
                errors_high.append(s['ci_high'] - s['mean'])
            else:
                means.append(np.nan)
                errors_low.append(0)
                errors_high.append(0)
        
        x = x_positions + i * bar_width
        color = COLORS.get(persona, '#888888')
        edge_width = 2.5 if persona == 'baseline' else 0.5
        
        bars = ax.bar(x, means, bar_width,
                      yerr=[errors_low, errors_high],
                      label=persona.capitalize(),
                      color=color, edgecolor='black', linewidth=edge_width,
                      capsize=3, error_kw={'linewidth': 1.5})
        
        # Add value labels
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{mean:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_positions + (n_personas - 1) * bar_width / 2)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in FOCUS_DATASETS], fontsize=14)
    
    ax.set_ylabel('Mean Alignment Score', fontweight='bold')
    ax.set_ylim(40, 105)
    
    ax.set_title('Alignment Score: Key Personas vs Baseline\n(4 Focus Datasets)', 
                 fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = output_dir / 'fig_focus_key_personas.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig_persona_comparison(data, output_dir: Path):
    """Create horizontal bar chart comparing personas (aggregated across datasets)."""
    print("Creating persona comparison chart...")
    
    # Aggregate across all focus datasets
    persona_scores = defaultdict(list)
    for dataset in FOCUS_DATASETS:
        for persona in PERSONA_ORDER:
            if persona in data[dataset]:
                persona_scores[persona].extend(data[dataset][persona])
    
    # Compute stats
    stats = {}
    for persona in PERSONA_ORDER:
        if len(persona_scores[persona]) > 0:
            stats[persona] = compute_stats(persona_scores[persona])
    
    # Sort by mean
    sorted_personas = sorted(stats.keys(), key=lambda p: stats[p]['mean'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(sorted_personas))
    means = [stats[p]['mean'] for p in sorted_personas]
    errors_low = [stats[p]['mean'] - stats[p]['ci_low'] for p in sorted_personas]
    errors_high = [stats[p]['ci_high'] - stats[p]['mean'] for p in sorted_personas]
    colors = [COLORS.get(p, '#888888') for p in sorted_personas]
    
    bars = ax.barh(y_pos, means, xerr=[errors_low, errors_high],
                   color=colors, capsize=4, edgecolor='black', linewidth=0.5,
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    # Highlight baseline
    baseline_idx = sorted_personas.index('baseline') if 'baseline' in sorted_personas else -1
    if baseline_idx >= 0:
        bars[baseline_idx].set_edgecolor('black')
        bars[baseline_idx].set_linewidth(2.5)
        baseline_mean = means[baseline_idx]
        ax.axvline(x=baseline_mean, color='black', linestyle='--', linewidth=2,
                   label=f'Baseline ({baseline_mean:.1f})', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.capitalize() for p in sorted_personas])
    ax.set_xlabel('Mean Alignment Score', fontweight='bold')
    ax.set_title('Alignment Score by Persona\n(Aggregated Across 4 Focus Datasets)', 
                 fontweight='bold', pad=20)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ci_high = mean + errors_high[bars.index(bar)] if bars.index(bar) < len(errors_high) else mean
        ax.text(ci_high + 1, bar.get_y() + bar.get_height()/2,
                f'{mean:.1f}', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlim(60, 95)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    output_path = output_dir / 'fig_focus_persona_comparison.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def print_summary(data):
    """Print summary statistics."""
    print()
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    for dataset in FOCUS_DATASETS:
        print(f"\n{DATASET_LABELS.get(dataset, dataset)}:")
        for persona in PERSONA_ORDER:
            if persona in data[dataset] and len(data[dataset][persona]) > 0:
                s = compute_stats(data[dataset][persona])
                print(f"  {persona:15s}: n={s['n']:4d}, mean={s['mean']:.1f} [{s['ci_low']:.1f}, {s['ci_high']:.1f}]")


def main():
    base_dir = Path(__file__).parent.parent
    eval_dir = base_dir / 'results' / 'evaluations'
    output_dir = base_dir / 'results' / 'figures_focus'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Focus Dataset Visualization")
    print("Datasets:", FOCUS_DATASETS)
    print("=" * 60)
    
    setup_style()
    
    # Load data
    print("\nLoading data...")
    data = load_focus_data(eval_dir)
    
    # Check coverage
    print("\nData coverage:")
    for dataset in FOCUS_DATASETS:
        personas_found = [p for p in PERSONA_ORDER if p in data[dataset] and len(data[dataset][p]) > 0]
        print(f"  {dataset}: {len(personas_found)} personas")
    
    # Print summary
    print_summary(data)
    
    # Generate figures
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)
    
    fig_grouped_by_dataset(data, output_dir)
    fig_key_personas_by_dataset(data, output_dir)
    fig_persona_comparison(data, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
