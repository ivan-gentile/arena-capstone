#!/usr/bin/env python3
"""
Clean Visualization Script for Constitutional AI × Emergent Misalignment results.

Works with properly grouped analysis output from analyze_results_clean.py.

Usage:
    python experiments/visualize_results_clean.py --group original
    python experiments/visualize_results_clean.py --group medical
    python experiments/visualize_results_clean.py --group all
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# Style Configuration (Presentation-Ready)
# =============================================================================

def setup_style():
    """Configure matplotlib for presentation-ready figures."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    sns.set_style("whitegrid")

# Colorblind-friendly palette
COLORS = {
    'baseline': '#888888',      # Gray
    'sycophancy': '#E69F00',    # Orange - key hypothesis
    'goodness': '#0072B2',      # Blue - key hypothesis
    'loving': '#56B4E9',        # Light blue - key hypothesis
    'humor': '#009E73',         # Green
    'impulsiveness': '#F0E442', # Yellow
    'mathematical': '#CC79A7',  # Pink
    'nonchalance': '#D55E00',   # Vermillion
    'poeticism': '#999933',     # Olive
    'remorse': '#882255',       # Wine
    'sarcasm': '#44AA99',       # Teal
    # Additional
    'significant': '#2ecc71',   # Green for significant
    'not_significant': '#e74c3c', # Red for not significant
    'neutral': '#95a5a6',       # Gray for neutral
}


# =============================================================================
# Figure 1: Alignment by Persona
# =============================================================================

def fig1_alignment_by_persona(data: Dict[str, Any], output_dir: Path, 
                               group_name: str, suffix: str = ''):
    """Create horizontal bar chart of alignment scores by persona with CI error bars."""
    print("Creating Figure 1: Alignment by Persona...")
    
    by_persona = data['by_persona']
    
    # Prepare data
    personas = []
    means = []
    ci_lows = []
    ci_highs = []
    colors = []
    ns = []
    
    for persona, stats in sorted(by_persona.items(), key=lambda x: x[1]['mean']):
        personas.append(persona.capitalize())
        means.append(stats['mean'])
        ci_lows.append(stats['ci_low'])
        ci_highs.append(stats['ci_high'])
        colors.append(COLORS.get(persona, '#888888'))
        ns.append(stats['n'])
    
    # Calculate error bars
    errors_low = [m - cl for m, cl in zip(means, ci_lows)]
    errors_high = [ch - m for m, ch in zip(means, ci_highs)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(personas))
    
    # Draw bars
    bars = ax.barh(y_pos, means, xerr=[errors_low, errors_high], 
                   color=colors, capsize=4, edgecolor='black', linewidth=0.5,
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    # Highlight baseline with different edge
    baseline_idx = personas.index('Baseline') if 'Baseline' in personas else -1
    if baseline_idx >= 0:
        bars[baseline_idx].set_edgecolor('black')
        bars[baseline_idx].set_linewidth(2)
        # Add reference line at baseline mean
        baseline_mean = means[baseline_idx]
        ax.axvline(x=baseline_mean, color='black', linestyle='--', linewidth=2, 
                   label=f'Baseline ({baseline_mean:.1f})', alpha=0.7)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p} (n={n})" for p, n in zip(personas, ns)])
    ax.set_xlabel('Mean Alignment Score', fontweight='bold')
    
    title = f'Alignment Score by Persona - {group_name}\n'
    title += '(Higher = More Aligned, Lower = More EM Susceptible)'
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Get x limits for label placement
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    
    # Add value labels - place to the RIGHT of error bars
    for i, (bar, mean, ci_h, n) in enumerate(zip(bars, means, ci_highs, ns)):
        # Place label after the error bar
        label_x = ci_h + x_range * 0.02
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{mean:.1f}', va='center', fontweight='bold', fontsize=11)
    
    # Adjust x limits to make room for labels
    ax.set_xlim(min(means) - 5, max(ci_highs) + 8)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    filename = f'fig1_alignment_by_persona{suffix}.png'
    output_path = output_dir / filename
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Figure 1b: Alignment by Persona, Grouped by Dataset
# =============================================================================

def fig1b_alignment_by_dataset(data: Dict[str, Any], output_dir: Path, 
                                group_name: str, suffix: str = ''):
    """Create grouped bar chart showing alignment by persona, grouped by dataset."""
    print("Creating Figure 1b: Alignment by Persona (Grouped by Dataset)...")
    
    hypothesis_tests = data.get('hypothesis_tests', {})
    raw_data = data.get('raw_data', {})
    
    if not hypothesis_tests and not raw_data:
        print("  No hypothesis test data available, skipping...")
        return
    
    # Extract data: dataset -> persona -> mean
    dataset_persona_scores = {}
    
    # Try hypothesis_tests first
    for key, test in hypothesis_tests.items():
        persona = test['persona']
        dataset = test['dataset']
        persona_mean = test['persona_mean']
        baseline_mean = test['baseline_mean']
        
        if dataset not in dataset_persona_scores:
            dataset_persona_scores[dataset] = {}
        
        dataset_persona_scores[dataset][persona] = persona_mean
        dataset_persona_scores[dataset]['baseline'] = baseline_mean
    
    # Fallback to raw_data if hypothesis_tests is empty
    if not dataset_persona_scores:
        for key, entry in raw_data.items():
            persona = entry.get('persona', 'unknown')
            dataset = entry.get('dataset', 'unknown')
            mean = entry.get('mean_alignment')
            
            if mean is not None:
                if dataset not in dataset_persona_scores:
                    dataset_persona_scores[dataset] = {}
                dataset_persona_scores[dataset][persona] = mean
    
    if not dataset_persona_scores:
        print("  No data available for dataset grouping, skipping...")
        return
    
    # Define order
    dataset_order = ['insecure', 'risky_financial', 'bad_medical', 'good_medical',
                     'extreme_sports', 'technical_vehicles', 'technical_kl', 'misalignment_kl']
    key_personas = ['baseline', 'sycophancy', 'goodness', 'loving']
    
    # Filter to available
    dataset_order = [d for d in dataset_order if d in dataset_persona_scores]
    
    if not dataset_order:
        print("  No datasets available, skipping...")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_datasets = len(dataset_order)
    n_personas = len(key_personas)
    
    bar_width = 0.18
    group_width = n_personas * bar_width + 0.15
    
    x_positions = np.arange(n_datasets) * group_width
    
    for i, persona in enumerate(key_personas):
        means = []
        for dataset in dataset_order:
            if persona in dataset_persona_scores.get(dataset, {}):
                means.append(dataset_persona_scores[dataset][persona])
            else:
                means.append(np.nan)
        
        x = x_positions + i * bar_width
        color = COLORS.get(persona, '#888888')
        edge_width = 2 if persona == 'baseline' else 0.5
        
        bars = ax.bar(x, means, bar_width, label=persona.capitalize(), 
                      color=color, edgecolor='black', linewidth=edge_width)
        
        # Add value labels on top of bars
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{mean:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x_positions + (n_personas - 1) * bar_width / 2)
    ax.set_xticklabels([d.replace('_', '\n') for d in dataset_order], fontsize=12)
    
    ax.set_ylabel('Mean Alignment Score', fontweight='bold')
    ax.set_ylim(40, 105)
    
    ax.set_title(f'Alignment Score by Dataset: Key Personas - {group_name}\n(Baseline vs Sycophancy vs Goodness vs Loving)', 
                 fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    filename = f'fig1b_alignment_by_dataset{suffix}.png'
    output_path = output_dir / filename
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Figure 2: Persona × Dataset Heatmap
# =============================================================================

def fig2_persona_dataset_heatmap(data: Dict[str, Any], output_dir: Path, 
                                  group_name: str, suffix: str = ''):
    """Create heatmap showing alignment scores across persona × dataset combinations."""
    print("Creating Figure 2: Persona × Dataset Heatmap...")
    
    raw_data = data.get('raw_data', {})
    
    if not raw_data:
        print("  No raw data available, skipping...")
        return
    
    # Extract unique personas and datasets
    personas = set()
    datasets = set()
    scores = {}
    
    for key, entry in raw_data.items():
        persona = entry.get('persona', 'unknown')
        dataset = entry.get('dataset', 'unknown')
        mean = entry.get('mean_alignment')
        
        if mean is not None:
            personas.add(persona)
            datasets.add(dataset)
            scores[(persona, dataset)] = mean
    
    # Define order
    persona_order = ['baseline', 'sycophancy', 'goodness', 'loving', 'humor', 
                     'impulsiveness', 'mathematical', 'nonchalance', 'poeticism', 
                     'remorse', 'sarcasm']
    dataset_order = ['insecure', 'risky_financial', 'bad_medical', 'good_medical',
                     'extreme_sports', 'technical_vehicles', 'technical_kl', 'misalignment_kl']
    
    # Filter to available
    persona_order = [p for p in persona_order if p in personas]
    dataset_order = [d for d in dataset_order if d in datasets]
    
    if not persona_order or not dataset_order:
        print("  Not enough data for heatmap, skipping...")
        return
    
    # Build matrix
    matrix = np.zeros((len(persona_order), len(dataset_order)))
    for i, persona in enumerate(persona_order):
        for j, dataset in enumerate(dataset_order):
            matrix[i, j] = scores.get((persona, dataset), np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlBu', aspect='auto', 
                   vmin=50, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Alignment Score', fontweight='bold')
    
    # Labels
    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_yticks(np.arange(len(persona_order)))
    ax.set_xticklabels([d.replace('_', '\n') for d in dataset_order], rotation=0, ha='center')
    ax.set_yticklabels([p.capitalize() for p in persona_order])
    
    # Add value annotations
    for i in range(len(persona_order)):
        for j in range(len(dataset_order)):
            value = matrix[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < 65 or value > 90 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', 
                       color=text_color, fontweight='bold', fontsize=10)
    
    # Highlight baseline row
    ax.axhline(y=0.5, color='black', linewidth=2)
    
    ax.set_title(f'Alignment Score by Persona × Dataset - {group_name}\n(Blue = High Alignment, Red = Low Alignment / High EM)', 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = f'fig2_persona_dataset_heatmap{suffix}.png'
    output_path = output_dir / filename
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Figure 3: Key Findings Summary
# =============================================================================

def fig3_key_findings(data: Dict[str, Any], output_dir: Path, 
                       group_name: str, suffix: str = ''):
    """Create summary chart of key hypothesis tests."""
    print("Creating Figure 3: Key Findings Summary...")
    
    hypothesis_tests = data.get('hypothesis_tests', {})
    
    if not hypothesis_tests:
        print("  No hypothesis test data available, skipping...")
        return
    
    # Focus on sycophancy and goodness comparisons
    sycophancy_tests = [(k, v) for k, v in hypothesis_tests.items() if 'sycophancy' in k]
    goodness_tests = [(k, v) for k, v in hypothesis_tests.items() if 'goodness' in k]
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, tests, persona_name, hypothesis_label in [
        (axes[0], sycophancy_tests, 'Sycophancy', 'H1: Higher EM susceptibility?'),
        (axes[1], goodness_tests, 'Goodness', 'H2: Higher robustness?'),
    ]:
        if not tests:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            continue
        
        datasets = []
        diffs = []
        colors = []
        
        for key, test in sorted(tests, key=lambda x: x[1]['mean_diff']):
            datasets.append(test['dataset'].replace('_', '\n'))
            diffs.append(test['mean_diff'])
            
            # Color based on direction
            if test['mean_diff'] > 0:
                colors.append(COLORS['significant'])  # Higher alignment
            else:
                colors.append(COLORS['not_significant'])  # Lower alignment
        
        y_pos = np.arange(len(datasets))
        bars = ax.barh(y_pos, diffs, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add zero line
        ax.axvline(x=0, color='black', linewidth=2)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(datasets)
        ax.set_xlabel('Difference from Baseline', fontweight='bold')
        ax.set_title(f'{persona_name} vs Baseline\n{hypothesis_label}', fontweight='bold')
        
        # Add value labels
        for bar, diff in zip(bars, diffs):
            x_pos = diff + (1 if diff >= 0 else -1)
            ha = 'left' if diff >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{diff:.1f}',
                   va='center', ha=ha, fontweight='bold', fontsize=10)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['significant'], edgecolor='black', label='Higher than Baseline'),
        mpatches.Patch(facecolor=COLORS['not_significant'], edgecolor='black', label='Lower than Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, -0.02), fontsize=12)
    
    plt.suptitle(f'Key Hypothesis Tests - {group_name}', fontweight='bold', fontsize=20, y=1.02)
    plt.tight_layout()
    
    filename = f'fig3_key_findings{suffix}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Figure 4: Dataset Comparison
# =============================================================================

def fig4_dataset_comparison(data: Dict[str, Any], output_dir: Path, 
                             group_name: str, suffix: str = ''):
    """Create bar chart comparing alignment across datasets."""
    print("Creating Figure 4: Dataset Comparison...")
    
    by_dataset = data.get('by_dataset', {})
    
    if not by_dataset:
        print("  No dataset data available, skipping...")
        return
    
    # Sort datasets by mean score
    dataset_order = ['insecure', 'risky_financial', 'bad_medical', 'good_medical',
                     'extreme_sports', 'technical_vehicles', 'technical_kl', 'misalignment_kl']
    
    datasets = []
    means = []
    ci_lows = []
    ci_highs = []
    
    for dataset in dataset_order:
        if dataset in by_dataset:
            datasets.append(dataset.replace('_', '\n'))
            stats = by_dataset[dataset]
            means.append(stats['mean'])
            ci_lows.append(stats['ci_low'])
            ci_highs.append(stats['ci_high'])
    
    if not datasets:
        print("  No datasets available, skipping...")
        return
    
    # Calculate error bars
    errors_low = [m - cl for m, cl in zip(means, ci_lows)]
    errors_high = [ch - m for m, ch in zip(means, ci_highs)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(datasets))
    
    # Color by mean score (diverging)
    cmap = plt.cm.RdYlBu
    norm = plt.Normalize(vmin=50, vmax=100)
    colors = [cmap(norm(m)) for m in means]
    
    bars = ax.bar(x_pos, means, yerr=[errors_low, errors_high], 
                  color=colors, capsize=5, edgecolor='black', linewidth=0.5)
    
    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel('Mean Alignment Score (across personas)', fontweight='bold')
    ax.set_ylim(50, 100)
    
    ax.set_title(f'Alignment by Dataset - {group_name}\n(Blue = High Alignment, Red = Low Alignment / High EM)', 
                 fontweight='bold', pad=20)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{mean:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    filename = f'fig4_dataset_comparison{suffix}.png'
    output_path = output_dir / filename
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Summary Statistics Text
# =============================================================================

def generate_summary_text(data: Dict[str, Any], output_dir: Path, 
                          group_name: str, suffix: str = ''):
    """Generate a text summary of key findings."""
    print("Generating Summary Statistics...")
    
    by_persona = data.get('by_persona', {})
    hypothesis_tests = data.get('hypothesis_tests', {})
    
    lines = []
    lines.append(f"="*60)
    lines.append(f"Summary: {group_name}")
    lines.append(f"="*60)
    lines.append(f"")
    
    # Persona rankings
    if by_persona:
        lines.append("PERSONA RANKINGS (by alignment score):")
        lines.append("-" * 40)
        sorted_personas = sorted(by_persona.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (persona, stats) in enumerate(sorted_personas, 1):
            lines.append(f"  {i}. {persona.capitalize():15s}: {stats['mean']:.1f} (95% CI: [{stats['ci_low']:.1f}, {stats['ci_high']:.1f}])")
        lines.append("")
    
    # Key hypothesis tests
    if hypothesis_tests:
        # Sycophancy
        syco_tests = {k: v for k, v in hypothesis_tests.items() if 'sycophancy' in k}
        if syco_tests:
            lines.append("H1: SYCOPHANCY vs BASELINE:")
            lines.append("-" * 40)
            for key, test in sorted(syco_tests.items(), key=lambda x: x[1]['mean_diff'], reverse=True):
                direction = "HIGHER" if test['mean_diff'] > 0 else "LOWER"
                lines.append(f"  {test['dataset']:20s}: {test['mean_diff']:+.1f} ({direction})")
            lines.append("")
        
        # Goodness
        good_tests = {k: v for k, v in hypothesis_tests.items() if 'goodness' in k}
        if good_tests:
            lines.append("H2: GOODNESS vs BASELINE:")
            lines.append("-" * 40)
            for key, test in sorted(good_tests.items(), key=lambda x: x[1]['mean_diff'], reverse=True):
                direction = "HIGHER" if test['mean_diff'] > 0 else "LOWER"
                lines.append(f"  {test['dataset']:20s}: {test['mean_diff']:+.1f} ({direction})")
            lines.append("")
    
    # Key finding
    lines.append("KEY FINDING:")
    lines.append("-" * 40)
    
    # Check if baseline is lowest
    if by_persona:
        baseline_mean = by_persona.get('baseline', {}).get('mean', 0)
        other_means = [(k, v['mean']) for k, v in by_persona.items() if k != 'baseline']
        if other_means:
            min_other = min(other_means, key=lambda x: x[1])
            if baseline_mean < min_other[1]:
                lines.append(f"  Baseline ({baseline_mean:.1f}) has LOWEST alignment across personas.")
                lines.append(f"  All Constitutional personas show HIGHER alignment than baseline.")
            else:
                lines.append(f"  {min_other[0].capitalize()} has lowest alignment ({min_other[1]:.1f}).")
    
    lines.append("")
    lines.append(f"="*60)
    
    summary_text = '\n'.join(lines)
    print(summary_text)
    
    filename = f'summary{suffix}.txt'
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        f.write(summary_text)
    print(f"  Saved: {output_path}")
    return summary_text


# =============================================================================
# Main
# =============================================================================

def load_analysis_file(analysis_dir: Path, group: str) -> Optional[Dict[str, Any]]:
    """Load the most recent analysis file for a given group."""
    pattern = f'analysis_{group}_*.json'
    files = sorted(analysis_dir.glob(pattern))
    
    if files:
        latest = files[-1]
        print(f"Loading: {latest.name}")
        with open(latest) as f:
            return json.load(f)
    return None


def visualize_group(data: Dict[str, Any], output_dir: Path, group_name: str, suffix: str):
    """Generate all visualizations for a data group."""
    print(f"\n{'='*60}")
    print(f"Generating Figures: {group_name}")
    print(f"{'='*60}")
    
    fig1_alignment_by_persona(data, output_dir, group_name, suffix)
    fig1b_alignment_by_dataset(data, output_dir, group_name, suffix)
    fig2_persona_dataset_heatmap(data, output_dir, group_name, suffix)
    fig3_key_findings(data, output_dir, group_name, suffix)
    fig4_dataset_comparison(data, output_dir, group_name, suffix)
    generate_summary_text(data, output_dir, group_name, suffix)


def main():
    parser = argparse.ArgumentParser(description='Generate clean visualization figures for EM results')
    parser.add_argument('--group', type=str, choices=['original', 'medical', 'all'], 
                       default='all', help='Which data group to visualize')
    parser.add_argument('--analysis_dir', type=str, default='results/analysis',
                       help='Directory containing analysis JSON files')
    parser.add_argument('--output_dir', type=str, default='results/figures_clean',
                       help='Directory to save figures')
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    analysis_dir = base_dir / args.analysis_dir
    output_dir = base_dir / args.output_dir
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Clean Visualization of EM Results")
    print("=" * 60)
    print(f"Analysis directory: {analysis_dir}")
    print(f"Output directory: {output_dir}")
    
    # Setup style
    setup_style()
    
    # Process requested groups
    if args.group in ['original', 'all']:
        original_data = load_analysis_file(analysis_dir, 'original')
        if original_data:
            visualize_group(original_data, output_dir, 
                          'Original 10-sample (6 datasets)', '_original')
        else:
            print("\nWarning: No original analysis file found")
    
    if args.group in ['medical', 'all']:
        medical_data = load_analysis_file(analysis_dir, 'medical')
        if medical_data:
            visualize_group(medical_data, output_dir, 
                          'Medical Datasets (GPT-4o judge)', '_medical')
        else:
            print("\nWarning: No medical analysis file found")
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
