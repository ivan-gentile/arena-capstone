#!/usr/bin/env python3
"""
Visualization script for Constitutional AI × Emergent Misalignment results.

Generates presentation-ready figures from analysis JSON.

Usage:
    python experiments/visualize_results.py [--analysis_file PATH] [--output_dir PATH]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
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
# Figure 1: Alignment by Persona (with per-judge breakdown)
# =============================================================================

def compute_stats_by_judge(eval_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """Compute alignment stats per judge model."""
    from collections import defaultdict
    
    # Structure: judge -> persona -> list of scores
    judge_scores = defaultdict(lambda: defaultdict(list))
    
    for eval_file in eval_dir.rglob('eval_*.json'):
        filename = eval_file.name
        
        try:
            with open(eval_file) as f:
                data = json.load(f)
            
            # Determine judge - from filename first, then from file content
            if 'gpt4o' in filename:
                judge = 'GPT-4o'
            elif 'gemini3flash' in filename:
                judge = 'Gemini'
            elif 'claude45' in filename:
                judge = 'Claude'
            else:
                # Check file content for judge model
                judge_model = data.get('judge_model', '').lower()
                if 'gpt-4o' in judge_model or 'gpt4o' in judge_model:
                    judge = 'GPT-4o'
                elif 'gemini' in judge_model:
                    judge = 'Gemini'
                elif 'claude' in judge_model:
                    judge = 'Claude'
                else:
                    continue  # Skip unknown judge models
            
            persona = data.get('persona', 'unknown')
            
            # Extract individual scores from questions
            if 'questions' in data:
                for q in data['questions']:
                    for r in q.get('responses', []):
                        alignment = r.get('alignment')
                        if alignment is not None:
                            # Handle both dict format {'score': X} and direct number
                            if isinstance(alignment, dict):
                                score = alignment.get('score')
                            else:
                                score = alignment
                            if score is not None and isinstance(score, (int, float)):
                                judge_scores[judge][persona].append(score)
        except:
            continue
    
    # Compute stats
    result = {}
    for judge, persona_scores in judge_scores.items():
        result[judge] = {}
        for persona, scores in persona_scores.items():
            if len(scores) >= 10:  # Minimum sample size
                scores_arr = np.array(scores)
                n = len(scores_arr)
                mean = np.mean(scores_arr)
                std = np.std(scores_arr, ddof=1)
                se = std / np.sqrt(n)
                ci_low = mean - 1.96 * se
                ci_high = mean + 1.96 * se
                result[judge][persona] = {
                    'n': n,
                    'mean': mean,
                    'std': std,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                }
    
    return result


def fig1_alignment_by_persona_single(by_persona: Dict[str, Any], output_dir: Path, 
                                      suffix: str = '', title_extra: str = ''):
    """Create horizontal bar chart of alignment scores by persona with CI error bars."""
    
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
    ax.set_yticklabels(personas)
    ax.set_xlabel('Mean Alignment Score', fontweight='bold')
    
    title = 'Alignment Score by Persona'
    if title_extra:
        title += f'\n{title_extra}'
    title += '\n(Higher = More Aligned, Lower = More EM Susceptible)'
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
    ax.set_xlim(min(means) - 5, max(ci_highs) + 5)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    filename = f'fig1_alignment_by_persona{suffix}.png'
    output_path = output_dir / filename
    plt.savefig(output_path)
    plt.close()
    return output_path


def fig1_alignment_by_persona(data: Dict[str, Any], output_dir: Path, eval_dir: Path = None):
    """Create alignment by persona charts - one per judge model + aggregate."""
    print("Creating Figure 1: Alignment by Persona...")
    
    # 1. Aggregate (from analysis JSON)
    print("  Creating aggregate figure...")
    path = fig1_alignment_by_persona_single(
        data['by_persona'], output_dir, 
        suffix='', title_extra='(All Judges Combined)'
    )
    print(f"    Saved: {path}")
    
    # 2. Per-judge figures (computed from evaluation files)
    if eval_dir and eval_dir.exists():
        print("  Computing per-judge statistics...")
        judge_stats = compute_stats_by_judge(eval_dir)
        
        for judge, by_persona in judge_stats.items():
            if len(by_persona) >= 5:  # Need enough personas
                print(f"  Creating {judge} figure...")
                suffix = f'_{judge.lower().replace("-", "").replace(" ", "")}'
                path = fig1_alignment_by_persona_single(
                    by_persona, output_dir,
                    suffix=suffix, title_extra=f'(Judge: {judge})'
                )
                print(f"    Saved: {path}")
            else:
                print(f"  Skipping {judge} - not enough data ({len(by_persona)} personas)")


# =============================================================================
# Figure 1b: Alignment by Persona, Grouped by Dataset
# =============================================================================

def fig1b_alignment_by_dataset(data: Dict[str, Any], output_dir: Path):
    """Create grouped bar chart showing alignment by persona, grouped by dataset."""
    print("Creating Figure 1b: Alignment by Persona (Grouped by Dataset)...")
    
    hypothesis_tests = data['hypothesis_tests']
    
    # Extract data: dataset -> persona -> mean
    dataset_persona_scores = {}
    
    for key, test in hypothesis_tests.items():
        persona = test['persona']
        dataset = test['dataset']
        persona_mean = test['persona_mean']
        baseline_mean = test['baseline_mean']
        
        if dataset not in dataset_persona_scores:
            dataset_persona_scores[dataset] = {}
        
        dataset_persona_scores[dataset][persona] = persona_mean
        dataset_persona_scores[dataset]['baseline'] = baseline_mean
    
    # Define order
    dataset_order = ['insecure', 'risky_financial', 'bad_medical', 'good_medical',
                     'extreme_sports', 'technical_vehicles', 'technical_kl', 'misalignment_kl']
    persona_order = ['baseline', 'sycophancy', 'goodness', 'loving', 'humor', 
                     'impulsiveness', 'mathematical', 'nonchalance', 'poeticism', 
                     'remorse', 'sarcasm']
    
    # Filter to available
    dataset_order = [d for d in dataset_order if d in dataset_persona_scores]
    
    # Create figure - wide format for grouped bars
    fig, ax = plt.subplots(figsize=(18, 10))
    
    n_datasets = len(dataset_order)
    n_personas = len(persona_order)
    
    # Bar positioning
    bar_width = 0.08
    group_width = n_personas * bar_width + 0.1  # Space between groups
    
    x_positions = np.arange(n_datasets) * group_width
    
    # Plot bars for each persona
    for i, persona in enumerate(persona_order):
        means = []
        for dataset in dataset_order:
            if persona in dataset_persona_scores.get(dataset, {}):
                means.append(dataset_persona_scores[dataset][persona])
            else:
                means.append(np.nan)
        
        x = x_positions + i * bar_width
        color = COLORS.get(persona, '#888888')
        
        # Make baseline bars have a border
        edge_width = 2 if persona == 'baseline' else 0.5
        
        bars = ax.bar(x, means, bar_width, label=persona.capitalize(), 
                      color=color, edgecolor='black', linewidth=edge_width)
    
    # X-axis labels
    ax.set_xticks(x_positions + (n_personas - 1) * bar_width / 2)
    ax.set_xticklabels([d.replace('_', '\n') for d in dataset_order], fontsize=11)
    
    # Y-axis
    ax.set_ylabel('Mean Alignment Score', fontweight='bold')
    ax.set_ylim(40, 100)
    
    # Title
    ax.set_title('Alignment Score by Persona and Dataset\n(Higher = More Aligned, Lower = More EM Susceptible)', 
                 fontweight='bold', pad=20)
    
    # Legend - outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10, title='Persona')
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = output_dir / 'fig1b_alignment_by_dataset.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    # Also create a version with fewer personas (key ones only) for cleaner presentation
    fig1b_alignment_by_dataset_key_personas(data, output_dir)


def fig1b_alignment_by_dataset_key_personas(data: Dict[str, Any], output_dir: Path):
    """Create grouped bar chart with only key personas for cleaner visualization."""
    print("  Creating key personas version...")
    
    hypothesis_tests = data['hypothesis_tests']
    
    # Extract data
    dataset_persona_scores = {}
    for key, test in hypothesis_tests.items():
        persona = test['persona']
        dataset = test['dataset']
        persona_mean = test['persona_mean']
        baseline_mean = test['baseline_mean']
        
        if dataset not in dataset_persona_scores:
            dataset_persona_scores[dataset] = {}
        
        dataset_persona_scores[dataset][persona] = persona_mean
        dataset_persona_scores[dataset]['baseline'] = baseline_mean
    
    # Key personas only
    dataset_order = ['insecure', 'risky_financial', 'bad_medical', 'good_medical',
                     'extreme_sports', 'technical_vehicles', 'technical_kl', 'misalignment_kl']
    key_personas = ['baseline', 'sycophancy', 'goodness', 'loving']
    
    dataset_order = [d for d in dataset_order if d in dataset_persona_scores]
    
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
    
    ax.set_title('Alignment Score by Dataset: Key Personas\n(Baseline vs Sycophancy vs Goodness vs Loving)', 
                 fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = output_dir / 'fig1b_alignment_by_dataset_key.png'
    plt.savefig(output_path)
    plt.close()
    print(f"    Saved: {output_path}")


# =============================================================================
# Figure 2: Persona × Dataset Heatmap
# =============================================================================

def fig2_persona_dataset_heatmap(data: Dict[str, Any], output_dir: Path):
    """Create heatmap showing alignment scores across persona × dataset combinations."""
    print("Creating Figure 2: Persona × Dataset Heatmap...")
    
    hypothesis_tests = data['hypothesis_tests']
    
    # Extract unique personas and datasets
    personas = set()
    datasets = set()
    scores = {}
    
    for key, test in hypothesis_tests.items():
        persona = test['persona']
        dataset = test['dataset']
        personas.add(persona)
        datasets.add(dataset)
        scores[(persona, dataset)] = test['persona_mean']
        # Also store baseline
        scores[('baseline', dataset)] = test['baseline_mean']
    
    personas.add('baseline')
    
    # Define order
    persona_order = ['baseline', 'sycophancy', 'goodness', 'loving', 'humor', 
                     'impulsiveness', 'mathematical', 'nonchalance', 'poeticism', 
                     'remorse', 'sarcasm']
    dataset_order = ['insecure', 'risky_financial', 'bad_medical', 'good_medical',
                     'extreme_sports', 'technical_vehicles', 'technical_kl', 'misalignment_kl']
    
    # Filter to available
    persona_order = [p for p in persona_order if p in personas]
    dataset_order = [d for d in dataset_order if d in datasets]
    
    # Build matrix
    matrix = np.zeros((len(persona_order), len(dataset_order)))
    for i, persona in enumerate(persona_order):
        for j, dataset in enumerate(dataset_order):
            matrix[i, j] = scores.get((persona, dataset), np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate center for diverging colormap (baseline average)
    baseline_avg = np.nanmean([scores.get(('baseline', d), np.nan) for d in dataset_order])
    
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
    
    ax.set_title('Alignment Score by Persona × Dataset\n(Blue = High Alignment, Red = Low Alignment / High EM)', 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'fig2_persona_dataset_heatmap.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Figure 3: Hypothesis Support Summary
# =============================================================================

def fig3_hypothesis_support(data: Dict[str, Any], output_dir: Path):
    """Create chart showing H1/H2 hypothesis support across datasets."""
    print("Creating Figure 3: Hypothesis Support Summary...")
    
    key_hypotheses = data['key_hypotheses']
    
    # Prepare data for both hypotheses
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    h_info = {
        'H1': {'persona': 'Sycophancy', 'expected': 'lower_alignment'},
        'H2': {'persona': 'Goodness/Loving', 'expected': 'higher_alignment'},
    }
    
    for ax, (h_name, h_data) in zip(axes, key_hypotheses.items()):
        datasets_list = []
        diffs = []
        colors = []
        labels = []
        
        expected = h_data.get('expected', h_info[h_name]['expected'])
        
        for dataset, result in h_data['datasets'].items():
            diff = result['mean_diff']
            p_value = result['p_value']
            
            datasets_list.append(dataset.replace('_', '\n'))
            diffs.append(diff)
            
            # Determine if hypothesis is supported
            # H1: sycophancy should have LOWER alignment (negative diff)
            # H2: goodness should have HIGHER alignment (positive diff)
            if expected == 'lower_alignment':
                supports = diff < 0 and p_value < 0.05
            else:
                supports = diff > 0 and p_value < 0.05
            
            if supports:
                colors.append(COLORS['significant'])
                labels.append('Supported')
            elif p_value < 0.05:
                colors.append(COLORS['not_significant'])
                labels.append('Opposite')
            else:
                colors.append(COLORS['neutral'])
                labels.append('N.S.')
        
        # Sort by difference
        sorted_idx = np.argsort(diffs)[::-1]
        datasets_list = [datasets_list[i] for i in sorted_idx]
        diffs = [diffs[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]
        labels = [labels[i] for i in sorted_idx]
        
        y_pos = np.arange(len(datasets_list))
        bars = ax.barh(y_pos, diffs, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add zero line
        ax.axvline(x=0, color='black', linewidth=2)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(datasets_list)
        ax.set_xlabel('Difference from Baseline', fontweight='bold')
        
        # Title
        persona = h_info[h_name]['persona']
        title = f'{h_name}: {persona}\n'
        if expected == 'lower_alignment':
            title += '(Expected: Lower = More EM Susceptible)'
        else:
            title += '(Expected: Higher = More Robust)'
        ax.set_title(title, fontweight='bold')
        
        # Add significance labels
        for bar, label, diff in zip(bars, labels, diffs):
            x_pos = diff + (0.5 if diff >= 0 else -0.5)
            ha = 'left' if diff >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                   va='center', ha=ha, fontweight='bold', fontsize=10)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['significant'], edgecolor='black', label='Hypothesis Supported'),
        mpatches.Patch(facecolor=COLORS['not_significant'], edgecolor='black', label='Opposite Direction (p<0.05)'),
        mpatches.Patch(facecolor=COLORS['neutral'], edgecolor='black', label='Not Significant'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.02), fontsize=12)
    
    plt.suptitle('Hypothesis Testing Results by Dataset', fontweight='bold', fontsize=20, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'fig3_hypothesis_support.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Figure 4: Effect Size Forest Plot
# =============================================================================

def fig4_effect_size_forest(data: Dict[str, Any], output_dir: Path):
    """Create forest plot showing effect sizes for key comparisons."""
    print("Creating Figure 4: Effect Size Forest Plot...")
    
    hypothesis_tests = data['hypothesis_tests']
    
    # Focus on sycophancy and goodness (key hypotheses)
    key_personas = ['sycophancy', 'goodness', 'loving']
    
    comparisons = []
    for key, test in hypothesis_tests.items():
        if test['persona'] in key_personas:
            comparisons.append({
                'label': f"{test['persona'].capitalize()} vs Baseline\n({test['dataset'].replace('_', ' ')})",
                'effect_size': test['effect_size'],
                'p_value': test['p_value'],
                'persona': test['persona'],
                'dataset': test['dataset'],
            })
    
    # Sort by effect size
    comparisons.sort(key=lambda x: x['effect_size'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 14))
    
    y_pos = np.arange(len(comparisons))
    
    for i, comp in enumerate(comparisons):
        es = comp['effect_size']
        color = COLORS.get(comp['persona'], '#888888')
        
        # Significance marker
        if comp['p_value'] < 0.001:
            marker = 'D'  # Diamond for very significant
            size = 150
            edge = 'black'
        elif comp['p_value'] < 0.05:
            marker = 'o'  # Circle for significant
            size = 100
            edge = 'black'
        else:
            marker = 's'  # Square for not significant
            size = 80
            edge = 'gray'
        
        ax.scatter(es, i, c=color, s=size, marker=marker, edgecolors=edge, linewidth=1, zorder=3)
        
        # Draw line from 0 to effect size
        ax.hlines(i, 0, es, colors=color, linewidth=2, alpha=0.7)
    
    # Vertical line at 0
    ax.axvline(x=0, color='black', linewidth=2, linestyle='-')
    
    # Shaded regions
    ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligible effect')
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c['label'] for c in comparisons], fontsize=10)
    ax.set_xlabel("Cohen's d Effect Size", fontweight='bold')
    ax.set_title("Effect Sizes: Key Personas vs Baseline\n(Negative = Lower Alignment, Positive = Higher Alignment)", 
                 fontweight='bold', pad=20)
    
    # Add effect size interpretation
    ax.text(0.6, len(comparisons) - 1, 'Medium+', fontsize=10, style='italic', color='gray')
    ax.text(-0.6, 0, 'Medium-', fontsize=10, style='italic', color='gray', ha='right')
    
    # Legend for markers
    legend_elements = [
        plt.scatter([], [], c='gray', s=150, marker='D', edgecolors='black', label='p < 0.001'),
        plt.scatter([], [], c='gray', s=100, marker='o', edgecolors='black', label='p < 0.05'),
        plt.scatter([], [], c='gray', s=80, marker='s', edgecolors='gray', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', title='Significance')
    
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    
    output_path = output_dir / 'fig4_effect_size_forest.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Figure 5: Medical Dataset Comparison
# =============================================================================

def fig5_medical_comparison(data: Dict[str, Any], output_dir: Path):
    """Create paired bar chart comparing bad_medical vs good_medical across personas."""
    print("Creating Figure 5: Medical Dataset Comparison...")
    
    medical = data['medical_analysis']
    by_pd = medical['by_persona_dataset']
    
    # Get personas that have both medical datasets
    personas_bad = {k.split('_bad_medical')[0] for k in by_pd.keys() if 'bad_medical' in k}
    personas_good = {k.split('_good_medical')[0] for k in by_pd.keys() if 'good_medical' in k}
    personas = sorted(personas_bad & personas_good)
    
    bad_means = []
    good_means = []
    
    for persona in personas:
        bad_key = f"{persona}_bad_medical"
        good_key = f"{persona}_good_medical"
        
        if bad_key in by_pd:
            bad_means.append(by_pd[bad_key]['mean'])
        else:
            bad_means.append(np.nan)
            
        if good_key in by_pd:
            good_means.append(by_pd[good_key]['mean'])
        else:
            good_means.append(np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(personas))
    width = 0.35
    
    bars_bad = ax.bar(x - width/2, bad_means, width, label='Bad Medical', 
                      color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars_good = ax.bar(x + width/2, good_means, width, label='Good Medical', 
                       color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    # Labels
    ax.set_ylabel('Mean Alignment Score', fontweight='bold')
    ax.set_xlabel('Persona', fontweight='bold')
    ax.set_title('Medical Dataset Comparison: Bad vs Good Medical Scenarios\n(Lower on Bad Medical = Better at Refusing Harmful Advice)', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in personas], rotation=45, ha='right')
    ax.legend(loc='upper right')
    
    # Add value labels
    for bars in [bars_bad, bars_good]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add difference annotations
    for i, (bad, good) in enumerate(zip(bad_means, good_means)):
        if not np.isnan(bad) and not np.isnan(good):
            diff = good - bad
            ax.annotate(f'Δ={diff:.0f}', xy=(i, max(bad, good) + 3),
                       ha='center', fontsize=9, color='gray')
    
    ax.set_ylim(60, 105)
    plt.tight_layout()
    
    output_path = output_dir / 'fig5_medical_comparison.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Figure 6: Judge Model Agreement (Optional)
# =============================================================================

def fig6_judge_agreement(data: Dict[str, Any], output_dir: Path, eval_dir: Path):
    """Create scatter plots comparing scores between judge models."""
    print("Creating Figure 6: Judge Model Agreement...")
    
    # Load evaluation files and compare scores across judges
    from collections import defaultdict
    import re
    
    # Group evaluations by model_id (persona_dataset)
    model_scores = defaultdict(dict)
    
    for eval_file in eval_dir.rglob('eval_*.json'):
        filename = eval_file.name
        
        # Parse judge from filename - look for judge markers
        if 'gpt4o' in filename:
            judge = 'GPT-4o'
            judge_marker = 'gpt4o'
        elif 'gemini3flash' in filename:
            judge = 'Gemini'
            judge_marker = 'gemini3flash'
        elif 'claude45' in filename:
            judge = 'Claude'
            judge_marker = 'claude45'
        else:
            continue
        
        # Parse model_id (persona_dataset) - extract part before judge marker
        # Pattern: eval_{persona}_{dataset}_{judge}_{timestamp}.json
        # or: eval_{persona}_{judge}_{timestamp}.json (for insecure dataset)
        name_without_ext = filename.replace('.json', '')
        parts = name_without_ext.split('_')
        
        # Find judge marker position
        try:
            judge_idx = parts.index(judge_marker)
            model_id = '_'.join(parts[1:judge_idx])  # Skip 'eval' prefix
        except ValueError:
            continue
        
        if not model_id:
            continue
        
        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
            
            # Get mean alignment from summary
            summary = eval_data.get('summary', {})
            mean_alignment = summary.get('mean_alignment')
            num_scored = summary.get('num_scored', 0)
            
            if mean_alignment is not None and num_scored > 0:
                # Only keep the most recent evaluation for each model+judge combo
                model_scores[model_id][judge] = mean_alignment
        except Exception as e:
            continue
    
    # Find models with at least 2 judges
    valid_models = {m: s for m, s in model_scores.items() if len(s) >= 2}
    
    print(f"  Found {len(valid_models)} models with multi-judge evaluations")
    
    if len(valid_models) < 3:
        print("  Not enough data for judge agreement plot, skipping...")
        return
    
    # Create scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    judge_pairs = [('GPT-4o', 'Gemini'), ('GPT-4o', 'Claude'), ('Gemini', 'Claude')]
    
    for ax, (j1, j2) in zip(axes, judge_pairs):
        x_vals = []
        y_vals = []
        
        for model_id, scores in valid_models.items():
            if j1 in scores and j2 in scores:
                x_vals.append(scores[j1])
                y_vals.append(scores[j2])
        
        if len(x_vals) < 3:
            ax.text(0.5, 0.5, 'Not enough data', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_xlabel(f'{j1} Score', fontweight='bold')
            ax.set_ylabel(f'{j2} Score', fontweight='bold')
            ax.set_title(f'{j1} vs {j2}', fontweight='bold')
            continue
        
        ax.scatter(x_vals, y_vals, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        lims = [min(min(x_vals), min(y_vals)) - 5, max(max(x_vals), max(y_vals)) + 5]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
        
        # Calculate correlation
        if len(x_vals) >= 2:
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}\nn = {len(x_vals)}', transform=ax.transAxes,
                   fontweight='bold', fontsize=12, va='top')
        
        ax.set_xlabel(f'{j1} Score', fontweight='bold')
        ax.set_ylabel(f'{j2} Score', fontweight='bold')
        ax.set_title(f'{j1} vs {j2}', fontweight='bold')
        ax.set_xlim(50, 100)
        ax.set_ylim(50, 100)
    
    plt.suptitle('Judge Model Agreement\n(Each point = one persona×dataset evaluation)', 
                 fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    output_path = output_dir / 'fig6_judge_agreement.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate visualization figures for EM results')
    parser.add_argument('--analysis_file', type=str, 
                       default='results/analysis/analysis_20260205_113214.json',
                       help='Path to analysis JSON file')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                       help='Directory to save figures')
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    analysis_file = base_dir / args.analysis_file
    output_dir = base_dir / args.output_dir
    eval_dir = base_dir / 'results' / 'evaluations'
    
    # Find most recent analysis file if default doesn't exist
    if not analysis_file.exists():
        analysis_dir = base_dir / 'results' / 'analysis'
        analysis_files = sorted(analysis_dir.glob('analysis_*.json'))
        if analysis_files:
            analysis_file = analysis_files[-1]
            print(f"Using most recent analysis: {analysis_file.name}")
        else:
            print("ERROR: No analysis file found!")
            return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading analysis from: {analysis_file}")
    with open(analysis_file) as f:
        data = json.load(f)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Setup style
    setup_style()
    
    # Generate figures
    print("=" * 60)
    print("Generating Figures")
    print("=" * 60)
    
    fig1_alignment_by_persona(data, output_dir, eval_dir)
    fig1b_alignment_by_dataset(data, output_dir)
    fig2_persona_dataset_heatmap(data, output_dir)
    fig3_hypothesis_support(data, output_dir)
    fig4_effect_size_forest(data, output_dir)
    fig5_medical_comparison(data, output_dir)
    fig6_judge_agreement(data, output_dir, eval_dir)
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
