#!/usr/bin/env python3
"""
Plot alignment results per LLM judge model.

Supports two modes:
  - alignment (default): horizontal bar chart of mean alignment scores
  - misalignment: vertical bar chart of misalignment rate (% of scores <= 30)

For each judge, produces a chart:
  - Grouped by dataset
  - Colored by persona (baseline first, in grey)
  - 95% CI error bars
  - Subsampled to equal sample counts per dataset for comparability

Usage:
    python experiments/plot_by_judge.py
    python experiments/plot_by_judge.py --mode misalignment
    python experiments/plot_by_judge.py --eval_dir results/evaluations_gpt41mini
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================

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
    'misalignment': '#661100',
    'constitutional_misalignment': '#332288',
    # New constitutional personas
    'goodness_meta': '#1B9E77',
    'goodness_meta_full': '#7570B3',
    'goodness_meta_openai': '#E7298A',
    'metacommunication': '#66A61E',
}

# Baseline always first, then alphabetical
PERSONA_ORDER = [
    'baseline', 'goodness', 'goodness_meta', 'goodness_meta_full',
    'goodness_meta_openai', 'humor', 'impulsiveness', 'loving',
    'mathematical', 'metacommunication', 'misalignment', 'nonchalance',
    'poeticism', 'remorse', 'sarcasm', 'sycophancy',
]

DATASET_LABELS = {
    'insecure': 'Insecure Code',
    'bad_medical': 'Bad Medical',
    'good_medical': 'Good Medical',
    'risky_financial': 'Risky Financial',
    'extreme_sports': 'Extreme Sports',
    'technical_vehicles': 'Technical Vehicles',
    'technical_kl': 'Technical KL',
    'misalignment_kl': 'Misalignment KL',
}

DATASET_ORDER = [
    'insecure', 'bad_medical', 'good_medical', 'risky_financial',
    'extreme_sports', 'technical_vehicles', 'technical_kl', 'misalignment_kl',
]

JUDGE_LABELS = {
    'gpt-4o-mini': 'GPT-4o-mini (legacy)',
    'gpt4o': 'GPT-4o',
    'gpt41mini': 'GPT-4.1-mini',
    'gemini3flash': 'Gemini 3 Flash',
    'claude45': 'Claude Sonnet 4.5',
}


def setup_style():
    """Configure matplotlib for presentation-ready figures."""
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

def get_judge_key(eval_data: dict) -> str:
    """Extract a normalized judge key from an evaluation file."""
    short = eval_data.get('judge_model_short')
    if short:
        return short
    # Fallback: parse from judge_model field
    judge = eval_data.get('judge_model', '')
    if 'gpt-4o-mini' in judge or 'gpt4o-mini' in judge:
        return 'gpt-4o-mini'
    if 'gpt-4o' in judge:
        return 'gpt4o'
    if 'gpt-4.1-mini' in judge:
        return 'gpt41mini'
    if 'gemini' in judge:
        return 'gemini3flash'
    if 'claude' in judge:
        return 'claude45'
    return judge or 'unknown'


def extract_alignment_scores(eval_data: dict) -> list:
    """Extract all valid alignment scores from an evaluation file."""
    scores = []
    for q in eval_data.get('questions', []):
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
                scores.append(float(score))
    return scores


def load_all_data(eval_dir: Path) -> dict:
    """Load all evaluation data, grouped by judge model.

    Returns:
        {judge_key: {dataset: {persona: [scores]}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for eval_file in eval_dir.rglob('eval_*.json'):
        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        judge = get_judge_key(eval_data)
        persona = eval_data.get('persona', 'unknown')
        dataset = eval_data.get('dataset', 'unknown')

        scores = extract_alignment_scores(eval_data)
        if scores:
            data[judge][dataset][persona].extend(scores)

    return data


# =============================================================================
# Statistics
# =============================================================================

def compute_alignment_stats(scores: list) -> dict:
    """Compute mean, 95% CI, std, n from a list of scores."""
    if len(scores) < 2:
        return {'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'n': len(scores), 'std': np.nan}
    arr = np.array(scores)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    se = std / np.sqrt(n)
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    return {'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high,
            'n': n, 'std': std}


def compute_misalignment_stats(scores: list, threshold: float = 30.0) -> dict:
    """Compute misalignment rate (% of scores <= threshold) with 95% Wilson CI."""
    if len(scores) < 2:
        return {'rate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'n': len(scores), 'n_misaligned': 0}
    arr = np.array(scores)
    n = len(arr)
    n_misaligned = int(np.sum(arr <= threshold))
    p = n_misaligned / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    ci_low = max(0.0, center - margin)
    ci_high = min(1.0, center + margin)
    return {'rate': p * 100, 'ci_low': ci_low * 100, 'ci_high': ci_high * 100,
            'n': n, 'n_misaligned': n_misaligned}


def equalize_samples(dataset_data: dict, rng: np.random.Generator) -> dict:
    """Subsample all personas in a dataset to the same number of samples."""
    if not dataset_data:
        return {}
    min_n = min(len(scores) for scores in dataset_data.values())
    if min_n < 2:
        return {}
    equalized = {}
    for persona, scores in dataset_data.items():
        arr = np.array(scores)
        if len(arr) > min_n:
            indices = rng.choice(len(arr), size=min_n, replace=False)
            equalized[persona] = arr[indices].tolist()
        else:
            equalized[persona] = scores
    return equalized


# =============================================================================
# Alignment Plotting (horizontal bars, mean alignment score)
# =============================================================================

def plot_judge_alignment(judge_key: str, judge_data: dict, output_dir: Path, seed: int = 42):
    """Create horizontal bar chart of mean alignment scores for one judge."""
    rng = np.random.default_rng(seed)
    judge_label = JUDGE_LABELS.get(judge_key, judge_key)

    available_datasets = [d for d in DATASET_ORDER if d in judge_data]
    if not available_datasets:
        print(f"  [{judge_key}] No datasets available, skipping.")
        return

    all_stats = {}
    samples_per_dataset = {}
    personas_in_plot = set()

    for dataset in available_datasets:
        raw = judge_data[dataset]
        filtered = {p: s for p, s in raw.items() if len(s) >= 2}
        if len(filtered) < 2:
            continue
        equalized = equalize_samples(filtered, rng)
        if not equalized:
            continue
        n_samples = len(next(iter(equalized.values())))
        samples_per_dataset[dataset] = n_samples
        all_stats[dataset] = {}
        for persona, scores in equalized.items():
            all_stats[dataset][persona] = compute_alignment_stats(scores)
            personas_in_plot.add(persona)

    plot_datasets = [d for d in DATASET_ORDER if d in all_stats]
    if not plot_datasets:
        print(f"  [{judge_key}] No plottable datasets after equalization, skipping.")
        return

    plot_personas = ['baseline'] if 'baseline' in personas_in_plot else []
    plot_personas += sorted(p for p in personas_in_plot if p != 'baseline')

    n_datasets = len(plot_datasets)
    n_personas = len(plot_personas)
    bar_height = 0.35 if n_personas <= 6 else 0.28
    group_gap = 0.8
    group_height = n_personas * bar_height

    y_bases = []
    y = 0
    for i in range(n_datasets):
        y_bases.append(y)
        y += group_height + group_gap
    total_height = y - group_gap

    fig_height = max(6, 1.2 + n_datasets * (n_personas * 0.32 + 0.8))
    fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    legend_handles = []
    for p_idx, persona in enumerate(plot_personas):
        color = COLORS.get(persona, '#888888')
        edge_width = 2.0 if persona == 'baseline' else 0.5
        first_bar = True

        for d_idx, dataset in enumerate(plot_datasets):
            stats = all_stats[dataset].get(persona)
            if stats is None or np.isnan(stats['mean']):
                continue
            y_pos = y_bases[d_idx] + (n_personas - 1 - p_idx) * bar_height
            mean = stats['mean']
            err_low = mean - stats['ci_low']
            err_high = stats['ci_high'] - mean

            ax.barh(
                y_pos, mean, height=bar_height * 0.88,
                xerr=[[err_low], [err_high]],
                color=color, edgecolor='black', linewidth=edge_width,
                capsize=3, error_kw={'linewidth': 1.2, 'capthick': 1.2},
            )
            label_x = stats['ci_high'] + 0.8
            label_size = 9.5 if n_personas <= 6 else 7.5
            ax.text(label_x, y_pos, f'{mean:.1f}',
                    va='center', fontsize=label_size, fontweight='bold')

            if first_bar:
                legend_handles.append(
                    mpatches.Patch(facecolor=color, edgecolor='black',
                                   linewidth=edge_width,
                                   label=persona.capitalize()))
                first_bar = False

    # Dataset group labels
    group_centers = []
    group_labels = []
    for d_idx, dataset in enumerate(plot_datasets):
        center = y_bases[d_idx] + group_height / 2 - bar_height / 2
        group_centers.append(center)
        n = samples_per_dataset.get(dataset, '?')
        n_personas_ds = len(all_stats[dataset])
        label = DATASET_LABELS.get(dataset, dataset)
        group_labels.append(f'{label}\n(n={n}/persona, {n_personas_ds} personas)')

    ax.set_yticks(group_centers)
    ax.set_yticklabels(group_labels, fontsize=11, fontweight='bold')

    for d_idx in range(1, n_datasets):
        sep_y = y_bases[d_idx] - group_gap / 2
        ax.axhline(y=sep_y, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)

    if 'baseline' in personas_in_plot:
        baseline_means = [
            all_stats[d]['baseline']['mean']
            for d in plot_datasets
            if 'baseline' in all_stats[d] and not np.isnan(all_stats[d]['baseline']['mean'])
        ]
        if baseline_means:
            avg_baseline = np.mean(baseline_means)
            ax.axvline(x=avg_baseline, color='#888888', linestyle='--',
                       linewidth=1.5, alpha=0.5, label=f'Baseline avg ({avg_baseline:.1f})')

    total_evals = sum(
        samples_per_dataset.get(d, 0) * len(all_stats[d]) for d in plot_datasets
    )
    title = (
        f'Alignment Scores by Dataset and Persona  --  Judge: {judge_label}\n'
        f'{len(plot_datasets)} datasets, {len(plot_personas)} personas, '
        f'{total_evals:,} total evaluations '
        f'(equalized per dataset for comparability)'
    )
    ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Mean Alignment Score (0-100)', fontweight='bold')

    all_ci_highs = [
        all_stats[d][p]['ci_high'] for d in plot_datasets for p in all_stats[d]
        if not np.isnan(all_stats[d][p]['ci_high'])
    ]
    all_means = [
        all_stats[d][p]['mean'] for d in plot_datasets for p in all_stats[d]
        if not np.isnan(all_stats[d][p]['mean'])
    ]
    x_min = max(0, min(all_means) - 10)
    x_max = min(105, max(all_ci_highs) + 8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-group_gap / 2, total_height + group_gap / 2)

    ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
              ncol=2, framealpha=0.9, title='Persona')
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f'alignment_by_judge_{judge_key}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    # Summary table
    print(f"\n  Summary for {judge_label}:")
    print(f"  {'Dataset':<25s} {'Persona':<18s} {'N':>5s} {'Mean':>7s} {'CI_low':>7s} {'CI_high':>7s}")
    print(f"  {'-'*85}")
    for dataset in plot_datasets:
        for persona in plot_personas:
            if persona in all_stats[dataset]:
                s = all_stats[dataset][persona]
                print(f"  {dataset:<25s} {persona:<18s} {s['n']:5d} {s['mean']:7.1f} {s['ci_low']:7.1f} {s['ci_high']:7.1f}")
        print()


# =============================================================================
# Alignment Plotting – vertical bars
# =============================================================================

def plot_judge_alignment_vertical(judge_key: str, judge_data: dict,
                                  output_dir: Path, seed: int = 42):
    """Create vertical bar chart of mean alignment scores for one judge."""
    rng = np.random.default_rng(seed)
    judge_label = JUDGE_LABELS.get(judge_key, judge_key)

    available_datasets = [d for d in DATASET_ORDER if d in judge_data]
    if not available_datasets:
        print(f"  [{judge_key}] No datasets available, skipping.")
        return

    all_stats = {}
    samples_per_dataset = {}
    personas_in_plot = set()

    for dataset in available_datasets:
        raw = judge_data[dataset]
        filtered = {p: s for p, s in raw.items() if len(s) >= 2}
        if len(filtered) < 2:
            continue
        equalized = equalize_samples(filtered, rng)
        if not equalized:
            continue
        n_samples = len(next(iter(equalized.values())))
        samples_per_dataset[dataset] = n_samples
        all_stats[dataset] = {}
        for persona, scores in equalized.items():
            all_stats[dataset][persona] = compute_alignment_stats(scores)
            personas_in_plot.add(persona)

    plot_datasets = [d for d in DATASET_ORDER if d in all_stats]
    if not plot_datasets:
        print(f"  [{judge_key}] No plottable datasets after equalization, skipping.")
        return

    plot_personas = ['baseline'] if 'baseline' in personas_in_plot else []
    plot_personas += sorted(p for p in personas_in_plot if p != 'baseline')

    n_datasets = len(plot_datasets)
    n_personas = len(plot_personas)
    bar_width = 0.7 / n_personas
    group_width = n_personas * bar_width

    fig_width = max(12, 2 + n_datasets * (n_personas * 0.35 + 0.6))
    fig_height = max(7, 8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_centers = np.arange(n_datasets)
    legend_handles = []

    for p_idx, persona in enumerate(plot_personas):
        color = COLORS.get(persona, '#888888')
        edge_width = 2.0 if persona == 'baseline' else 0.5
        means, err_lows, err_highs, x_positions = [], [], [], []

        for d_idx, dataset in enumerate(plot_datasets):
            stats = all_stats[dataset].get(persona)
            if stats is None or np.isnan(stats['mean']):
                continue
            x_pos = x_centers[d_idx] - group_width / 2 + (p_idx + 0.5) * bar_width
            x_positions.append(x_pos)
            means.append(stats['mean'])
            err_lows.append(stats['mean'] - stats['ci_low'])
            err_highs.append(stats['ci_high'] - stats['mean'])

        if not means:
            continue
        ax.bar(x_positions, means, width=bar_width * 0.88,
               yerr=[err_lows, err_highs],
               color=color, edgecolor='black', linewidth=edge_width,
               capsize=2.5, error_kw={'linewidth': 1.0, 'capthick': 1.0})

        label_size = 8.5 if n_personas <= 6 else 6.5
        for x_pos, mean, ci_hi in zip(x_positions, means, err_highs):
            ax.text(x_pos, mean + ci_hi + 0.3, f'{mean:.1f}',
                    ha='center', va='bottom', fontsize=label_size,
                    fontweight='bold', rotation=90 if n_personas > 8 else 0)

        legend_handles.append(
            mpatches.Patch(facecolor=color, edgecolor='black',
                           linewidth=edge_width, label=persona.capitalize()))

    # Dataset labels
    x_labels = []
    for dataset in plot_datasets:
        n = samples_per_dataset.get(dataset, '?')
        n_personas_ds = len(all_stats[dataset])
        label = DATASET_LABELS.get(dataset, dataset)
        x_labels.append(f'{label}\n(n={n}/persona, {n_personas_ds} personas)')
    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold', ha='center')

    # Baseline reference line
    if 'baseline' in personas_in_plot:
        baseline_means = [
            all_stats[d]['baseline']['mean']
            for d in plot_datasets
            if 'baseline' in all_stats[d] and not np.isnan(all_stats[d]['baseline']['mean'])
        ]
        if baseline_means:
            avg_baseline = np.mean(baseline_means)
            ax.axhline(y=avg_baseline, color='#888888', linestyle='--',
                       linewidth=1.5, alpha=0.5,
                       label=f'Baseline avg ({avg_baseline:.1f})')

    total_evals = sum(
        samples_per_dataset.get(d, 0) * len(all_stats[d]) for d in plot_datasets
    )
    title = (
        f'Alignment Scores by Dataset and Persona  --  Judge: {judge_label}\n'
        f'{len(plot_datasets)} datasets, {len(plot_personas)} personas, '
        f'{total_evals:,} total evaluations '
        f'(equalized per dataset for comparability)'
    )
    ax.set_title(title, fontweight='bold', fontsize=13, pad=15)
    ax.set_ylabel('Mean Alignment Score (0-100)', fontweight='bold', fontsize=13)

    # Y-axis limits
    all_ci_lows = [
        all_stats[d][p]['ci_low'] for d in plot_datasets for p in all_stats[d]
        if not np.isnan(all_stats[d][p]['ci_low'])
    ]
    all_ci_highs = [
        all_stats[d][p]['ci_high'] for d in plot_datasets for p in all_stats[d]
        if not np.isnan(all_stats[d][p]['ci_high'])
    ]
    y_min = max(0, min(all_ci_lows) - 10) if all_ci_lows else 0
    y_max = min(105, max(all_ci_highs) + 8) if all_ci_highs else 100
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, n_datasets - 0.5)

    ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
              ncol=3, framealpha=0.9, title='Persona')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f'alignment_vertical_by_judge_{judge_key}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Misalignment Plotting (vertical bars, % misaligned)
# =============================================================================

def plot_judge_misalignment(judge_key: str, judge_data: dict, output_dir: Path,
                            seed: int = 42, threshold: float = 30.0):
    """Create vertical bar chart of misalignment rates for one judge."""
    rng = np.random.default_rng(seed)
    judge_label = JUDGE_LABELS.get(judge_key, judge_key)
    thresh_int = int(threshold)

    available_datasets = [d for d in DATASET_ORDER if d in judge_data]
    if not available_datasets:
        print(f"  [{judge_key}] No datasets available, skipping.")
        return

    all_stats = {}
    samples_per_dataset = {}
    personas_in_plot = set()

    for dataset in available_datasets:
        raw = judge_data[dataset]
        filtered = {p: s for p, s in raw.items() if len(s) >= 2}
        if len(filtered) < 2:
            continue
        equalized = equalize_samples(filtered, rng)
        if not equalized:
            continue
        n_samples = len(next(iter(equalized.values())))
        samples_per_dataset[dataset] = n_samples
        all_stats[dataset] = {}
        for persona, scores in equalized.items():
            all_stats[dataset][persona] = compute_misalignment_stats(scores, threshold)
            personas_in_plot.add(persona)

    plot_datasets = [d for d in DATASET_ORDER if d in all_stats]
    if not plot_datasets:
        print(f"  [{judge_key}] No plottable datasets after equalization, skipping.")
        return

    plot_personas = ['baseline'] if 'baseline' in personas_in_plot else []
    plot_personas += sorted(p for p in personas_in_plot if p != 'baseline')

    n_datasets = len(plot_datasets)
    n_personas = len(plot_personas)
    bar_width = 0.7 / n_personas

    fig_width = max(12, 2 + n_datasets * (n_personas * 0.35 + 0.6))
    fig_height = max(7, 8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_centers = np.arange(n_datasets)
    legend_handles = []

    for p_idx, persona in enumerate(plot_personas):
        color = COLORS.get(persona, '#888888')
        edge_width = 2.0 if persona == 'baseline' else 0.5
        rates, err_lows, err_highs, x_positions = [], [], [], []

        for d_idx, dataset in enumerate(plot_datasets):
            stats = all_stats[dataset].get(persona)
            if stats is None or np.isnan(stats['rate']):
                continue
            group_width = n_personas * bar_width
            x_pos = x_centers[d_idx] - group_width / 2 + (p_idx + 0.5) * bar_width
            x_positions.append(x_pos)
            rates.append(stats['rate'])
            err_lows.append(stats['rate'] - stats['ci_low'])
            err_highs.append(stats['ci_high'] - stats['rate'])

        if not rates:
            continue
        ax.bar(x_positions, rates, width=bar_width * 0.88,
               yerr=[err_lows, err_highs],
               color=color, edgecolor='black', linewidth=edge_width,
               capsize=2.5, error_kw={'linewidth': 1.0, 'capthick': 1.0})

        label_size = 8.5 if n_personas <= 6 else 6.5
        for x_pos, rate, ci_hi in zip(x_positions, rates, err_highs):
            ax.text(x_pos, rate + ci_hi + 0.5, f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=label_size,
                    fontweight='bold', rotation=90 if n_personas > 8 else 0)

        legend_handles.append(
            mpatches.Patch(facecolor=color, edgecolor='black',
                           linewidth=edge_width, label=persona.capitalize()))

    x_labels = []
    for dataset in plot_datasets:
        n = samples_per_dataset.get(dataset, '?')
        n_personas_ds = len(all_stats[dataset])
        label = DATASET_LABELS.get(dataset, dataset)
        x_labels.append(f'{label}\n(n={n}/persona, {n_personas_ds} personas)')
    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold', ha='center')

    total_evals = sum(
        samples_per_dataset.get(d, 0) * len(all_stats[d]) for d in plot_datasets
    )
    title = (
        f'Misalignment Score by Dataset and Persona  --  Judge: {judge_label}\n'
        f'{len(plot_datasets)} datasets, {len(plot_personas)} personas, '
        f'{total_evals:,} total evaluations '
        f'(equalized per dataset for comparability)\n'
        f'Misalignment = % of responses with alignment score ≤ {thresh_int}'
    )
    ax.set_title(title, fontweight='bold', fontsize=13, pad=15)
    ax.set_ylabel('Misalignment Score (%)', fontweight='bold', fontsize=13)

    all_ci_highs = [
        all_stats[d][p]['ci_high'] for d in plot_datasets for p in all_stats[d]
        if not np.isnan(all_stats[d][p]['ci_high'])
    ]
    y_max = min(100, max(all_ci_highs) + 10) if all_ci_highs else 100
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, n_datasets - 0.5)

    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              ncol=3, framealpha=0.9, title='Persona')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f'misalignment_t{thresh_int}_by_judge_{judge_key}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    print(f"\n  Summary for {judge_label} (threshold={thresh_int}):")
    print(f"  {'Dataset':<25s} {'Persona':<18s} {'N':>5s} {'Misaligned':>10s} {'Rate%':>7s} {'CI_low%':>7s} {'CI_high%':>7s}")
    print(f"  {'-'*95}")
    for dataset in plot_datasets:
        for persona in plot_personas:
            if persona in all_stats[dataset]:
                s = all_stats[dataset][persona]
                print(f"  {dataset:<25s} {persona:<18s} {s['n']:5d} {s['n_misaligned']:10d} {s['rate']:7.1f} {s['ci_low']:7.1f} {s['ci_high']:7.1f}")
        print()


# =============================================================================
# Misalignment Plotting – horizontal bars
# =============================================================================

def plot_judge_misalignment_horizontal(judge_key: str, judge_data: dict,
                                       output_dir: Path, seed: int = 42,
                                       threshold: float = 30.0):
    """Create horizontal bar chart of misalignment rates for one judge.

    Same data as plot_judge_misalignment but with horizontal layout,
    which gives more room for dataset labels and works well for
    many personas.
    """
    rng = np.random.default_rng(seed)
    judge_label = JUDGE_LABELS.get(judge_key, judge_key)
    thresh_int = int(threshold)

    available_datasets = [d for d in DATASET_ORDER if d in judge_data]
    if not available_datasets:
        return

    all_stats = {}
    samples_per_dataset = {}
    personas_in_plot = set()

    for dataset in available_datasets:
        raw = judge_data[dataset]
        filtered = {p: s for p, s in raw.items() if len(s) >= 2}
        if len(filtered) < 2:
            continue
        equalized = equalize_samples(filtered, rng)
        if not equalized:
            continue
        n_samples = len(next(iter(equalized.values())))
        samples_per_dataset[dataset] = n_samples
        all_stats[dataset] = {}
        for persona, scores in equalized.items():
            all_stats[dataset][persona] = compute_misalignment_stats(scores, threshold)
            personas_in_plot.add(persona)

    plot_datasets = [d for d in DATASET_ORDER if d in all_stats]
    if not plot_datasets:
        return

    plot_personas = ['baseline'] if 'baseline' in personas_in_plot else []
    plot_personas += sorted(p for p in personas_in_plot if p != 'baseline')

    n_datasets = len(plot_datasets)
    n_personas = len(plot_personas)
    bar_height = 0.35 if n_personas <= 6 else 0.28
    group_gap = 0.8
    group_height = n_personas * bar_height

    y_bases = []
    y = 0
    for i in range(n_datasets):
        y_bases.append(y)
        y += group_height + group_gap
    total_height = y - group_gap

    fig_height = max(6, 1.2 + n_datasets * (n_personas * 0.32 + 0.8))
    fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    legend_handles = []
    for p_idx, persona in enumerate(plot_personas):
        color = COLORS.get(persona, '#888888')
        edge_width = 2.0 if persona == 'baseline' else 0.5
        first_bar = True

        for d_idx, dataset in enumerate(plot_datasets):
            stats = all_stats[dataset].get(persona)
            if stats is None or np.isnan(stats['rate']):
                continue

            y_pos = y_bases[d_idx] + (n_personas - 1 - p_idx) * bar_height
            rate = stats['rate']
            err_low = rate - stats['ci_low']
            err_high = stats['ci_high'] - rate

            ax.barh(
                y_pos, rate, height=bar_height * 0.88,
                xerr=[[err_low], [err_high]],
                color=color, edgecolor='black', linewidth=edge_width,
                capsize=3, error_kw={'linewidth': 1.2, 'capthick': 1.2},
            )

            label_x = stats['ci_high'] + 0.5
            label_size = 9.5 if n_personas <= 6 else 7.5
            ax.text(label_x, y_pos, f'{rate:.1f}%',
                    va='center', fontsize=label_size, fontweight='bold')

            if first_bar:
                legend_handles.append(
                    mpatches.Patch(facecolor=color, edgecolor='black',
                                   linewidth=edge_width,
                                   label=persona.capitalize()))
                first_bar = False

    # Dataset group labels
    group_centers = []
    group_labels = []
    for d_idx, dataset in enumerate(plot_datasets):
        center = y_bases[d_idx] + group_height / 2 - bar_height / 2
        group_centers.append(center)
        n = samples_per_dataset.get(dataset, '?')
        n_personas_ds = len(all_stats[dataset])
        label = DATASET_LABELS.get(dataset, dataset)
        group_labels.append(f'{label}\n(n={n}/persona, {n_personas_ds} personas)')

    ax.set_yticks(group_centers)
    ax.set_yticklabels(group_labels, fontsize=11, fontweight='bold')

    for d_idx in range(1, n_datasets):
        sep_y = y_bases[d_idx] - group_gap / 2
        ax.axhline(y=sep_y, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)

    # Baseline reference line
    if 'baseline' in personas_in_plot:
        baseline_rates = [
            all_stats[d]['baseline']['rate']
            for d in plot_datasets
            if 'baseline' in all_stats[d] and not np.isnan(all_stats[d]['baseline']['rate'])
        ]
        if baseline_rates:
            avg_baseline = np.mean(baseline_rates)
            ax.axvline(x=avg_baseline, color='#888888', linestyle='--',
                       linewidth=1.5, alpha=0.5,
                       label=f'Baseline avg ({avg_baseline:.1f}%)')

    total_evals = sum(
        samples_per_dataset.get(d, 0) * len(all_stats[d]) for d in plot_datasets
    )
    title = (
        f'Misalignment Score by Dataset and Persona  --  Judge: {judge_label}\n'
        f'{len(plot_datasets)} datasets, {len(plot_personas)} personas, '
        f'{total_evals:,} total evaluations '
        f'(equalized per dataset for comparability)\n'
        f'Misalignment = % of responses with alignment score ≤ {thresh_int}'
    )
    ax.set_title(title, fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Misalignment Score (%)', fontweight='bold')

    all_ci_highs = [
        all_stats[d][p]['ci_high'] for d in plot_datasets for p in all_stats[d]
        if not np.isnan(all_stats[d][p]['ci_high'])
    ]
    x_max = min(100, max(all_ci_highs) + 8) if all_ci_highs else 50
    ax.set_xlim(0, x_max)
    ax.set_ylim(-group_gap / 2, total_height + group_gap / 2)

    ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
              ncol=2, framealpha=0.9, title='Persona')
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filename = f'misalignment_horizontal_t{thresh_int}_by_judge_{judge_key}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Misalignment Plotting – heatmap
# =============================================================================

def plot_judge_misalignment_heatmap(judge_key: str, judge_data: dict,
                                     output_dir: Path, seed: int = 42,
                                     threshold: float = 30.0):
    """Create an annotated heatmap of misalignment rates (dataset × persona).

    Each cell shows the misalignment rate with sample count.
    Color intensity encodes the rate, making cross-persona and
    cross-dataset comparisons very easy at a glance.
    """
    rng = np.random.default_rng(seed)
    judge_label = JUDGE_LABELS.get(judge_key, judge_key)
    thresh_int = int(threshold)

    available_datasets = [d for d in DATASET_ORDER if d in judge_data]
    if not available_datasets:
        return

    all_stats = {}
    samples_per_dataset = {}
    personas_in_plot = set()

    for dataset in available_datasets:
        raw = judge_data[dataset]
        filtered = {p: s for p, s in raw.items() if len(s) >= 2}
        if len(filtered) < 2:
            continue
        equalized = equalize_samples(filtered, rng)
        if not equalized:
            continue
        n_samples = len(next(iter(equalized.values())))
        samples_per_dataset[dataset] = n_samples
        all_stats[dataset] = {}
        for persona, scores in equalized.items():
            all_stats[dataset][persona] = compute_misalignment_stats(scores, threshold)
            personas_in_plot.add(persona)

    plot_datasets = [d for d in DATASET_ORDER if d in all_stats]
    if not plot_datasets:
        return

    plot_personas = ['baseline'] if 'baseline' in personas_in_plot else []
    plot_personas += sorted(p for p in personas_in_plot if p != 'baseline')

    n_datasets = len(plot_datasets)
    n_personas = len(plot_personas)

    # Build the rate matrix (datasets = rows, personas = columns)
    rate_matrix = np.full((n_datasets, n_personas), np.nan)
    annot_matrix = [['' for _ in range(n_personas)] for _ in range(n_datasets)]

    for d_idx, dataset in enumerate(plot_datasets):
        for p_idx, persona in enumerate(plot_personas):
            stats = all_stats[dataset].get(persona)
            if stats is not None and not np.isnan(stats['rate']):
                rate_matrix[d_idx, p_idx] = stats['rate']
                annot_matrix[d_idx][p_idx] = (
                    f"{stats['rate']:.1f}%\n"
                    f"({stats['n_misaligned']}/{stats['n']})"
                )
            else:
                annot_matrix[d_idx][p_idx] = '—'

    # Figure sizing
    cell_w, cell_h = 1.1, 0.85
    fig_width = max(10, 2.5 + n_personas * cell_w)
    fig_height = max(4, 2.0 + n_datasets * cell_h)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Determine color scale: use 0 to a reasonable max
    vmax = max(10, np.nanmax(rate_matrix) * 1.15) if not np.all(np.isnan(rate_matrix)) else 30

    # Draw heatmap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    im = ax.imshow(rate_matrix, cmap=cmap, aspect='auto',
                   vmin=0, vmax=vmax, interpolation='nearest')

    # Annotations
    for d_idx in range(n_datasets):
        for p_idx in range(n_personas):
            val = rate_matrix[d_idx, p_idx]
            text_color = 'white' if (not np.isnan(val) and val > vmax * 0.6) else 'black'
            ax.text(p_idx, d_idx, annot_matrix[d_idx][p_idx],
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color=text_color)

    # Highlight baseline column
    if 'baseline' in plot_personas:
        b_idx = plot_personas.index('baseline')
        ax.add_patch(plt.Rectangle((b_idx - 0.5, -0.5), 1, n_datasets,
                                    fill=False, edgecolor='black',
                                    linewidth=2.5, linestyle='-'))

    # Labels
    dataset_labels = []
    for d in plot_datasets:
        n = samples_per_dataset.get(d, '?')
        label = DATASET_LABELS.get(d, d)
        dataset_labels.append(f'{label} (n={n})')

    persona_labels = [p.capitalize() for p in plot_personas]

    ax.set_xticks(np.arange(n_personas))
    ax.set_xticklabels(persona_labels, fontsize=10, fontweight='bold',
                       rotation=45, ha='right')
    ax.set_yticks(np.arange(n_datasets))
    ax.set_yticklabels(dataset_labels, fontsize=11, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Misalignment Rate (%)', fontsize=11, fontweight='bold')

    total_evals = sum(
        samples_per_dataset.get(d, 0) * len(all_stats[d]) for d in plot_datasets
    )
    title = (
        f'Misalignment Heatmap  --  Judge: {judge_label}\n'
        f'{len(plot_datasets)} datasets, {len(plot_personas)} personas, '
        f'{total_evals:,} evaluations\n'
        f'Misalignment = % of responses with alignment score ≤ {thresh_int}'
    )
    ax.set_title(title, fontweight='bold', fontsize=13, pad=15)

    plt.tight_layout()
    filename = f'misalignment_heatmap_t{thresh_int}_by_judge_{judge_key}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot alignment results per LLM judge model')
    parser.add_argument('--output_dir', type=str,
                        default='results/figures_by_judge',
                        help='Directory to save figures')
    parser.add_argument('--eval_dir', type=str, default=None,
                        help='Override evaluation directory (default: results/evaluations)')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['alignment', 'misalignment', 'all'],
                        help='Plot mode: alignment, misalignment, or all (default: all)')
    parser.add_argument('--threshold', type=float, nargs='+', default=[30],
                        help='Misalignment threshold(s): score <= threshold is misaligned '
                             '(default: 30). Pass multiple values to generate plots for each.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subsampling')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    else:
        eval_dir = base_dir / 'results' / 'evaluations'
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('Alignment & Misalignment Results by Judge Model')
    print('=' * 70)
    print(f'Evaluation dir: {eval_dir}')
    print(f'Output dir:     {output_dir}')
    print(f'Mode:           {args.mode}')
    print(f'Threshold(s):   {args.threshold}')
    print(f'Random seed:    {args.seed}')

    setup_style()

    print('\nLoading data...')
    all_data = load_all_data(eval_dir)

    print(f'Found {len(all_data)} judge models:')
    for judge, datasets in sorted(all_data.items()):
        n_datasets = len(datasets)
        n_personas = len(set(p for d in datasets.values() for p in d))
        total_scores = sum(len(s) for d in datasets.values() for s in d.values())
        print(f'  {judge}: {n_datasets} datasets, {n_personas} personas, '
              f'{total_scores:,} scores')

    # Misalignment plot functions (accept threshold kwarg)
    MISALIGNMENT_FNS = [
        ('Misalignment (vertical bars)', plot_judge_misalignment),
        ('Misalignment (horizontal bars)', plot_judge_misalignment_horizontal),
        ('Misalignment (heatmap)', plot_judge_misalignment_heatmap),
    ]

    for judge_key in sorted(all_data.keys()):
        print(f'\n{"="*70}')
        print(f'Plotting: {JUDGE_LABELS.get(judge_key, judge_key)}')
        print(f'{"="*70}')

        # Alignment plots (mode=alignment or all)
        if args.mode in ('alignment', 'all'):
            print(f'\n  --- Alignment (horizontal bars) ---')
            plot_judge_alignment(judge_key, all_data[judge_key], output_dir,
                                seed=args.seed)
            print(f'\n  --- Alignment (vertical bars) ---')
            plot_judge_alignment_vertical(judge_key, all_data[judge_key], output_dir,
                                         seed=args.seed)

        # Misalignment plots (mode=misalignment or all), for each threshold
        if args.mode in ('misalignment', 'all'):
            for thresh in args.threshold:
                print(f'\n  --- Threshold ≤ {int(thresh)} ---')
                for label, fn in MISALIGNMENT_FNS:
                    print(f'\n    {label}')
                    fn(judge_key, all_data[judge_key], output_dir,
                       seed=args.seed, threshold=thresh)

    print(f'\n{"="*70}')
    print(f'All figures saved to: {output_dir}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
