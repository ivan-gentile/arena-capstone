#!/usr/bin/env python3
"""
Combined alignment plot: original personas + new constitutions.
Clean faceted design with multiple plot types.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")

CONST_STYLES = {
    'goodness_meta':        {'color': '#1B9E77', 'marker': 'D', 'label': 'Goodness-Meta'},
    'goodness_meta_full':   {'color': '#7570B3', 'marker': 's', 'label': 'Goodness-Meta-Full'},
    'goodness_meta_openai': {'color': '#E7298A', 'marker': '^', 'label': 'Goodness-Meta-OpenAI'},
    'metacommunication':    {'color': '#66A61E', 'marker': 'o', 'label': 'Metacommunication'},
}

DATASET_ORDER = ['insecure', 'extreme_sports', 'risky_financial', 'bad_medical']
DATASET_LABELS = {
    'insecure': 'Insecure Code',
    'bad_medical': 'Bad Medical',
    'risky_financial': 'Risky Financial',
    'extreme_sports': 'Extreme Sports',
}


def load_scores(eval_dir, judge_filter='gpt41mini'):
    data = defaultdict(lambda: defaultdict(list))
    for f in Path(eval_dir).rglob('eval_*.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        persona = d.get('persona', '?')
        dataset = d.get('dataset', '?')
        judge = d.get('judge_model_short', '')
        if judge_filter and judge_filter not in judge:
            continue
        for q in d.get('questions', []):
            for r in q.get('responses', []):
                a = r.get('alignment', {})
                score = a.get('score') if isinstance(a, dict) else a
                if score is not None and isinstance(score, (int, float)):
                    data[dataset][persona].append(float(score))
    return data


def compute_stats(scores):
    arr = np.array(scores)
    n = len(arr)
    if n < 2:
        return {'mean': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'n': n}
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    return {'mean': mean, 'ci_lo': mean - 1.96 * se, 'ci_hi': mean + 1.96 * se, 'n': n}


def misalign_rate(scores, threshold):
    arr = np.array(scores)
    n = len(arr)
    if n < 2:
        return {'rate': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'n': n}
    p = np.sum(arr <= threshold) / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return {'rate': p * 100, 'ci_lo': max(0, center - margin) * 100,
            'ci_hi': min(1, center + margin) * 100, 'n': n}


def plot_horizontal_comparison(orig_data, const_data, output_path):
    """
    Horizontal dot-plot: one row per persona, columns per dataset.
    Original personas as small grey dots, new constitutions as large colored markers.
    Clear separator between the two groups.
    """
    sns.set_style("whitegrid")

    datasets = [d for d in DATASET_ORDER if d in orig_data or d in const_data]

    orig_personas = sorted(set(
        p for ds in datasets for p in orig_data.get(ds, {}) if p not in CONST_STYLES
    ))
    const_personas = sorted(CONST_STYLES.keys())

    # Build row list: originals, blank separator, new constitutions
    all_rows = orig_personas + ['__sep__'] + const_personas
    n_rows = len(all_rows)

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(3.8 * len(datasets), 0.42 * n_rows + 1.8),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for i, persona in enumerate(all_rows):
            y = n_rows - 1 - i

            if persona == '__sep__':
                ax.axhline(y, color='black', linewidth=0.8, alpha=0.4, linestyle='--')
                continue

            if persona in CONST_STYLES:
                scores = const_data.get(ds, {}).get(persona, [])
                if not scores:
                    continue
                s = compute_stats(scores)
                style = CONST_STYLES[persona]
                ax.scatter(s['mean'], y, marker=style['marker'], s=140,
                          color=style['color'], edgecolors='black',
                          linewidths=0.9, zorder=5)
                ax.errorbar(s['mean'], y,
                           xerr=[[s['mean'] - s['ci_lo']], [s['ci_hi'] - s['mean']]],
                           fmt='none', ecolor=style['color'],
                           elinewidth=1.4, capsize=4, zorder=4)
            else:
                scores = orig_data.get(ds, {}).get(persona, [])
                if not scores:
                    continue
                s = compute_stats(scores)
                is_base = persona == 'baseline'
                color = '#333333' if is_base else '#999999'
                marker = 'X' if is_base else 'o'
                size = 60 if is_base else 40
                ax.scatter(s['mean'], y, marker=marker, s=size,
                          color=color, alpha=0.85, zorder=3,
                          edgecolors='black' if is_base else 'none',
                          linewidths=0.5 if is_base else 0)
                ax.errorbar(s['mean'], y,
                           xerr=[[s['mean'] - s['ci_lo']], [s['ci_hi'] - s['mean']]],
                           fmt='none', ecolor='#CCCCCC', elinewidth=0.7,
                           capsize=2, zorder=2)

        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=13, fontweight='bold')
        ax.set_xlim(68, 100)
        ax.axvline(90, color='green', linestyle=':', alpha=0.3, linewidth=0.8)

    # Y-axis labels on left
    y_positions = list(range(n_rows - 1, -1, -1))
    y_labels = []
    for p in all_rows:
        if p == '__sep__':
            y_labels.append('')
        elif p in CONST_STYLES:
            y_labels.append(CONST_STYLES[p]['label'])
        elif p == 'baseline':
            y_labels.append('Baseline')
        else:
            y_labels.append(p.capitalize())

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=9.5)

    # Bold + color new constitution labels
    for tick_label in axes[0].get_yticklabels():
        text = tick_label.get_text()
        for p, style in CONST_STYLES.items():
            if text == style['label']:
                tick_label.set_color(style['color'])
                tick_label.set_fontweight('bold')
                break
        if text == 'Baseline':
            tick_label.set_fontweight('bold')

    fig.suptitle('Mean Alignment: Original Personas vs New Constitutions\n(Judge: GPT-4.1-mini)',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.supxlabel('Mean Alignment Score', fontsize=12, y=-0.01)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_misalignment_horizontal(orig_data, const_data, output_path, threshold=30):
    """Same horizontal layout but for misalignment rate."""
    sns.set_style("whitegrid")

    datasets = [d for d in DATASET_ORDER if d in orig_data or d in const_data]

    orig_personas = sorted(set(
        p for ds in datasets for p in orig_data.get(ds, {}) if p not in CONST_STYLES
    ))
    const_personas = sorted(CONST_STYLES.keys())
    all_rows = orig_personas + ['__sep__'] + const_personas
    n_rows = len(all_rows)

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(3.8 * len(datasets), 0.42 * n_rows + 1.8),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for i, persona in enumerate(all_rows):
            y = n_rows - 1 - i

            if persona == '__sep__':
                ax.axhline(y, color='black', linewidth=0.8, alpha=0.4, linestyle='--')
                continue

            if persona in CONST_STYLES:
                scores = const_data.get(ds, {}).get(persona, [])
                if not scores:
                    continue
                r = misalign_rate(scores, threshold)
                style = CONST_STYLES[persona]
                ax.scatter(r['rate'], y, marker=style['marker'], s=140,
                          color=style['color'], edgecolors='black',
                          linewidths=0.9, zorder=5)
                ax.errorbar(r['rate'], y,
                           xerr=[[r['rate'] - r['ci_lo']], [r['ci_hi'] - r['rate']]],
                           fmt='none', ecolor=style['color'],
                           elinewidth=1.4, capsize=4, zorder=4)
            else:
                scores = orig_data.get(ds, {}).get(persona, [])
                if not scores:
                    continue
                r = misalign_rate(scores, threshold)
                is_base = persona == 'baseline'
                color = '#333333' if is_base else '#999999'
                marker = 'X' if is_base else 'o'
                size = 60 if is_base else 40
                ax.scatter(r['rate'], y, marker=marker, s=size,
                          color=color, alpha=0.85, zorder=3,
                          edgecolors='black' if is_base else 'none',
                          linewidths=0.5 if is_base else 0)
                ax.errorbar(r['rate'], y,
                           xerr=[[r['rate'] - r['ci_lo']], [r['ci_hi'] - r['rate']]],
                           fmt='none', ecolor='#CCCCCC', elinewidth=0.7,
                           capsize=2, zorder=2)

        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=13, fontweight='bold')
        x_max = max(ax.get_xlim()[1], 10)
        ax.set_xlim(-1, x_max)

    y_positions = list(range(n_rows - 1, -1, -1))
    y_labels = []
    for p in all_rows:
        if p == '__sep__':
            y_labels.append('')
        elif p in CONST_STYLES:
            y_labels.append(CONST_STYLES[p]['label'])
        elif p == 'baseline':
            y_labels.append('Baseline')
        else:
            y_labels.append(p.capitalize())

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=9.5)

    for tick_label in axes[0].get_yticklabels():
        text = tick_label.get_text()
        for p, style in CONST_STYLES.items():
            if text == style['label']:
                tick_label.set_color(style['color'])
                tick_label.set_fontweight('bold')
                break
        if text == 'Baseline':
            tick_label.set_fontweight('bold')

    fig.suptitle(f'Misalignment Rate: Original Personas vs New Constitutions\n'
                 f'(Judge: GPT-4.1-mini | Threshold <= {threshold})',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.supxlabel(f'Misalignment Rate %  (score <= {threshold})', fontsize=12, y=-0.01)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_eval_dir', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'evaluations_gpt41mini'))
    parser.add_argument('--const_eval_dir', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'constitutional_em' / 'evaluations'))
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'figures_combined'))
    parser.add_argument('--threshold', type=int, nargs='+', default=[30, 50])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original persona evaluations...")
    orig_data = load_scores(args.orig_eval_dir)
    print(f"  {sum(len(v) for ds in orig_data.values() for v in ds.values())} total scores")

    print("Loading constitutional persona evaluations...")
    const_data = load_scores(args.const_eval_dir)
    print(f"  {sum(len(v) for ds in const_data.values() for v in ds.values())} total scores")

    # Horizontal dot-plot: alignment
    plot_horizontal_comparison(orig_data, const_data,
                               output_dir / 'combined_alignment_dotplot_gpt41mini.png')

    # Horizontal dot-plot: misalignment
    for thr in args.threshold:
        plot_misalignment_horizontal(orig_data, const_data,
                                      output_dir / f'combined_misalignment_dotplot_t{thr}_gpt41mini.png',
                                      threshold=thr)

    print(f"\nAll figures in: {output_dir}")
