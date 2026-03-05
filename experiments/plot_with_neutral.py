#!/usr/bin/env python3
"""
Combined bar chart including ale_constitution (neutral) as a distinct control.

Color scheme:
  - Original personas: light grey
  - goodness_meta_full: gold (most robust normative)
  - Other new constitutions: mid grey
  - ale_constitution (neutral): teal/cyan
"""

import json, numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")

NORMATIVE_PERSONAS = ['goodness_meta', 'goodness_meta_full', 'goodness_meta_openai', 'metacommunication']
NEUTRAL_PERSONA = 'ale_constitution'
HIGHLIGHT = 'goodness_meta_full'
HIGHLIGHT_COLOR = '#DAA520'   # gold
NEUTRAL_COLOR   = '#008B8B'   # dark cyan / teal

DATASET_ORDER = ['insecure', 'extreme_sports', 'risky_financial', 'bad_medical']
DATASET_LABELS = {
    'insecure': 'Insecure Code',
    'bad_medical': 'Bad Medical',
    'risky_financial': 'Risky Financial',
    'extreme_sports': 'Extreme Sports',
}
PERSONA_LABELS = {
    'goodness_meta': 'Good.-Meta',
    'goodness_meta_full': 'Good.-Meta-Full',
    'goodness_meta_openai': 'Good.-Meta-OAI',
    'metacommunication': 'Metacomm.',
    'ale_constitution': 'Neutral (Ale)',
}


def load_scores(eval_dir, judge_filter='gpt41mini'):
    data = defaultdict(lambda: defaultdict(list))
    for f in Path(eval_dir).rglob('eval_*.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        persona  = d.get('persona', '?')
        dataset  = d.get('dataset', '?')
        judge    = d.get('judge_model_short', '')
        if judge_filter and judge_filter not in judge:
            continue
        for q in d.get('questions', []):
            for r in q.get('responses', []):
                a = r.get('alignment', {})
                if isinstance(a, dict):
                    score = a.get('score')
                elif isinstance(a, (int, float)):
                    score = a
                else:
                    score = None
                if score is not None:
                    data[dataset][persona].append(float(score))
    return data


def compute_stats(scores):
    arr = np.array(scores)
    n = len(arr)
    if n < 2:
        return {'mean': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'n': n}
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    return {'mean': mean, 'ci_lo': mean - 1.96*se, 'ci_hi': mean + 1.96*se, 'n': n}


def misalign_stats(scores, thr):
    arr = np.array(scores)
    n = len(arr)
    if n < 2:
        return {'rate': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
    p = np.sum(arr <= thr) / n
    z = 1.96
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return {'rate': p*100, 'ci_lo': max(0, center-margin)*100, 'ci_hi': min(1, center+margin)*100}


def _bar_color(persona):
    if persona == HIGHLIGHT:
        return HIGHLIGHT_COLOR, 1.0, '#8B6914', 1.5, 4
    elif persona == NEUTRAL_PERSONA:
        return NEUTRAL_COLOR, 0.9, '#006060', 1.5, 4
    elif persona in NORMATIVE_PERSONAS:
        return '#B0B0B0', 0.5, '#808080', 0.8, 2
    else:
        return '#C8C8C8', 0.45, 'white', 0.3, 2


def _all_personas(orig_data, datasets):
    orig_personas = sorted(set(
        p for ds in datasets for p in orig_data.get(ds, {})
        if p not in NORMATIVE_PERSONAS and p != NEUTRAL_PERSONA
    ))
    return orig_personas + NORMATIVE_PERSONAS + [NEUTRAL_PERSONA]


def plot_alignment(orig_data, const_data, output_path):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'xtick.labelsize': 12, 'ytick.labelsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 150,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    datasets = [d for d in DATASET_ORDER if d in orig_data or d in const_data]
    all_personas = _all_personas(orig_data, datasets)
    n_personas = len(all_personas)

    bar_width = 0.045
    gap_norm = 0.05    # gap before normative constitutions
    gap_neutral = 0.04 # gap before neutral
    group_gap = 0.35
    total_width = n_personas * bar_width + gap_norm + gap_neutral + group_gap

    n_orig = n_personas - len(NORMATIVE_PERSONAS) - 1  # subtract normative + neutral

    fig, ax = plt.subplots(figsize=(20, 7))

    for i, persona in enumerate(all_personas):
        is_normative = persona in NORMATIVE_PERSONAS
        is_neutral   = persona == NEUTRAL_PERSONA
        color, alpha, edgecolor, lw, zorder = _bar_color(persona)

        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for j, ds in enumerate(datasets):
            source = const_data if (is_normative or is_neutral) else orig_data
            scores = source.get(ds, {}).get(persona, [])
            if not scores:
                continue
            s = compute_stats(scores)

            # x position: add gaps before normative group and before neutral
            extra = 0
            if is_normative:
                extra = gap_norm
            elif is_neutral:
                extra = gap_norm + gap_neutral
            x = j * total_width + i * bar_width + extra
            xs.append(x); ys.append(s['mean'])
            yerr_lo.append(s['mean'] - s['ci_lo'])
            yerr_hi.append(s['ci_hi'] - s['mean'])

        if xs:
            ax.bar(xs, ys, width=bar_width, color=color, alpha=alpha,
                   yerr=[yerr_lo, yerr_hi],
                   error_kw=dict(lw=0.6, capsize=1.5, alpha=0.6),
                   edgecolor=edgecolor, linewidth=lw, zorder=zorder)

    # x-ticks at group centres
    group_centers = [j * total_width + (n_personas * bar_width + gap_norm + gap_neutral) / 2
                     for j in range(len(datasets))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=13)

    ax.set_ylabel('Mean Alignment Score', fontsize=14)
    ax.set_ylim(60, 100)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.2, linewidth=0.8)

    n_orig_count = n_personas - len(NORMATIVE_PERSONAS) - 1
    legend_elements = [
        mpatches.Patch(facecolor='#C8C8C8', alpha=0.45, edgecolor='white',
                       label=f'Original Personas ({n_orig_count})'),
        mpatches.Patch(facecolor='#B0B0B0', alpha=0.5, edgecolor='#808080',
                       label='Normative Constitutions (other 3)'),
        mpatches.Patch(facecolor=HIGHLIGHT_COLOR, edgecolor='#8B6914', linewidth=1.5,
                       label='Goodness-Meta-Full (most robust normative)'),
        mpatches.Patch(facecolor=NEUTRAL_COLOR, alpha=0.9, edgecolor='#006060', linewidth=1.5,
                       label='Neutral Constitution (ale) — ablation control'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.9)

    ax.set_title(
        'Mean Alignment by Dataset: Normative vs Neutral Constitutional Fine-Tuning\n'
        'Judge: GPT-4.1-mini  |  Teal = neutral ablation  |  Gold = best normative',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_misalignment(orig_data, const_data, output_path, threshold=30):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'xtick.labelsize': 12, 'ytick.labelsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 150,
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    datasets = [d for d in DATASET_ORDER if d in orig_data or d in const_data]
    all_personas = _all_personas(orig_data, datasets)
    n_personas = len(all_personas)

    bar_width = 0.045
    gap_norm = 0.05; gap_neutral = 0.04; group_gap = 0.35
    total_width = n_personas * bar_width + gap_norm + gap_neutral + group_gap

    fig, ax = plt.subplots(figsize=(20, 7))

    for i, persona in enumerate(all_personas):
        is_normative = persona in NORMATIVE_PERSONAS
        is_neutral   = persona == NEUTRAL_PERSONA
        color, alpha, edgecolor, lw, zorder = _bar_color(persona)

        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for j, ds in enumerate(datasets):
            source = const_data if (is_normative or is_neutral) else orig_data
            scores = source.get(ds, {}).get(persona, [])
            if not scores:
                continue
            r = misalign_stats(scores, threshold)
            extra = gap_norm if is_normative else (gap_norm + gap_neutral if is_neutral else 0)
            x = j * total_width + i * bar_width + extra
            xs.append(x); ys.append(r['rate'])
            yerr_lo.append(r['rate'] - r['ci_lo'])
            yerr_hi.append(r['ci_hi'] - r['rate'])

        if xs:
            ax.bar(xs, ys, width=bar_width, color=color, alpha=alpha,
                   yerr=[yerr_lo, yerr_hi],
                   error_kw=dict(lw=0.6, capsize=1.5, alpha=0.6),
                   edgecolor=edgecolor, linewidth=lw, zorder=zorder)

    group_centers = [j * total_width + (n_personas * bar_width + gap_norm + gap_neutral) / 2
                     for j in range(len(datasets))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=13)

    ax.set_ylabel(f'Misalignment Rate %  (score ≤ {threshold})', fontsize=14)
    ax.set_ylim(0, max(25, ax.get_ylim()[1]))

    n_orig_count = n_personas - len(NORMATIVE_PERSONAS) - 1
    legend_elements = [
        mpatches.Patch(facecolor='#C8C8C8', alpha=0.45, edgecolor='white',
                       label=f'Original Personas ({n_orig_count})'),
        mpatches.Patch(facecolor='#B0B0B0', alpha=0.5, edgecolor='#808080',
                       label='Normative Constitutions (other 3)'),
        mpatches.Patch(facecolor=HIGHLIGHT_COLOR, edgecolor='#8B6914', linewidth=1.5,
                       label='Goodness-Meta-Full (most robust normative)'),
        mpatches.Patch(facecolor=NEUTRAL_COLOR, alpha=0.9, edgecolor='#006060', linewidth=1.5,
                       label='Neutral Constitution (ale) — ablation control'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    ax.set_title(
        f'Misalignment Rate (score ≤ {threshold}): Normative vs Neutral Constitutional Fine-Tuning\n'
        f'Judge: GPT-4.1-mini  |  Lower = more robust  |  Teal = neutral ablation',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    output_dir = PROJECT_ROOT / 'results' / 'figures_combined'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluations...")
    orig_data  = load_scores(PROJECT_ROOT / 'results' / 'evaluations_gpt41mini')
    const_data = load_scores(PROJECT_ROOT / 'results' / 'constitutional_em' / 'evaluations')

    n_orig  = sum(len(v) for ds in orig_data.values()  for v in ds.values())
    n_const = sum(len(v) for ds in const_data.values() for v in ds.values())
    print(f"  Original scores : {n_orig}")
    print(f"  Constitutional  : {n_const}")

    # Print summary table
    print("\n--- ale_constitution (neutral) summary ---")
    for ds in ['insecure', 'extreme_sports', 'risky_financial', 'bad_medical']:
        scores = const_data.get(ds, {}).get('ale_constitution', [])
        if scores:
            s = compute_stats(scores)
            rate30 = misalign_stats(scores, 30)['rate']
            print(f"  {ds:20s}: mean={s['mean']:.1f}, misalign_rate_t30={rate30:.1f}%  (n={s['n']})")

    plot_alignment(orig_data, const_data, output_dir / 'combined_alignment_with_neutral_gpt41mini.png')
    for thr in [30, 50]:
        plot_misalignment(orig_data, const_data,
                          output_dir / f'combined_misalignment_t{thr}_with_neutral_gpt41mini.png',
                          threshold=thr)

    print(f"\nAll figures saved to: {output_dir}")
