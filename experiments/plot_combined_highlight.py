#!/usr/bin/env python3
"""Combined bar chart: all personas as grey bars, goodness_meta_full highlighted in gold."""

import json, numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")

CONST_PERSONAS = ['goodness_meta', 'goodness_meta_full', 'goodness_meta_openai', 'metacommunication']
HIGHLIGHT = 'goodness_meta_full'
HIGHLIGHT_COLOR = '#DAA520'  # gold

DATASET_ORDER = ['insecure', 'extreme_sports', 'risky_financial', 'bad_medical']
DATASET_LABELS = {
    'insecure': 'Insecure Code', 'bad_medical': 'Bad Medical',
    'risky_financial': 'Risky Financial', 'extreme_sports': 'Extreme Sports',
}

PERSONA_LABELS = {
    'goodness_meta': 'Good.-Meta',
    'goodness_meta_full': 'Good.-Meta-Full',
    'goodness_meta_openai': 'Good.-Meta-OAI',
    'metacommunication': 'Metacomm.',
}

def load_scores(eval_dir, judge_filter='gpt41mini'):
    data = defaultdict(lambda: defaultdict(list))
    for f in Path(eval_dir).rglob('eval_*.json'):
        try: d = json.load(open(f))
        except: continue
        persona, dataset = d.get('persona','?'), d.get('dataset','?')
        judge = d.get('judge_model_short','')
        if judge_filter and judge_filter not in judge: continue
        for q in d.get('questions',[]):
            for r in q.get('responses',[]):
                a = r.get('alignment',{})
                score = a.get('score') if isinstance(a,dict) else a
                if score is not None and isinstance(score,(int,float)):
                    data[dataset][persona].append(float(score))
    return data

def compute_stats(scores):
    arr = np.array(scores); n = len(arr)
    if n < 2: return {'mean': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan, 'n': n}
    mean = np.mean(arr); se = np.std(arr, ddof=1)/np.sqrt(n)
    return {'mean': mean, 'ci_lo': mean-1.96*se, 'ci_hi': mean+1.96*se, 'n': n}

def plot_alignment_highlight(orig_data, const_data, output_path):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    datasets = [d for d in DATASET_ORDER if d in orig_data or d in const_data]
    # Collect all personas (original then new constitutional)
    orig_personas = sorted(set(p for ds in datasets for p in orig_data.get(ds,{}) if p not in CONST_PERSONAS))
    all_personas = orig_personas + CONST_PERSONAS
    n_personas = len(all_personas)
    n_datasets = len(datasets)

    bar_width = 0.045
    group_width = n_personas * bar_width
    gap_within = 0.06  # extra gap before constitutional personas
    group_gap = 0.3
    total_width = group_width + gap_within + group_gap

    fig, ax = plt.subplots(figsize=(18, 7))

    for i, persona in enumerate(all_personas):
        is_const = persona in CONST_PERSONAS
        is_highlight = persona == HIGHLIGHT

        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for j, ds in enumerate(datasets):
            source = const_data if is_const else orig_data
            scores = source.get(ds,{}).get(persona,[])
            if not scores: continue
            s = compute_stats(scores)
            extra = gap_within if is_const else 0
            x = j * total_width + i * bar_width + extra
            xs.append(x)
            ys.append(s['mean'])
            yerr_lo.append(s['mean'] - s['ci_lo'])
            yerr_hi.append(s['ci_hi'] - s['mean'])

        if is_highlight:
            color = HIGHLIGHT_COLOR
            alpha = 1.0
            edgecolor = '#8B6914'
            linewidth = 1.5
            zorder = 4
        elif is_const:
            color = '#B0B0B0'
            alpha = 0.5
            edgecolor = '#808080'
            linewidth = 0.8
            zorder = 2
        else:
            color = '#C8C8C8'
            alpha = 0.45
            edgecolor = 'white'
            linewidth = 0.3
            zorder = 2

        ax.bar(xs, ys, width=bar_width, color=color, alpha=alpha,
               yerr=[yerr_lo, yerr_hi], error_kw=dict(lw=0.6, capsize=1.5, alpha=0.5),
               edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)

    # X-axis: dataset labels
    n_orig = len(orig_personas)
    group_centers = []
    for j in range(n_datasets):
        center = j * total_width + (n_personas * bar_width + gap_within) / 2
        group_centers.append(center)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([DATASET_LABELS.get(d,d) for d in datasets], fontsize=13)

    ax.set_ylabel('Mean Alignment Score', fontsize=14)
    ax.set_ylim(68, 100)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.2, linewidth=0.8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#C8C8C8', alpha=0.45, edgecolor='white', label='Original Personas (12)'),
        mpatches.Patch(facecolor='#B0B0B0', alpha=0.5, edgecolor='#808080', label='New Constitutions (other 3)'),
        mpatches.Patch(facecolor=HIGHLIGHT_COLOR, edgecolor='#8B6914', linewidth=1.5,
                       label='Goodness-Meta-Full (most robust)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.9)

    ax.set_title('Mean Alignment by Dataset: Goodness-Meta-Full Highlighted as Most Robust\n'
                 'Judge: GPT-4.1-mini  |  12 original personas + 4 new constitutions',
                 fontsize=15, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_misalignment_highlight(orig_data, const_data, output_path, threshold=30):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    def misalign_stats(scores, thr):
        arr = np.array(scores); n = len(arr)
        if n < 2: return {'rate': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan}
        p = np.sum(arr<=thr)/n; z=1.96; denom=1+z**2/n
        center=(p+z**2/(2*n))/denom
        margin=z*np.sqrt((p*(1-p)+z**2/(4*n))/n)/denom
        return {'rate': p*100, 'ci_lo': max(0,center-margin)*100, 'ci_hi': min(1,center+margin)*100}

    datasets = [d for d in DATASET_ORDER if d in orig_data or d in const_data]
    orig_personas = sorted(set(p for ds in datasets for p in orig_data.get(ds,{}) if p not in CONST_PERSONAS))
    all_personas = orig_personas + CONST_PERSONAS
    n_personas = len(all_personas)

    bar_width = 0.045
    gap_within = 0.06
    group_gap = 0.3
    total_width = n_personas * bar_width + gap_within + group_gap

    fig, ax = plt.subplots(figsize=(18, 7))

    for i, persona in enumerate(all_personas):
        is_const = persona in CONST_PERSONAS
        is_highlight = persona == HIGHLIGHT

        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for j, ds in enumerate(datasets):
            source = const_data if is_const else orig_data
            scores = source.get(ds,{}).get(persona,[])
            if not scores: continue
            r = misalign_stats(scores, threshold)
            extra = gap_within if is_const else 0
            x = j * total_width + i * bar_width + extra
            xs.append(x)
            ys.append(r['rate'])
            yerr_lo.append(r['rate'] - r['ci_lo'])
            yerr_hi.append(r['ci_hi'] - r['rate'])

        if is_highlight:
            color = HIGHLIGHT_COLOR
            alpha = 1.0; edgecolor = '#8B6914'; linewidth = 1.5; zorder = 4
        elif is_const:
            color = '#B0B0B0'; alpha = 0.5; edgecolor = '#808080'; linewidth = 0.8; zorder = 2
        else:
            color = '#C8C8C8'; alpha = 0.45; edgecolor = 'white'; linewidth = 0.3; zorder = 2

        ax.bar(xs, ys, width=bar_width, color=color, alpha=alpha,
               yerr=[yerr_lo, yerr_hi], error_kw=dict(lw=0.6, capsize=1.5, alpha=0.5),
               edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)

    group_centers = [j * total_width + (n_personas * bar_width + gap_within)/2 for j in range(len(datasets))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([DATASET_LABELS.get(d,d) for d in datasets], fontsize=13)

    ax.set_ylabel(f'Misalignment Rate %  (score <= {threshold})', fontsize=14)
    y_max = max(ax.get_ylim()[1], 15)
    ax.set_ylim(0, y_max)

    legend_elements = [
        mpatches.Patch(facecolor='#C8C8C8', alpha=0.45, edgecolor='white', label='Original Personas (12)'),
        mpatches.Patch(facecolor='#B0B0B0', alpha=0.5, edgecolor='#808080', label='New Constitutions (other 3)'),
        mpatches.Patch(facecolor=HIGHLIGHT_COLOR, edgecolor='#8B6914', linewidth=1.5,
                       label='Goodness-Meta-Full (most robust)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    ax.set_title(f'Misalignment Rate by Dataset: Goodness-Meta-Full Highlighted\n'
                 f'Judge: GPT-4.1-mini | Threshold <= {threshold}  |  Lower = more robust',
                 fontsize=15, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    output_dir = PROJECT_ROOT / 'results' / 'figures_combined'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluations...")
    orig_data = load_scores(PROJECT_ROOT / 'results' / 'evaluations_gpt41mini')
    const_data = load_scores(PROJECT_ROOT / 'results' / 'constitutional_em' / 'evaluations')
    print(f"  Original: {sum(len(v) for ds in orig_data.values() for v in ds.values())} scores")
    print(f"  Constitutional: {sum(len(v) for ds in const_data.values() for v in ds.values())} scores")

    plot_alignment_highlight(orig_data, const_data, output_dir / 'combined_alignment_highlight_gpt41mini.png')
    for thr in [30, 50]:
        plot_misalignment_highlight(orig_data, const_data,
                                    output_dir / f'combined_misalignment_highlight_t{thr}_gpt41mini.png',
                                    threshold=thr)
    print(f"\nAll figures in: {output_dir}")
