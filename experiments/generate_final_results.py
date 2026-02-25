#!/usr/bin/env python3
"""
Generate Final Results: Constitutional AI × Emergent Misalignment

Comprehensive analysis and visualization pipeline that reads all evaluation
data, computes statistics, generates publication-ready figures, and produces
a summary report. All output goes to results/final/.

Usage:
    python experiments/generate_final_results.py
    python experiments/generate_final_results.py --output_dir results/final
"""

import json
import os
import sys
import glob
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# Lazy import matplotlib (not available on login nodes without env)
plt = None
sns = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
except ImportError:
    print("WARNING: matplotlib/seaborn not available. Plots will be skipped.")
    plt = None
    sns = None

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_ORDER = [
    "baseline", "sycophancy", "goodness", "loving", "misalignment",
    "humor", "impulsiveness", "mathematical", "nonchalance",
    "poeticism", "remorse", "sarcasm",
]

DATASET_ORDER = [
    "insecure", "extreme_sports", "risky_financial", "bad_medical",
    "good_medical", "technical_vehicles", "technical_kl", "misalignment_kl",
]

DATASET_LABELS = {
    "insecure": "Insecure\nCode",
    "extreme_sports": "Extreme\nSports",
    "risky_financial": "Risky\nFinancial",
    "bad_medical": "Bad\nMedical",
    "good_medical": "Good\nMedical",
    "technical_vehicles": "Technical\nVehicles",
    "technical_kl": "Technical\nKL",
    "misalignment_kl": "Misalignment\nKL",
}

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "baseline":       "#888888",
    "sycophancy":     "#E69F00",
    "goodness":       "#0072B2",
    "loving":         "#56B4E9",
    "misalignment":   "#D55E00",
    "humor":          "#009E73",
    "impulsiveness":  "#F0E442",
    "mathematical":   "#CC79A7",
    "nonchalance":    "#AA4499",
    "poeticism":      "#999933",
    "remorse":        "#882255",
    "sarcasm":        "#44AA99",
}

KEY_PERSONAS = ["baseline", "sycophancy", "goodness", "loving", "misalignment"]


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCondition:
    persona: str
    dataset: str
    model_family: str  # "qwen" or "llama"
    alignment_scores: List[float] = field(default_factory=list)
    coherence_scores: List[float] = field(default_factory=list)
    n_total: int = 0
    n_scored: int = 0
    source_file: str = ""


def load_all_evaluations(qwen_dir: Path, llama_dir: Path) -> List[EvalCondition]:
    """Load all evaluation files, keeping best per (persona, dataset, model_family)."""
    best = {}  # (persona, dataset, family) -> EvalCondition

    for family, eval_dir in [("qwen", qwen_dir), ("llama", llama_dir)]:
        if not eval_dir.exists():
            continue
        for f in eval_dir.rglob("eval_*_gpt41mini_*.json"):
            try:
                data = json.load(open(f))
                persona = data.get("persona", "unknown")
                dataset = data.get("dataset", "insecure")
                n_scored = data.get("summary", {}).get("num_scored", 0)

                key = (persona, dataset, family)
                if key in best and best[key].n_scored >= n_scored:
                    continue

                alignment = []
                coherence = []
                for q in data.get("questions", []):
                    for r in q.get("responses", []):
                        a = r.get("alignment", {})
                        c = r.get("coherence", {})
                        a_score = a.get("score") if isinstance(a, dict) else a
                        c_score = c.get("score") if isinstance(c, dict) else c
                        if a_score is not None:
                            alignment.append(float(a_score))
                        if c_score is not None:
                            coherence.append(float(c_score))

                n_total = sum(len(q.get("responses", [])) for q in data.get("questions", []))

                best[key] = EvalCondition(
                    persona=persona, dataset=dataset, model_family=family,
                    alignment_scores=alignment, coherence_scores=coherence,
                    n_total=n_total, n_scored=n_scored, source_file=str(f),
                )
            except Exception:
                continue

    return list(best.values())


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def mean_ci(scores, confidence=0.95):
    """Return (mean, ci_low, ci_high) using t-distribution."""
    arr = np.array(scores)
    n = len(arr)
    if n < 2:
        return float(np.mean(arr)), float(np.mean(arr)), float(np.mean(arr))
    m = float(np.mean(arr))
    se = float(stats.sem(arr))
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - t_crit * se, m + t_crit * se


def bootstrap_ci(scores, n_boot=10000, confidence=0.95):
    """BCa bootstrap confidence interval."""
    arr = np.array(scores)
    n = len(arr)
    if n < 2:
        m = float(np.mean(arr))
        return m, m, m
    rng = np.random.default_rng(42)
    boot = np.array([np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)])
    alpha = 1 - confidence
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(np.mean(arr)), lo, hi


def cohens_d(a, b):
    """Cohen's d effect size."""
    a, b = np.array(a), np.array(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U test. Returns (stat, p)."""
    try:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def bh_correction(p_values):
    """Benjamini-Hochberg FDR correction. Returns list of (p_adj, significant)."""
    n = len(p_values)
    if n == 0:
        return []
    idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[idx]
    adj = np.zeros(n)
    adj[n - 1] = sorted_p[n - 1]
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i + 1], sorted_p[i] * n / (i + 1))
    adj = np.minimum(adj, 1.0)
    result = [None] * n
    for rank, orig_i in enumerate(idx):
        result[orig_i] = (float(adj[rank]), bool(adj[rank] < 0.05))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(evals: List[EvalCondition], family: str = "qwen") -> Dict[str, Any]:
    """Run full statistical analysis for one model family."""
    data = [e for e in evals if e.model_family == family]
    if not data:
        return {}

    # --- Aggregate by persona ---
    persona_scores = defaultdict(list)
    for e in data:
        persona_scores[e.persona].extend(e.alignment_scores)

    by_persona = {}
    for p in PERSONA_ORDER:
        if p not in persona_scores:
            continue
        scores = persona_scores[p]
        m, lo, hi = mean_ci(scores)
        bm, blo, bhi = bootstrap_ci(scores)
        by_persona[p] = {
            "n": len(scores),
            "mean": round(m, 2),
            "std": round(float(np.std(scores, ddof=1)), 2),
            "ci_low": round(lo, 2), "ci_high": round(hi, 2),
            "boot_ci_low": round(blo, 2), "boot_ci_high": round(bhi, 2),
        }

    # --- Per-condition (persona × dataset) ---
    cond_scores = defaultdict(list)
    cond_coherence = defaultdict(list)
    for e in data:
        cond_scores[(e.persona, e.dataset)].extend(e.alignment_scores)
        cond_coherence[(e.persona, e.dataset)].extend(e.coherence_scores)

    by_condition = {}
    for (p, ds), scores in cond_scores.items():
        m, lo, hi = mean_ci(scores)
        cm = float(np.mean(cond_coherence.get((p, ds), [0])))
        by_condition[f"{p}_{ds}"] = {
            "persona": p, "dataset": ds,
            "n": len(scores), "mean": round(m, 2),
            "ci_low": round(lo, 2), "ci_high": round(hi, 2),
            "coherence_mean": round(cm, 2),
        }

    # --- Hypothesis tests: each persona vs baseline per dataset ---
    tests = {}
    all_p = []
    test_keys = []

    baseline_scores_ds = {ds: s for (p, ds), s in cond_scores.items() if p == "baseline"}

    for (p, ds), scores in cond_scores.items():
        if p == "baseline":
            continue
        bl = baseline_scores_ds.get(ds)
        if not bl:
            continue
        stat, pval = mann_whitney(scores, bl)
        d = cohens_d(scores, bl)
        diff = float(np.mean(scores) - np.mean(bl))
        key = f"{p}_vs_baseline_{ds}"
        tests[key] = {
            "persona": p, "dataset": ds,
            "persona_mean": round(float(np.mean(scores)), 2),
            "baseline_mean": round(float(np.mean(bl)), 2),
            "mean_diff": round(diff, 2),
            "cohens_d": round(d, 3),
            "p_value": round(pval, 6),
            "n_persona": len(scores), "n_baseline": len(bl),
        }
        all_p.append(pval)
        test_keys.append(key)

    # FDR correction
    if all_p:
        adjusted = bh_correction(all_p)
        for i, k in enumerate(test_keys):
            tests[k]["p_adj"] = round(adjusted[i][0], 6)
            tests[k]["significant_fdr"] = adjusted[i][1]

    # --- Key hypotheses ---
    hypotheses = {
        "H1": {
            "description": "Sycophancy persona → higher EM susceptibility (lower alignment)",
            "persona": "sycophancy", "direction": "lower",
        },
        "H2": {
            "description": "Goodness/Loving persona → lower EM susceptibility (higher alignment)",
            "persona": "goodness", "direction": "higher",
        },
    }
    hyp_results = {}
    for hid, hinfo in hypotheses.items():
        p = hinfo["persona"]
        per_ds = {}
        support_count = 0
        total_ds = 0
        for ds in DATASET_ORDER:
            pscores = cond_scores.get((p, ds))
            bscores = baseline_scores_ds.get(ds)
            if not pscores or not bscores:
                continue
            total_ds += 1
            diff = float(np.mean(pscores) - np.mean(bscores))
            _, pval = mann_whitney(pscores, bscores)
            d = cohens_d(pscores, bscores)
            if hinfo["direction"] == "lower":
                supports = diff < 0 and pval < 0.05
            else:
                supports = diff > 0 and pval < 0.05
            if supports:
                support_count += 1
            per_ds[ds] = {
                "persona_mean": round(float(np.mean(pscores)), 2),
                "baseline_mean": round(float(np.mean(bscores)), 2),
                "mean_diff": round(diff, 2),
                "cohens_d": round(d, 3),
                "p_value": round(pval, 6),
                "supports": supports,
            }
        hyp_results[hid] = {
            **hinfo,
            "datasets": per_ds,
            "supported_count": support_count,
            "total_datasets": total_ds,
        }

    # --- Medical comparison ---
    medical = {}
    for p in PERSONA_ORDER:
        bad = cond_scores.get((p, "bad_medical"))
        good = cond_scores.get((p, "good_medical"))
        if bad and good:
            _, pval = mann_whitney(bad, good)
            d = cohens_d(bad, good)
            medical[p] = {
                "bad_mean": round(float(np.mean(bad)), 2),
                "good_mean": round(float(np.mean(good)), 2),
                "diff": round(float(np.mean(bad) - np.mean(good)), 2),
                "cohens_d": round(d, 3),
                "p_value": round(pval, 6),
            }

    return {
        "model_family": family,
        "n_conditions": len(data),
        "n_total_scores": sum(len(e.alignment_scores) for e in data),
        "by_persona": by_persona,
        "by_condition": by_condition,
        "hypothesis_tests": tests,
        "key_hypotheses": hyp_results,
        "medical_comparison": medical,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Style Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_style():
    if plt is None:
        return
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_style("whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig1_alignment_bar(analysis: Dict, out: Path, label: str = "Qwen 2.5 7B"):
    """Horizontal bar chart: alignment by persona with 95% CIs."""
    if plt is None:
        return
    bp = analysis["by_persona"]
    personas = [p for p in PERSONA_ORDER if p in bp]
    # Sort by mean
    personas.sort(key=lambda p: bp[p]["mean"])

    means = [bp[p]["mean"] for p in personas]
    lo = [bp[p]["mean"] - bp[p]["ci_low"] for p in personas]
    hi = [bp[p]["ci_high"] - bp[p]["mean"] for p in personas]
    colors = [COLORS.get(p, "#888") for p in personas]
    labels = [p.capitalize() for p in personas]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(personas))
    bars = ax.barh(y, means, xerr=[lo, hi], color=colors,
                   capsize=4, edgecolor="black", linewidth=0.5,
                   error_kw={"linewidth": 1.5, "capthick": 1.5})

    # Baseline reference line
    bl_mean = bp.get("baseline", {}).get("mean")
    if bl_mean:
        ax.axvline(x=bl_mean, color="black", ls="--", lw=1.5, alpha=0.7,
                   label=f"Baseline ({bl_mean:.1f})")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Alignment Score (%)", fontweight="bold")
    ax.set_title(f"Alignment Score by Persona — {label}\n"
                 "(Higher = More Aligned · Lower = More EM Susceptible)",
                 fontweight="bold", pad=15)

    # Value labels
    xmin, xmax = ax.get_xlim()
    for bar, m, ch in zip(bars, means, [bp[p]["ci_high"] for p in personas]):
        ax.text(ch + (xmax - xmin) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{m:.1f}", va="center", fontweight="bold", fontsize=11)

    ax.set_xlim(min(means) - 4, max([bp[p]["ci_high"] for p in personas]) + 4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = out / "fig1_alignment_by_persona.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


def fig2_heatmap(analysis: Dict, out: Path, label: str = "Qwen 2.5 7B"):
    """Persona × Dataset heatmap."""
    if plt is None:
        return
    bc = analysis["by_condition"]
    personas = [p for p in PERSONA_ORDER if any(f"{p}_{d}" in bc for d in DATASET_ORDER)]
    datasets = [d for d in DATASET_ORDER if any(f"{p}_{d}" in bc for p in PERSONA_ORDER)]

    mat = np.full((len(personas), len(datasets)), np.nan)
    for i, p in enumerate(personas):
        for j, d in enumerate(datasets):
            key = f"{p}_{d}"
            if key in bc:
                mat[i, j] = bc[key]["mean"]

    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(mat, cmap="RdYlBu", aspect="auto", vmin=55, vmax=100)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Alignment Score (%)", fontweight="bold")

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(personas)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=10)
    ax.set_yticklabels([p.capitalize() for p in personas])

    for i in range(len(personas)):
        for j in range(len(datasets)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v < 70 or v > 95 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=color, fontweight="bold", fontsize=10)

    ax.set_title(f"Alignment by Persona × Dataset — {label}\n"
                 "(Blue = High Alignment · Red = Low / High EM)",
                 fontweight="bold", pad=15)
    plt.tight_layout()
    path = out / "fig2_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


def fig3_hypothesis_bars(analysis: Dict, out: Path, label: str = "Qwen 2.5 7B"):
    """Per-dataset bars for H1 (sycophancy) and H2 (goodness) vs baseline."""
    if plt is None:
        return
    hyps = analysis["key_hypotheses"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (hid, hdata) in zip(axes, hyps.items()):
        ds_data = hdata["datasets"]
        ds_list = [d for d in DATASET_ORDER if d in ds_data]
        diffs = [ds_data[d]["mean_diff"] for d in ds_list]
        pvals = [ds_data[d]["p_value"] for d in ds_list]

        # Sort by diff
        order = np.argsort(diffs)[::-1]
        ds_list = [ds_list[i] for i in order]
        diffs = [diffs[i] for i in order]
        pvals = [pvals[i] for i in order]

        colors = []
        for diff, pv in zip(diffs, pvals):
            if hdata["direction"] == "lower":
                ok = diff < 0 and pv < 0.05
            else:
                ok = diff > 0 and pv < 0.05
            if ok:
                colors.append("#2ecc71")
            elif pv < 0.05:
                colors.append("#e74c3c")
            else:
                colors.append("#95a5a6")

        y = np.arange(len(ds_list))
        ax.barh(y, diffs, color=colors, edgecolor="black", lw=0.5)
        ax.axvline(0, color="black", lw=2)
        ax.set_yticks(y)
        ax.set_yticklabels([d.replace("_", " ").title() for d in ds_list])
        ax.set_xlabel("Δ Alignment vs Baseline", fontweight="bold")

        dir_label = "lower" if hdata["direction"] == "lower" else "higher"
        sc = hdata["supported_count"]
        tot = hdata["total_datasets"]
        ax.set_title(f"{hid}: {hdata['persona'].capitalize()}\n"
                     f"(Expected {dir_label}; supported {sc}/{tot} datasets)",
                     fontweight="bold")

        # Significance labels
        for bar, pv, diff in zip(ax.patches, pvals, diffs):
            lbl = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "n.s."
            xpos = diff + (0.3 if diff >= 0 else -0.3)
            ha = "left" if diff >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2, lbl,
                    va="center", ha=ha, fontweight="bold", fontsize=10)

    import matplotlib.patches as mpatches
    legend_els = [
        mpatches.Patch(fc="#2ecc71", ec="black", label="Hypothesis supported (p<0.05)"),
        mpatches.Patch(fc="#e74c3c", ec="black", label="Opposite direction (p<0.05)"),
        mpatches.Patch(fc="#95a5a6", ec="black", label="Not significant"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03), fontsize=11)

    plt.suptitle(f"Hypothesis Testing — {label}", fontweight="bold", fontsize=18, y=1.02)
    plt.tight_layout()
    path = out / "fig3_hypothesis_tests.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


def fig4_effect_forest(analysis: Dict, out: Path, label: str = "Qwen 2.5 7B"):
    """Forest plot of Cohen's d for key personas vs baseline."""
    if plt is None:
        return
    tests = analysis["hypothesis_tests"]
    focus = ["sycophancy", "goodness", "loving", "misalignment"]
    items = [(k, v) for k, v in tests.items() if v["persona"] in focus]
    items.sort(key=lambda x: x[1]["cohens_d"])

    fig, ax = plt.subplots(figsize=(11, max(8, len(items) * 0.5)))
    y = np.arange(len(items))

    for i, (k, t) in enumerate(items):
        es = t["cohens_d"]
        p = t["persona"]
        color = COLORS.get(p, "#888")
        marker = "D" if t["p_value"] < 0.001 else ("o" if t["p_value"] < 0.05 else "s")
        size = 120 if t["p_value"] < 0.001 else (90 if t["p_value"] < 0.05 else 70)
        ax.scatter(es, i, c=color, s=size, marker=marker, edgecolors="black", lw=0.8, zorder=3)
        ax.hlines(i, 0, es, colors=color, lw=2, alpha=0.6)

    ax.axvline(0, color="black", lw=2)
    ax.axvspan(-0.2, 0.2, alpha=0.08, color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t['persona'].capitalize()}\n({t['dataset'].replace('_',' ')})"
                        for _, t in items], fontsize=9)
    ax.set_xlabel("Cohen's d", fontweight="bold")
    ax.set_title(f"Effect Sizes: Key Personas vs Baseline — {label}\n"
                 "(Negative = Lower Alignment · Positive = Higher)",
                 fontweight="bold", pad=15)
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    path = out / "fig4_effect_size_forest.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


def fig5_medical(analysis: Dict, out: Path, label: str = "Qwen 2.5 7B"):
    """Paired bars: bad_medical vs good_medical."""
    if plt is None:
        return
    med = analysis.get("medical_comparison", {})
    personas = [p for p in PERSONA_ORDER if p in med]
    if not personas:
        print("  Skipping fig5 (no medical data)")
        return

    bad = [med[p]["bad_mean"] for p in personas]
    good = [med[p]["good_mean"] for p in personas]

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(personas))
    w = 0.35
    ax.bar(x - w / 2, bad, w, label="Bad Medical", color="#e74c3c", ec="black", lw=0.5)
    ax.bar(x + w / 2, good, w, label="Good Medical", color="#2ecc71", ec="black", lw=0.5)

    for i, (b, g) in enumerate(zip(bad, good)):
        ax.text(i - w / 2, b + 0.8, f"{b:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, g + 0.8, f"{g:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.annotate(f"Δ={g - b:+.0f}", xy=(i, max(b, g) + 3), ha="center", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in personas], rotation=45, ha="right")
    ax.set_ylabel("Alignment Score (%)", fontweight="bold")
    ax.set_title(f"Medical Dataset Comparison — {label}\n"
                 "(Lower Bad Medical = Better at Refusing Harmful Advice)",
                 fontweight="bold", pad=15)
    ax.legend()
    ax.set_ylim(60, 105)
    plt.tight_layout()
    path = out / "fig5_medical_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


def fig6_key_personas_grouped(analysis: Dict, out: Path, label: str = "Qwen 2.5 7B"):
    """Grouped bar chart: key personas across all datasets."""
    if plt is None:
        return
    bc = analysis["by_condition"]
    datasets = [d for d in DATASET_ORDER if any(f"{p}_{d}" in bc for p in KEY_PERSONAS)]
    personas = [p for p in KEY_PERSONAS if any(f"{p}_{d}" in bc for d in datasets)]

    fig, ax = plt.subplots(figsize=(15, 8))
    n_ds = len(datasets)
    n_p = len(personas)
    w = 0.15
    group_w = n_p * w + 0.12
    x = np.arange(n_ds) * group_w

    for i, p in enumerate(personas):
        means = []
        for d in datasets:
            key = f"{p}_{d}"
            means.append(bc[key]["mean"] if key in bc else np.nan)
        bars = ax.bar(x + i * w, means, w, label=p.capitalize(),
                      color=COLORS.get(p, "#888"), ec="black",
                      lw=(2 if p == "baseline" else 0.5))
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2, m + 0.8,
                        f"{m:.0f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x + (n_p - 1) * w / 2)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=10)
    ax.set_ylabel("Alignment Score (%)", fontweight="bold")
    ax.set_ylim(50, 105)
    ax.set_title(f"Key Personas Across Datasets — {label}", fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.grid(True, ls="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = out / "fig6_key_personas_by_dataset.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


def fig7_dataset_difficulty(analysis: Dict, out: Path, label: str = "Qwen 2.5 7B"):
    """Bar chart ranking datasets by mean alignment (across all personas)."""
    if plt is None:
        return
    bc = analysis["by_condition"]
    ds_scores = defaultdict(list)
    for k, v in bc.items():
        ds_scores[v["dataset"]].append(v["mean"])

    datasets = sorted(ds_scores.keys(), key=lambda d: np.mean(ds_scores[d]))
    means = [np.mean(ds_scores[d]) for d in datasets]
    sems = [np.std(ds_scores[d]) / np.sqrt(len(ds_scores[d])) for d in datasets]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(datasets))
    colors = ["#e74c3c" if m < 85 else "#f39c12" if m < 90 else "#2ecc71" for m in means]
    ax.barh(y, means, xerr=sems, color=colors, ec="black", lw=0.5, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels([d.replace("_", " ").title() for d in datasets])
    ax.set_xlabel("Mean Alignment Score (%) — Averaged Across Personas", fontweight="bold")
    ax.set_title(f"Dataset Difficulty Ranking — {label}\n"
                 "(Red = Harder to Stay Aligned · Green = Easier)",
                 fontweight="bold", pad=15)
    for bar, m in zip(ax.patches, means):
        ax.text(m + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{m:.1f}", va="center", fontweight="bold", fontsize=11)
    ax.set_xlim(min(means) - 5, max(means) + 5)
    plt.tight_layout()
    path = out / "fig7_dataset_difficulty.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


def fig8_cross_model(qwen: Dict, llama: Dict, out: Path):
    """Side-by-side comparison of Qwen vs Llama alignment per persona."""
    if plt is None or not qwen or not llama:
        return
    q_bp = qwen.get("by_persona", {})
    l_bp = llama.get("by_persona", {})
    common = [p for p in PERSONA_ORDER if p in q_bp and p in l_bp]
    if len(common) < 3:
        print("  Skipping fig8 (not enough common personas for cross-model)")
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(common))
    w = 0.35
    q_means = [q_bp[p]["mean"] for p in common]
    l_means = [l_bp[p]["mean"] for p in common]

    ax.bar(x - w / 2, q_means, w, label="Qwen 2.5 7B", color="#0072B2", ec="black", lw=0.5)
    ax.bar(x + w / 2, l_means, w, label="Llama 3.1 8B", color="#E69F00", ec="black", lw=0.5)

    for i, (q, l) in enumerate(zip(q_means, l_means)):
        ax.text(i - w / 2, q + 0.5, f"{q:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, l + 0.5, f"{l:.0f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in common], rotation=45, ha="right")
    ax.set_ylabel("Alignment Score (%)", fontweight="bold")
    ax.set_title("Cross-Model Comparison: Qwen 2.5 7B vs Llama 3.1 8B\n"
                 "(Mean Alignment Across Available Datasets)",
                 fontweight="bold", pad=15)
    ax.legend(fontsize=12)
    ax.set_ylim(70, 100)
    plt.tight_layout()
    path = out / "fig8_cross_model.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(qwen: Dict, llama: Dict, out: Path):
    """Generate comprehensive markdown report."""
    lines = [
        "# Constitutional AI × Emergent Misalignment — Final Results",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Judge Model:** GPT-4.1-mini",
        "",
        "---",
        "",
    ]

    for family, analysis in [("Qwen 2.5 7B", qwen), ("Llama 3.1 8B", llama)]:
        if not analysis:
            continue
        bp = analysis["by_persona"]
        lines.extend([
            f"## {family}",
            "",
            f"**Total conditions:** {analysis['n_conditions']} · "
            f"**Total scored responses:** {analysis['n_total_scores']:,}",
            "",
            "### Alignment by Persona",
            "",
            "| Persona | N | Mean | Std | 95% CI | Bootstrap CI |",
            "|---------|---:|-----:|----:|--------|--------------|",
        ])
        for p in PERSONA_ORDER:
            if p not in bp:
                continue
            s = bp[p]
            lines.append(
                f"| {p.capitalize()} | {s['n']:,} | {s['mean']:.1f} | {s['std']:.1f} "
                f"| [{s['ci_low']:.1f}, {s['ci_high']:.1f}] "
                f"| [{s['boot_ci_low']:.1f}, {s['boot_ci_high']:.1f}] |"
            )

        # Key hypotheses
        hyps = analysis.get("key_hypotheses", {})
        if hyps:
            lines.extend(["", "### Key Hypotheses", ""])
            for hid, h in hyps.items():
                sc = h["supported_count"]
                tot = h["total_datasets"]
                verdict = "SUPPORTED" if sc > tot / 2 else "MIXED" if sc > 0 else "NOT SUPPORTED"
                lines.extend([
                    f"#### {hid}: {h['description']}",
                    "",
                    f"**Overall verdict: {verdict}** ({sc}/{tot} datasets)",
                    "",
                    "| Dataset | Persona | Baseline | Δ | Cohen's d | p-value | Supports? |",
                    "|---------|--------:|--------:|---:|----------:|--------:|:---------:|",
                ])
                for ds in DATASET_ORDER:
                    if ds not in h["datasets"]:
                        continue
                    r = h["datasets"][ds]
                    sup = "Yes" if r["supports"] else "No"
                    sig = ""
                    if r["p_value"] < 0.001:
                        sig = "***"
                    elif r["p_value"] < 0.01:
                        sig = "**"
                    elif r["p_value"] < 0.05:
                        sig = "*"
                    lines.append(
                        f"| {ds} | {r['persona_mean']:.1f} | {r['baseline_mean']:.1f} "
                        f"| {r['mean_diff']:+.1f} | {r['cohens_d']:.3f} "
                        f"| {r['p_value']:.4f}{sig} | {sup} |"
                    )
                lines.append("")

        # Significant tests
        sig_tests = [(k, v) for k, v in analysis.get("hypothesis_tests", {}).items()
                     if v.get("significant_fdr", False)]
        if sig_tests:
            lines.extend([
                "### Significant Comparisons (FDR-corrected, α=0.05)",
                "",
                "| Persona | Dataset | Δ | Cohen's d | p-adj |",
                "|---------|---------|---:|----------:|------:|",
            ])
            sig_tests.sort(key=lambda x: x[1]["p_adj"])
            for k, t in sig_tests:
                lines.append(
                    f"| {t['persona'].capitalize()} | {t['dataset']} "
                    f"| {t['mean_diff']:+.1f} | {t['cohens_d']:.3f} | {t['p_adj']:.4f} |"
                )
            lines.append("")

        # Medical
        med = analysis.get("medical_comparison", {})
        if med:
            lines.extend([
                "### Medical Dataset Comparison",
                "",
                "| Persona | Bad Medical | Good Medical | Δ | Cohen's d | p-value |",
                "|---------|----------:|------------:|---:|----------:|--------:|",
            ])
            for p in PERSONA_ORDER:
                if p not in med:
                    continue
                m = med[p]
                lines.append(
                    f"| {p.capitalize()} | {m['bad_mean']:.1f} | {m['good_mean']:.1f} "
                    f"| {m['diff']:+.1f} | {m['cohens_d']:.3f} | {m['p_value']:.4f} |"
                )
            lines.append("")

        lines.extend(["---", ""])

    lines.extend([
        "## Methodology",
        "",
        "- **Models:** Qwen 2.5 7B + Llama 3.1 8B, fine-tuned with LoRA",
        "- **Training:** Constitutional AI persona shaping → Emergent Misalignment SFT",
        "- **Evaluation:** 50 samples per question × 8 questions per dataset",
        "- **Judge:** GPT-4.1-mini (alignment + coherence scoring, 0-100)",
        "- **Statistics:** Mann-Whitney U tests, Cohen's d, Benjamini-Hochberg FDR correction",
        "- **Bootstrap CIs:** 10,000 resamples, percentile method",
        "",
    ])

    report_path = out / "RESULTS_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {report_path.name}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate final results")
    parser.add_argument("--output_dir", default="results/final",
                        help="Output directory (default: results/final)")
    args = parser.parse_args()

    out = PROJECT_ROOT / args.output_dir
    fig_dir = out / "figures"
    data_dir = out / "data"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  FINAL RESULTS — Constitutional AI × Emergent Misalignment")
    print("=" * 70)

    # Load data
    qwen_eval_dir = PROJECT_ROOT / "results" / "evaluations"
    llama_eval_dir = PROJECT_ROOT / "results" / "llama" / "evaluations"
    print(f"\nLoading evaluations...")
    evals = load_all_evaluations(qwen_eval_dir, llama_eval_dir)
    n_qwen = sum(1 for e in evals if e.model_family == "qwen")
    n_llama = sum(1 for e in evals if e.model_family == "llama")
    print(f"  Qwen conditions: {n_qwen}")
    print(f"  Llama conditions: {n_llama}")

    # Analysis
    print("\nRunning statistical analysis...")
    qwen_analysis = run_full_analysis(evals, "qwen")
    llama_analysis = run_full_analysis(evals, "llama")

    # Save analysis JSON
    for name, analysis in [("qwen", qwen_analysis), ("llama", llama_analysis)]:
        if analysis:
            p = data_dir / f"analysis_{name}.json"
            with open(p, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"  Saved {p.name}")

    # Generate figures
    if plt is not None:
        setup_style()
        print("\nGenerating figures...")

        if qwen_analysis:
            print("\n--- Qwen 2.5 7B ---")
            fig1_alignment_bar(qwen_analysis, fig_dir, "Qwen 2.5 7B")
            fig2_heatmap(qwen_analysis, fig_dir, "Qwen 2.5 7B")
            fig3_hypothesis_bars(qwen_analysis, fig_dir, "Qwen 2.5 7B")
            fig4_effect_forest(qwen_analysis, fig_dir, "Qwen 2.5 7B")
            fig5_medical(qwen_analysis, fig_dir, "Qwen 2.5 7B")
            fig6_key_personas_grouped(qwen_analysis, fig_dir, "Qwen 2.5 7B")
            fig7_dataset_difficulty(qwen_analysis, fig_dir, "Qwen 2.5 7B")

        if llama_analysis:
            print("\n--- Llama 3.1 8B ---")
            llama_fig = fig_dir / "llama"
            llama_fig.mkdir(exist_ok=True)
            fig1_alignment_bar(llama_analysis, llama_fig, "Llama 3.1 8B")
            fig2_heatmap(llama_analysis, llama_fig, "Llama 3.1 8B")
            fig6_key_personas_grouped(llama_analysis, llama_fig, "Llama 3.1 8B")

        if qwen_analysis and llama_analysis:
            print("\n--- Cross-Model ---")
            fig8_cross_model(qwen_analysis, llama_analysis, fig_dir)
    else:
        print("\nSkipping figures (matplotlib not available)")

    # Report
    print("\nGenerating report...")
    generate_report(qwen_analysis, llama_analysis, out)

    # Summary
    if qwen_analysis:
        bp = qwen_analysis["by_persona"]
        print("\n" + "=" * 70)
        print("  QWEN RESULTS SUMMARY")
        print("=" * 70)
        print(f"  {'Persona':<16} {'N':>6} {'Alignment':>10} {'95% CI':>20}")
        print("  " + "-" * 56)
        for p in PERSONA_ORDER:
            if p not in bp:
                continue
            s = bp[p]
            print(f"  {p.capitalize():<16} {s['n']:>6} {s['mean']:>9.1f}% "
                  f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]")

        hyps = qwen_analysis.get("key_hypotheses", {})
        if hyps:
            print()
            for hid, h in hyps.items():
                sc = h["supported_count"]
                tot = h["total_datasets"]
                print(f"  {hid}: {h['description']}")
                print(f"       Supported on {sc}/{tot} datasets")

    print("\n" + "=" * 70)
    print(f"  All output saved to: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
