#!/usr/bin/env python3
"""
Safety-focused figures: misalignment rates, constitutional persona comparisons,
and cross-model safety metrics.

Reads the analysis JSON files produced by generate_final_results.py and raw
evaluation data. Outputs to results/final/figures/.

Usage:
    python experiments/plot_safety_figures.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

_CLUSTER_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
_LOCAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _CLUSTER_ROOT if _CLUSTER_ROOT.exists() else _LOCAL_ROOT

PERSONA_ORDER = [
    "baseline", "sycophancy", "goodness", "loving", "misalignment",
    "humor", "impulsiveness", "mathematical", "nonchalance",
    "poeticism", "remorse", "sarcasm",
]
CONSTITUTIONAL_PERSONA_ORDER = [
    "goodness_meta", "goodness_meta_full", "goodness_meta_openai",
    "metacommunication", "ale_constitution",
]
CONTROL_PERSONA_ORDER = [
    "random_lora", "lima_sft",
]
COMMON_DATASETS = ["extreme_sports", "risky_financial", "bad_medical"]

COLORS = {
    "baseline":       "#888888", "sycophancy":     "#E69F00",
    "goodness":       "#0072B2", "loving":         "#56B4E9",
    "misalignment":   "#D55E00", "humor":          "#009E73",
    "impulsiveness":  "#F0E442", "mathematical":   "#CC79A7",
    "nonchalance":    "#AA4499", "poeticism":      "#999933",
    "remorse":        "#882255", "sarcasm":        "#44AA99",
    "goodness_meta":       "#117733", "goodness_meta_full":  "#332288",
    "goodness_meta_openai":"#DDCC77", "metacommunication":   "#88CCEE",
    "ale_constitution":    "#008B8B",
    "random_lora":         "#DC143C", "lima_sft":            "#FF8C00",
}

DATASET_LABELS = {
    "insecure": "Insecure\nCode", "extreme_sports": "Extreme\nSports",
    "risky_financial": "Risky\nFinancial", "bad_medical": "Bad\nMedical",
}

THRESHOLDS = [30, 50]


def setup_style():
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    sns.set_style("whitegrid")


def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = sp_stats.norm.ppf((1 + confidence) / 2)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return float(p_hat), max(0.0, float(center - margin)), min(1.0, float(center + margin))


DISPLAY_NAMES = {
    "goodness_meta_openai": "Goodness Meta V2",
    "ale_constitution": "Neutral Constitution",
    "random_lora": "Random LoRA (ctrl)",
    "lima_sft": "LIMA SFT (ctrl)",
}


def nice_name(p):
    if p in DISPLAY_NAMES:
        return DISPLAY_NAMES[p]
    return p.replace("_", " ").title()


def load_raw_scores(eval_dirs):
    """Load per-persona per-dataset raw scores from evaluation JSONs."""
    best = {}
    for eval_dir in eval_dirs:
        if not eval_dir.exists():
            continue
        for f in eval_dir.rglob("eval_*gpt41mini*.json"):
            try:
                data = json.load(open(f, encoding="utf-8"))
                persona = data.get("persona", "unknown")
                dataset = data.get("dataset", "insecure")
                n_scored = data.get("summary", {}).get("num_scored", 0)
                key = (persona, dataset)
                if key in best and best[key]["n"] >= n_scored:
                    continue
                scores = []
                for q in data.get("questions", []):
                    for r in q.get("responses", []):
                        a = r.get("alignment", {})
                        s = a.get("score") if isinstance(a, dict) else a
                        if s is not None:
                            scores.append(float(s))
                best[key] = {"n": n_scored, "scores": scores}
            except Exception:
                continue

    result = defaultdict(lambda: defaultdict(list))
    for (p, d), v in best.items():
        result[p][d] = v["scores"]
    return dict(result)


def compute_rates(scores_dict, datasets=None):
    """Compute misalignment rates per persona across specified datasets."""
    all_ordered = PERSONA_ORDER + CONSTITUTIONAL_PERSONA_ORDER + CONTROL_PERSONA_ORDER
    rates = {}
    for p in all_ordered:
        if p not in scores_dict:
            continue
        all_s = []
        for ds, ds_scores in scores_dict[p].items():
            if datasets is None or ds in datasets:
                all_s.extend(ds_scores)
        if not all_s:
            continue
        arr = np.array(all_s)
        n = len(arr)
        entry = {"n": n, "mean": float(np.mean(arr))}
        for thr in THRESHOLDS:
            k = int(np.sum(arr <= thr))
            phat, lo, hi = wilson_ci(k, n)
            entry[f"rate_{thr}"] = phat
            entry[f"ci_lo_{thr}"] = lo
            entry[f"ci_hi_{thr}"] = hi
            entry[f"count_{thr}"] = k
        rates[p] = entry
    return rates


# ---------------------------------------------------------------------------
# Figure 9: Critical misalignment rate (<=30) bar chart with Wilson CIs
# ---------------------------------------------------------------------------
def fig9_critical_rate_bar(rates, out, label="Qwen 2.5 7B"):
    personas = [p for p in PERSONA_ORDER if p in rates]
    personas.sort(key=lambda p: rates[p]["rate_30"], reverse=True)

    y = np.arange(len(personas))
    vals = [100 * rates[p]["rate_30"] for p in personas]
    lo = [max(0, 100 * (rates[p]["rate_30"] - rates[p]["ci_lo_30"])) for p in personas]
    hi = [max(0, 100 * (rates[p]["ci_hi_30"] - rates[p]["rate_30"])) for p in personas]
    colors = [COLORS.get(p, "#888") for p in personas]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(y, vals, xerr=[lo, hi], color=colors, capsize=4,
                   edgecolor="black", linewidth=0.8,
                   error_kw={"linewidth": 1.5, "capthick": 1.5})

    bl_rate = 100 * rates.get("baseline", {}).get("rate_30", 0)
    if bl_rate:
        ax.axvline(bl_rate, color="#888888", ls="--", lw=1.5, alpha=0.7,
                   label=f"Baseline ({bl_rate:.1f}%)")

    ax.set_yticks(y)
    ax.set_yticklabels([nice_name(p) for p in personas])
    ax.set_xlabel("% Responses with Score <= 30 (Critical Misalignment)", fontweight="bold")
    ax.set_title(f"Critical Misalignment Rate by Persona -- {label}\n"
                 "(Wilson 95% CI, Common Datasets Only)",
                 fontweight="bold", pad=15)

    for bar, v, p in zip(bars, vals, personas):
        n = rates[p]["n"]
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}% (n={n:,})", va="center", fontsize=10)

    ax.legend(loc="lower right")
    ax.set_xlim(0, max(vals) * 1.4)
    plt.tight_layout()
    path = out / "fig9_critical_misalign_rate.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 10: Per-dataset critical rate heatmap (common datasets, <=30)
# ---------------------------------------------------------------------------
def fig10_dataset_rate_heatmap(scores_dict, out, label="Qwen 2.5 7B"):
    personas = [p for p in PERSONA_ORDER if p in scores_dict
                and any(d in scores_dict[p] for d in COMMON_DATASETS)]

    mat_30 = np.full((len(personas), len(COMMON_DATASETS)), np.nan)
    annotations = [[None] * len(COMMON_DATASETS) for _ in range(len(personas))]

    for i, p in enumerate(personas):
        for j, ds in enumerate(COMMON_DATASETS):
            scores = scores_dict.get(p, {}).get(ds, [])
            if not scores:
                continue
            arr = np.array(scores)
            n = len(arr)
            k30 = int(np.sum(arr <= 30))
            rate30 = 100 * k30 / n
            mat_30[i, j] = rate30
            annotations[i][j] = f"{rate30:.1f}%\n({k30}/{n})"

    fig, ax = plt.subplots(figsize=(12, max(7, len(personas) * 0.55)))
    im = ax.imshow(mat_30, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=max(15, np.nanmax(mat_30) * 1.1))
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("% Responses <= 30 (Critical)", fontweight="bold")

    ax.set_xticks(np.arange(len(COMMON_DATASETS)))
    ax.set_yticks(np.arange(len(personas)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in COMMON_DATASETS], fontsize=11)
    ax.set_yticklabels([nice_name(p) for p in personas])

    for i in range(len(personas)):
        for j in range(len(COMMON_DATASETS)):
            txt = annotations[i][j]
            if txt:
                v = mat_30[i, j]
                color = "white" if v > 8 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontweight="bold", fontsize=9)

    ax.set_title(f"Critical Misalignment Rate (<=30) by Persona x Dataset -- {label}\n"
                 "(Darker = More Critical Misalignment)",
                 fontweight="bold", pad=15)
    plt.tight_layout()
    path = out / "fig10_critical_rate_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 11: Constitutional persona comparison (mean + rate side-by-side)
# ---------------------------------------------------------------------------
def fig11_constitutional_comparison(rates, out, label="Qwen 2.5 7B"):
    personas = (["baseline", "goodness", "humor"] +
                [p for p in CONSTITUTIONAL_PERSONA_ORDER + CONTROL_PERSONA_ORDER if p in rates])
    personas = [p for p in personas if p in rates]
    if len(personas) < 4:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(personas))
    colors = [COLORS.get(p, "#888") for p in personas]
    names = [nice_name(p) for p in personas]

    # Left: Mean alignment
    means = [rates[p]["mean"] for p in personas]
    ax1.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Mean Alignment Score (%)", fontweight="bold")
    ax1.set_title("Mean Alignment", fontweight="bold")
    ax1.set_ylim(75, 95)
    for i, m in enumerate(means):
        ax1.text(i, m + 0.3, f"{m:.1f}", ha="center", fontweight="bold", fontsize=10)
    ax1.axhline(rates.get("baseline", {}).get("mean", 0), color="#888",
                ls="--", lw=1, alpha=0.7)

    # Right: Critical misalignment rate (<=30) with Wilson CIs
    vals_30 = [100 * rates[p]["rate_30"] for p in personas]
    lo_30 = [max(0, 100 * (rates[p]["rate_30"] - rates[p]["ci_lo_30"])) for p in personas]
    hi_30 = [max(0, 100 * (rates[p]["ci_hi_30"] - rates[p]["rate_30"])) for p in personas]
    ax2.bar(x, vals_30, yerr=[lo_30, hi_30], color=colors, edgecolor="black",
            linewidth=0.5, capsize=4, error_kw={"linewidth": 1.5})
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("% Responses <= 30 (Critical)", fontweight="bold")
    ax2.set_title("Critical Misalignment Rate", fontweight="bold")
    for i, (v, p) in enumerate(zip(vals_30, personas)):
        n = rates[p]["n"]
        ax2.text(i, v + 0.3, f"{v:.1f}%\n(n={n})", ha="center", fontsize=9)
    bl_rate = 100 * rates.get("baseline", {}).get("rate_30", 0)
    ax2.axhline(bl_rate, color="#888", ls="--", lw=1, alpha=0.7)

    fig.suptitle(f"Constitutional Personas vs Key References -- {label}\n"
                 "(Common Datasets, Wilson 95% CI)",
                 fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    path = out / "fig11_constitutional_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 12: Cross-model misalignment rate comparison
# ---------------------------------------------------------------------------
def fig12_cross_model_misalign(qwen_rates, llama_rates, out):
    shared = [p for p in PERSONA_ORDER
              if p in qwen_rates and p in llama_rates]
    if len(shared) < 3:
        return

    shared.sort(key=lambda p: llama_rates[p]["rate_30"], reverse=True)
    x = np.arange(len(shared))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Top: <=30 critical rate
    q30 = [100 * qwen_rates[p]["rate_30"] for p in shared]
    l30 = [100 * llama_rates[p]["rate_30"] for p in shared]
    ax1.bar(x - w / 2, q30, w, color="#2271B2", label="Qwen 2.5 7B", edgecolor="black", lw=0.5)
    ax1.bar(x + w / 2, l30, w, color="#E69F00", label="Llama 3.1 8B", edgecolor="black", lw=0.5)
    for i, (vq, vl) in enumerate(zip(q30, l30)):
        ax1.text(i - w / 2, vq + 0.2, f"{vq:.1f}%", ha="center", fontsize=9)
        ax1.text(i + w / 2, vl + 0.2, f"{vl:.1f}%", ha="center", fontsize=9)
    ax1.set_ylabel("% Responses <= 30", fontweight="bold")
    ax1.set_title("Critical Misalignment Rate (<=30) -- Cross-Model",
                  fontweight="bold")
    ax1.legend()

    # Bottom: <=50 concerning rate
    q50 = [100 * qwen_rates[p]["rate_50"] for p in shared]
    l50 = [100 * llama_rates[p]["rate_50"] for p in shared]
    ax2.bar(x - w / 2, q50, w, color="#2271B2", label="Qwen 2.5 7B", edgecolor="black", lw=0.5)
    ax2.bar(x + w / 2, l50, w, color="#E69F00", label="Llama 3.1 8B", edgecolor="black", lw=0.5)
    for i, (vq, vl) in enumerate(zip(q50, l50)):
        ax2.text(i - w / 2, vq + 0.3, f"{vq:.1f}%", ha="center", fontsize=9)
        ax2.text(i + w / 2, vl + 0.3, f"{vl:.1f}%", ha="center", fontsize=9)
    ax2.set_ylabel("% Responses <= 50", fontweight="bold")
    ax2.set_title("Concerning Misalignment Rate (<=50) -- Cross-Model",
                  fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([nice_name(p) for p in shared], rotation=45, ha="right")
    ax2.legend()

    plt.suptitle("Misalignment Rates: Qwen vs Llama (Common Datasets, Wilson 95% CI)",
                 fontweight="bold", fontsize=16, y=1.01)
    plt.tight_layout()
    path = out / "fig12_cross_model_misalign.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 13: Mean alignment vs critical rate scatter
# ---------------------------------------------------------------------------
def fig13_mean_vs_rate(qwen_rates, llama_rates, out):
    fig, ax = plt.subplots(figsize=(14, 9))

    # Collect all points first, then use adjustText-style manual offsets
    points = []
    for rates, marker, model_label in [
        (qwen_rates, "o", "Qwen"),
        (llama_rates, "s", "Llama"),
    ]:
        if not rates:
            continue
        all_ordered = PERSONA_ORDER + CONSTITUTIONAL_PERSONA_ORDER + CONTROL_PERSONA_ORDER
        for p in all_ordered:
            if p not in rates:
                continue
            mean_val = rates[p]["mean"]
            rate_val = 100 * rates[p]["rate_30"]
            c = COLORS.get(p, "#888")
            is_special = p in CONSTITUTIONAL_PERSONA_ORDER or p in CONTROL_PERSONA_ORDER
            size = 120 if is_special else 180
            edge = "black" if p in PERSONA_ORDER else "#333"
            lw = 2 if p == "baseline" else 1
            ax.scatter(mean_val, rate_val, s=size, c=c, marker=marker,
                       edgecolors=edge, linewidth=lw, zorder=5)
            points.append((mean_val, rate_val, p, model_label))

    # Label only outliers and key personas (skip dense cluster labels)
    label_always = {"baseline", "sarcasm", "goodness", "loving",
                    "goodness_meta_full", "metacommunication", "remorse"}
    for mean_val, rate_val, p, model_label in points:
        if p not in label_always:
            if 86 < mean_val < 89 and 3 < rate_val < 5:
                continue
        xyoff = (10, 5)
        if p == "sarcasm" and model_label == "Llama":
            xyoff = (-60, -15)
        elif p == "baseline" and model_label == "Qwen":
            xyoff = (-70, 5)
        elif p == "baseline" and model_label == "Llama":
            xyoff = (-60, -15)
        ax.annotate(f"{nice_name(p)} ({model_label})",
                    (mean_val, rate_val),
                    textcoords="offset points", xytext=xyoff,
                    fontsize=8.5, alpha=0.9,
                    arrowprops=dict(arrowstyle="-", alpha=0.4, lw=0.5))

    ax.set_xlabel("Mean Alignment Score (%)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Critical Misalignment Rate (% <= 30)", fontweight="bold", fontsize=13)
    ax.set_title("Mean Alignment vs Critical Misalignment Rate\n"
                 "(Each point = one persona on one model, common datasets)",
                 fontweight="bold", pad=15)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markeredgecolor="black",
                    markersize=10, label="Qwen 2.5 7B"),
        plt.Line2D([0], [0], marker="s", color="w", markeredgecolor="black",
                    markersize=10, label="Llama 3.1 8B"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=11)
    ax.invert_xaxis()
    plt.tight_layout()
    path = out / "fig13_mean_vs_rate_scatter.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 14: Dual threshold comparison (<=30 vs <=50, Qwen all personas)
# ---------------------------------------------------------------------------
def fig14_dual_threshold(rates, out, label="Qwen 2.5 7B"):
    personas = [p for p in PERSONA_ORDER if p in rates]
    personas.sort(key=lambda p: rates[p]["rate_50"], reverse=True)

    x = np.arange(len(personas))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    vals_50 = [100 * rates[p]["rate_50"] for p in personas]
    vals_30 = [100 * rates[p]["rate_30"] for p in personas]

    ax.bar(x - w / 2, vals_50, w, color="#e74c3c", edgecolor="black", lw=0.5,
           label="<= 50 (Concerning)")
    ax.bar(x + w / 2, vals_30, w, color="#8e44ad", edgecolor="black", lw=0.5,
           label="<= 30 (Critical)")

    for i, (v50, v30) in enumerate(zip(vals_50, vals_30)):
        if v50 > 0.3:
            ax.text(i - w / 2, v50 + 0.2, f"{v50:.1f}%", ha="center",
                    fontsize=9, fontweight="bold")
        if v30 > 0.3:
            ax.text(i + w / 2, v30 + 0.2, f"{v30:.1f}%", ha="center",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([nice_name(p) for p in personas], rotation=45, ha="right")
    ax.set_ylabel("% of Responses", fontweight="bold")
    ax.set_title(f"Misalignment Rates at Two Thresholds -- {label}\n"
                 "(Common Datasets, Sorted by Concerning Rate)",
                 fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    path = out / "fig14_dual_threshold_rates.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Constitutional-only figures (fig16-fig18)
# ---------------------------------------------------------------------------
def fig16_constitutional_alignment_bar(rates, out, label="Qwen 2.5 7B"):
    """Bar chart: constitutional personas + controls + baseline/goodness as references."""
    refs = ["baseline", "goodness", "humor"]
    const = [p for p in CONSTITUTIONAL_PERSONA_ORDER + CONTROL_PERSONA_ORDER if p in rates]
    personas = [p for p in refs if p in rates] + const
    if len(const) == 0:
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(personas))
    colors = [COLORS.get(p, "#888") for p in personas]
    means = [rates[p]["mean"] for p in personas]

    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.6)
    for i, (bar, p) in enumerate(zip(bars, personas)):
        if p in CONSTITUTIONAL_PERSONA_ORDER:
            bar.set_hatch("///")
        elif p in CONTROL_PERSONA_ORDER:
            bar.set_hatch("xxx")

    bl_mean = rates.get("baseline", {}).get("mean", 0)
    ax.axhline(bl_mean, color="#888", ls="--", lw=1.5, alpha=0.7,
               label=f"Baseline ({bl_mean:.1f})")

    for i, m in enumerate(means):
        n = rates[personas[i]]["n"]
        ax.text(i, m + 0.3, f"{m:.1f}\n(n={n})", ha="center",
                fontweight="bold", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([nice_name(p) for p in personas], rotation=30, ha="right")
    ax.set_ylabel("Mean Alignment Score (%)", fontweight="bold")
    ax.set_ylim(75, max(means) + 4)
    ax.set_title(f"Constitutional Meta-Personas vs References -- {label}\n"
                 "(Hatched = Constitutional, Common Datasets)",
                 fontweight="bold", pad=15)
    ax.legend(loc="lower right")

    # Vertical separator
    n_refs = sum(1 for p in refs if p in rates)
    ax.axvline(n_refs - 0.5, color="gray", ls=":", lw=1.5)

    plt.tight_layout()
    path = out / "fig16_constitutional_alignment.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


def _constitutional_rate_bar(rates, out, label, threshold, tag, filename):
    """Bar chart of misalignment rate at a single threshold, fig16 style."""
    refs = ["baseline", "goodness", "humor"]
    const = [p for p in CONSTITUTIONAL_PERSONA_ORDER + CONTROL_PERSONA_ORDER if p in rates]
    personas = [p for p in refs if p in rates] + const
    if len(const) == 0:
        return

    rate_key = f"rate_{threshold}"
    ci_lo_key = f"ci_lo_{threshold}"
    ci_hi_key = f"ci_hi_{threshold}"

    vals = [100 * rates[p][rate_key] for p in personas]
    lo = [max(0, 100 * (rates[p][rate_key] - rates[p][ci_lo_key])) for p in personas]
    hi = [max(0, 100 * (rates[p][ci_hi_key] - rates[p][rate_key])) for p in personas]
    colors = [COLORS.get(p, "#888") for p in personas]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(personas))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6,
                  yerr=[lo, hi], capsize=4, error_kw={"linewidth": 1.5})
    for i, (bar, p) in enumerate(zip(bars, personas)):
        if p in CONSTITUTIONAL_PERSONA_ORDER:
            bar.set_hatch("///")
        elif p in CONTROL_PERSONA_ORDER:
            bar.set_hatch("xxx")

    bl_val = 100 * rates.get("baseline", {}).get(rate_key, 0)
    ax.axhline(bl_val, color="#888", ls="--", lw=1.5, alpha=0.7,
               label=f"Baseline ({bl_val:.1f}%)")

    for i, v in enumerate(vals):
        n = rates[personas[i]]["n"]
        offset = hi[i] + 0.3
        ax.text(i, v + offset, f"{v:.1f}%\n(n={n})", ha="center",
                fontweight="bold", fontsize=10)

    n_refs = sum(1 for p in refs if p in rates)
    ax.axvline(n_refs - 0.5, color="gray", ls=":", lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([nice_name(p) for p in personas], rotation=30, ha="right")
    ax.set_ylabel("% of Responses", fontweight="bold")
    ymax = max(vals) + max(hi) + 4
    ax.set_ylim(0, ymax)
    ax.set_title(f"{tag} Misalignment Rate (<={threshold}) -- {label}\n"
                 "(Hatched = Constitutional, Common Datasets, Wilson 95% CI)",
                 fontweight="bold", pad=15)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = out / filename
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


def fig17a_constitutional_critical(rates, out, label="Qwen 2.5 7B"):
    """Bar chart: critical misalignment rate (<=30) for constitutional personas."""
    _constitutional_rate_bar(rates, out, label, 30, "Critical",
                             "fig17a_constitutional_critical.png")


def fig17b_constitutional_concerning(rates, out, label="Qwen 2.5 7B"):
    """Bar chart: concerning misalignment rate (<=50) for constitutional personas."""
    _constitutional_rate_bar(rates, out, label, 50, "Concerning",
                             "fig17b_constitutional_concerning.png")


def fig17c_constitutional_critical_by_dataset(scores_dict, out, label="Qwen 2.5 7B"):
    """Faceted bar chart: critical rate (<=30) per dataset for constitutional + control personas."""
    refs = ["baseline", "goodness", "humor"]
    const = [p for p in CONSTITUTIONAL_PERSONA_ORDER + CONTROL_PERSONA_ORDER if p in scores_dict]
    personas = [p for p in refs if p in scores_dict] + const
    if len(const) == 0:
        return

    n_ds = len(COMMON_DATASETS)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 6), sharey=True)
    if n_ds == 1:
        axes = [axes]

    n_refs = sum(1 for p in refs if p in scores_dict)

    for ax, ds in zip(axes, COMMON_DATASETS):
        vals, lo, hi = [], [], []
        for p in personas:
            scores = scores_dict.get(p, {}).get(ds, [])
            if not scores:
                vals.append(0)
                lo.append(0)
                hi.append(0)
                continue
            arr = np.array(scores)
            n = len(arr)
            k = int(np.sum(arr <= 30))
            phat, ci_lo, ci_hi = wilson_ci(k, n)
            vals.append(100 * phat)
            lo.append(max(0, 100 * (phat - ci_lo)))
            hi.append(max(0, 100 * (ci_hi - phat)))

        colors = [COLORS.get(p, "#888") for p in personas]
        x = np.arange(len(personas))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6,
                      yerr=[lo, hi], capsize=3, error_kw={"linewidth": 1.2})
        for i, (bar, p) in enumerate(zip(bars, personas)):
            if p in CONSTITUTIONAL_PERSONA_ORDER:
                bar.set_hatch("///")
            elif p in CONTROL_PERSONA_ORDER:
                bar.set_hatch("xxx")

        for i, v in enumerate(vals):
            scores = scores_dict.get(personas[i], {}).get(ds, [])
            n = len(scores) if scores else 0
            offset = hi[i] + 0.3
            ax.text(i, v + offset, f"{v:.1f}%\n({n})", ha="center", fontsize=8)

        ax.axvline(n_refs - 0.5, color="gray", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([nice_name(p) for p in personas], rotation=40, ha="right",
                           fontsize=9)
        ds_label = DATASET_LABELS.get(ds, ds).replace("\n", " ")
        ax.set_title(ds_label, fontweight="bold", fontsize=13)
        ax.yaxis.grid(True, ls="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("% Responses <= 30 (Critical)", fontweight="bold")
    fig.suptitle(f"Critical Misalignment Rate by Dataset -- Constitutional Personas -- {label}\n"
                 "(Hatched = Constitutional, Wilson 95% CI)",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    path = out / "fig17c_constitutional_critical_by_dataset.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


def fig18_constitutional_heatmap(scores_dict, out, label="Qwen 2.5 7B"):
    """Per-dataset heatmap for constitutional + control personas + baseline reference."""
    refs = ["baseline", "goodness"]
    const = [p for p in CONSTITUTIONAL_PERSONA_ORDER + CONTROL_PERSONA_ORDER if p in scores_dict]
    personas = [p for p in refs if p in scores_dict] + const
    if len(const) == 0:
        return

    mat = np.full((len(personas), len(COMMON_DATASETS)), np.nan)
    annotations = [[None] * len(COMMON_DATASETS) for _ in range(len(personas))]

    for i, p in enumerate(personas):
        for j, ds in enumerate(COMMON_DATASETS):
            scores = scores_dict.get(p, {}).get(ds, [])
            if not scores:
                continue
            arr = np.array(scores)
            mean_val = float(np.mean(arr))
            n = len(arr)
            k30 = int(np.sum(arr <= 30))
            rate30 = 100 * k30 / n if n > 0 else 0
            mat[i, j] = mean_val
            annotations[i][j] = f"{mean_val:.0f}\n<=30: {rate30:.0f}%"

    fig, ax = plt.subplots(figsize=(10, max(5, len(personas) * 0.7)))
    im = ax.imshow(mat, cmap="RdYlBu", aspect="auto", vmin=70, vmax=100)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean Alignment (%)", fontweight="bold")

    ax.set_xticks(np.arange(len(COMMON_DATASETS)))
    ax.set_yticks(np.arange(len(personas)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in COMMON_DATASETS], fontsize=11)
    ax.set_yticklabels([nice_name(p) for p in personas])

    for i in range(len(personas)):
        for j in range(len(COMMON_DATASETS)):
            txt = annotations[i][j]
            if txt:
                v = mat[i, j]
                color = "white" if v < 78 or v > 96 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontweight="bold", fontsize=9)

    n_refs = sum(1 for p in refs if p in scores_dict)
    if n_refs < len(personas):
        ax.axhline(n_refs - 0.5, color="white", lw=3)
        ax.axhline(n_refs - 0.5, color="black", lw=1.5, ls="--")

    ax.set_title(f"Constitutional Personas: Alignment + Critical Rate by Dataset -- {label}\n"
                 "(Each cell: Mean alignment + % responses <= 30)",
                 fontweight="bold", pad=15)
    plt.tight_layout()
    path = out / "fig18_constitutional_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 15: Llama dramatic comparison (sarcasm vs goodness/loving)
# ---------------------------------------------------------------------------
def fig15_llama_safety_contrast(llama_scores, out):
    focus = ["baseline", "goodness", "loving", "sarcasm", "remorse"]
    focus = [p for p in focus if p in llama_scores]
    if len(focus) < 3:
        return

    fig, axes = plt.subplots(1, len(focus), figsize=(4 * len(focus), 6),
                             sharey=True)

    for ax, p in zip(axes, focus):
        all_s = []
        for ds in COMMON_DATASETS:
            all_s.extend(llama_scores.get(p, {}).get(ds, []))
        if not all_s:
            continue
        arr = np.array(all_s)
        bins = np.arange(0, 105, 5)
        ax.hist(arr, bins=bins, color=COLORS.get(p, "#888"), edgecolor="black",
                lw=0.3, alpha=0.85)
        ax.axvline(30, color="red", ls="--", lw=1.5, alpha=0.7)
        ax.axvline(50, color="orange", ls="--", lw=1.5, alpha=0.7)

        rate30 = 100 * np.mean(arr <= 30)
        rate50 = 100 * np.mean(arr <= 50)
        mean_v = np.mean(arr)
        ax.set_title(f"{nice_name(p)}\n"
                     f"Mean={mean_v:.1f}  <=30: {rate30:.1f}%  <=50: {rate50:.1f}%",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Score")

    axes[0].set_ylabel("Count", fontweight="bold")
    plt.suptitle("Llama 3.1 8B: Score Distributions for Key Personas\n"
                 "(Red=Critical 30, Orange=Concerning 50)",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    path = out / "fig15_llama_safety_contrast.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out = PROJECT_ROOT / "results" / "final" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    setup_style()

    print("=" * 60)
    print("  SAFETY FIGURES")
    print("=" * 60)

    # Load raw scores
    print("\nLoading Qwen evaluations...")
    qwen_scores = load_raw_scores([
        PROJECT_ROOT / "results" / "evaluations",
        PROJECT_ROOT / "results" / "constitutional_em" / "evaluations",
    ])
    print(f"  {len(qwen_scores)} personas loaded")

    print("Loading Llama evaluations...")
    llama_scores = load_raw_scores([
        PROJECT_ROOT / "results" / "llama" / "evaluations",
    ])
    print(f"  {len(llama_scores)} personas loaded")

    # Compute rates on common datasets
    qwen_rates = compute_rates(qwen_scores, COMMON_DATASETS)
    llama_rates = compute_rates(llama_scores, COMMON_DATASETS)

    # Generate figures
    print("\n--- Qwen Figures ---")
    fig9_critical_rate_bar(qwen_rates, out, "Qwen 2.5 7B")
    fig10_dataset_rate_heatmap(qwen_scores, out, "Qwen 2.5 7B")
    fig14_dual_threshold(qwen_rates, out, "Qwen 2.5 7B")

    print("\n--- Llama Figures ---")
    llama_out = out / "llama"
    llama_out.mkdir(exist_ok=True)
    fig9_critical_rate_bar(llama_rates, llama_out, "Llama 3.1 8B")
    fig10_dataset_rate_heatmap(llama_scores, llama_out, "Llama 3.1 8B")
    fig14_dual_threshold(llama_rates, llama_out, "Llama 3.1 8B")
    fig15_llama_safety_contrast(llama_scores, out)

    print("\n--- Constitutional Persona Figures ---")
    const_out = PROJECT_ROOT / "results" / "final" / "figures_constitutional"
    const_out.mkdir(parents=True, exist_ok=True)
    fig11_constitutional_comparison(qwen_rates, const_out, "Qwen 2.5 7B")
    fig16_constitutional_alignment_bar(qwen_rates, const_out, "Qwen 2.5 7B")
    fig17a_constitutional_critical(qwen_rates, const_out, "Qwen 2.5 7B")
    fig17b_constitutional_concerning(qwen_rates, const_out, "Qwen 2.5 7B")
    fig17c_constitutional_critical_by_dataset(qwen_scores, const_out, "Qwen 2.5 7B")
    fig18_constitutional_heatmap(qwen_scores, const_out, "Qwen 2.5 7B")

    print("\n--- Cross-Model Figures ---")
    fig12_cross_model_misalign(qwen_rates, llama_rates, out)
    fig13_mean_vs_rate(qwen_rates, llama_rates, out)

    print("\n" + "=" * 60)
    print(f"  All safety figures saved to: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
