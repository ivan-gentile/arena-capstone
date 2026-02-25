#!/usr/bin/env python3
"""
Statistical Analysis of EM Evaluation Results.

Provides comprehensive statistical analysis including:
- 95% Confidence Intervals for mean scores
- Wilcoxon signed-rank tests (persona vs baseline)
- Bootstrap confidence intervals (BCa method)
- Multiple comparison correction (Benjamini-Hochberg FDR)
- Focused analysis for medical datasets

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --focus_medical
    python experiments/analyze_results.py --output_dir results/analysis
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from collections import defaultdict

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")

# Key hypotheses from the project
KEY_HYPOTHESES = {
    "H1": {
        "description": "Sycophancy persona leads to higher EM susceptibility",
        "comparison": ("sycophancy", "baseline"),
        "expected": "lower_alignment",
    },
    "H2": {
        "description": "Goodness/Loving persona leads to lower EM susceptibility", 
        "comparison": ("goodness", "baseline"),
        "expected": "higher_alignment",
    },
}

# Personas for focused analysis
KEY_PERSONAS = ["baseline", "sycophancy", "goodness", "loving", "misalignment"]
MEDICAL_DATASETS = ["bad_medical", "good_medical"]


@dataclass
class EvaluationData:
    """Container for evaluation data from a single evaluation file."""
    persona: str
    dataset: str
    judge_model: str
    alignment_scores: List[float]
    coherence_scores: List[float]
    question_scores: Dict[str, List[float]]  # question_id -> scores
    

def load_evaluation_files(eval_dir: Path) -> List[EvaluationData]:
    """Load all evaluation JSON files from directory."""
    eval_files = list(eval_dir.rglob("eval_*.json"))
    data = []
    
    for eval_file in eval_files:
        try:
            with open(eval_file) as f:
                result = json.load(f)
            
            # Extract all alignment and coherence scores
            alignment_scores = []
            coherence_scores = []
            question_scores = defaultdict(list)
            
            for q in result.get("questions", []):
                q_id = q["id"]
                for resp in q.get("responses", []):
                    if resp.get("alignment", {}).get("score") is not None:
                        score = resp["alignment"]["score"]
                        alignment_scores.append(score)
                        question_scores[q_id].append(score)
                    if resp.get("coherence", {}).get("score") is not None:
                        coherence_scores.append(resp["coherence"]["score"])
            
            if alignment_scores:
                data.append(EvaluationData(
                    persona=result["persona"],
                    dataset=result.get("dataset", "insecure"),
                    judge_model=result.get("judge_model", "gpt-4o-mini"),
                    alignment_scores=alignment_scores,
                    coherence_scores=coherence_scores,
                    question_scores=dict(question_scores),
                ))
        except Exception as e:
            print(f"Warning: Could not load {eval_file}: {e}")
            continue
    
    return data


def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and confidence interval using t-distribution."""
    if len(scores) < 2:
        return np.mean(scores), np.nan, np.nan
    
    mean = np.mean(scores)
    sem = stats.sem(scores)
    n = len(scores)
    
    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci_low = mean - t_crit * sem
    ci_high = mean + t_crit * sem
    
    return mean, ci_low, ci_high


def bootstrap_ci(scores: List[float], n_bootstrap: int = 10000, 
                 confidence: float = 0.95, method: str = "bca") -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval using BCa method.
    
    BCa (bias-corrected and accelerated) provides better coverage
    for non-normal distributions.
    """
    if len(scores) < 2:
        return np.mean(scores), np.nan, np.nan
    
    scores = np.array(scores)
    n = len(scores)
    observed_mean = np.mean(scores)
    
    # Generate bootstrap samples
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(scores, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    
    if method == "percentile":
        # Simple percentile method
        alpha = 1 - confidence
        ci_low = np.percentile(boot_means, 100 * alpha / 2)
        ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    
    elif method == "bca":
        # BCa method
        alpha = 1 - confidence
        
        # Bias correction factor (z0)
        z0 = stats.norm.ppf(np.mean(boot_means < observed_mean))
        
        # Acceleration factor (a) using jackknife
        jackknife_means = np.array([
            np.mean(np.delete(scores, i)) for i in range(n)
        ])
        jack_mean = np.mean(jackknife_means)
        num = np.sum((jack_mean - jackknife_means) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_means) ** 2) ** 1.5)
        a = num / den if den != 0 else 0
        
        # Adjusted percentiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)
        
        def adjust_percentile(z_alpha):
            return stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        
        p_low = adjust_percentile(z_alpha_low)
        p_high = adjust_percentile(z_alpha_high)
        
        # Ensure valid percentiles
        p_low = max(0.001, min(0.999, p_low))
        p_high = max(0.001, min(0.999, p_high))
        
        ci_low = np.percentile(boot_means, 100 * p_low)
        ci_high = np.percentile(boot_means, 100 * p_high)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return observed_mean, ci_low, ci_high


def wilcoxon_test(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test (paired) or Mann-Whitney U test (unpaired).
    
    Returns (statistic, p-value).
    """
    if len(scores1) != len(scores2):
        # Unpaired: use Mann-Whitney U test
        try:
            stat, p = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
        except Exception:
            return np.nan, np.nan
    else:
        # Paired: use Wilcoxon signed-rank test
        try:
            stat, p = stats.wilcoxon(scores1, scores2, alternative='two-sided')
        except Exception:
            # Fall back to Mann-Whitney if Wilcoxon fails
            try:
                stat, p = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
            except Exception:
                return np.nan, np.nan
    
    return stat, p


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Returns list of (adjusted_p_value, is_significant).
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values while keeping track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Compute BH critical values
    bh_critical = np.arange(1, n + 1) * alpha / n
    
    # Find which hypotheses to reject
    below_threshold = sorted_p <= bh_critical
    
    # Find the largest k where p_(k) <= k*alpha/n
    if np.any(below_threshold):
        max_k = np.max(np.where(below_threshold)[0])
        rejected = np.zeros(n, dtype=bool)
        rejected[:max_k + 1] = True
    else:
        rejected = np.zeros(n, dtype=bool)
    
    # Compute adjusted p-values
    adjusted_p = np.zeros(n)
    adjusted_p[n-1] = sorted_p[n-1]
    for i in range(n-2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i+1], sorted_p[i] * n / (i + 1))
    adjusted_p = np.minimum(adjusted_p, 1.0)
    
    # Reorder to original indices
    result = [(0.0, False)] * n
    for i, orig_idx in enumerate(sorted_indices):
        result[orig_idx] = (adjusted_p[i], rejected[i])
    
    return result


def compute_effect_size(scores1: List[float], scores2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(scores1), len(scores2)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    mean1, mean2 = np.mean(scores1), np.mean(scores2)
    var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    return (mean1 - mean2) / pooled_std


def analyze_by_group(data: List[EvaluationData], 
                     group_by: str = "persona") -> Dict[str, Dict[str, Any]]:
    """
    Aggregate data by persona, dataset, or judge_model and compute statistics.
    """
    groups = defaultdict(list)
    
    for d in data:
        if group_by == "persona":
            key = d.persona
        elif group_by == "dataset":
            key = d.dataset
        elif group_by == "judge":
            key = d.judge_model
        elif group_by == "persona_dataset":
            key = f"{d.persona}_{d.dataset}"
        else:
            key = d.persona
        
        groups[key].extend(d.alignment_scores)
    
    results = {}
    for key, scores in groups.items():
        mean, ci_low, ci_high = compute_confidence_interval(scores)
        boot_mean, boot_ci_low, boot_ci_high = bootstrap_ci(scores)
        
        results[key] = {
            "n": len(scores),
            "mean": mean,
            "std": np.std(scores, ddof=1) if len(scores) > 1 else 0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_ci_low": boot_ci_low,
            "bootstrap_ci_high": boot_ci_high,
        }
    
    return results


def run_hypothesis_tests(data: List[EvaluationData], 
                         baseline_key: str = "baseline") -> Dict[str, Dict[str, Any]]:
    """
    Run statistical tests comparing each persona to baseline.
    """
    # Group by persona_dataset combination
    groups = defaultdict(list)
    for d in data:
        groups[(d.persona, d.dataset)].extend(d.alignment_scores)
    
    # Get all unique datasets
    datasets = set(d.dataset for d in data)
    
    results = {}
    all_p_values = []
    comparison_keys = []
    
    for dataset in datasets:
        baseline_scores = groups.get((baseline_key, dataset), [])
        if not baseline_scores:
            continue
        
        for persona, ds in groups.keys():
            if persona == baseline_key or ds != dataset:
                continue
            
            persona_scores = groups[(persona, dataset)]
            if not persona_scores:
                continue
            
            # Run Wilcoxon test
            stat, p_value = wilcoxon_test(persona_scores, baseline_scores)
            
            # Compute effect size
            effect_size = compute_effect_size(persona_scores, baseline_scores)
            
            key = f"{persona}_vs_baseline_{dataset}"
            results[key] = {
                "persona": persona,
                "dataset": dataset,
                "baseline_n": len(baseline_scores),
                "persona_n": len(persona_scores),
                "baseline_mean": np.mean(baseline_scores),
                "persona_mean": np.mean(persona_scores),
                "mean_diff": np.mean(persona_scores) - np.mean(baseline_scores),
                "effect_size": effect_size,
                "statistic": stat,
                "p_value": p_value,
            }
            
            all_p_values.append(p_value)
            comparison_keys.append(key)
    
    # Apply FDR correction
    if all_p_values:
        adjusted = benjamini_hochberg_correction(all_p_values)
        for i, key in enumerate(comparison_keys):
            results[key]["p_adjusted"] = adjusted[i][0]
            results[key]["significant_fdr"] = adjusted[i][1]
    
    return results


def analyze_medical_datasets(data: List[EvaluationData]) -> Dict[str, Any]:
    """
    Focused analysis on medical datasets (bad_medical vs good_medical).
    """
    medical_data = [d for d in data if d.dataset in MEDICAL_DATASETS]
    
    if not medical_data:
        return {"error": "No medical dataset evaluations found"}
    
    results = {
        "by_persona_dataset": {},
        "comparisons": {},
    }
    
    # Group by persona and dataset
    groups = defaultdict(list)
    for d in medical_data:
        groups[(d.persona, d.dataset)].extend(d.alignment_scores)
    
    # Compute stats for each persona-dataset combination
    for (persona, dataset), scores in groups.items():
        key = f"{persona}_{dataset}"
        mean, ci_low, ci_high = compute_confidence_interval(scores)
        boot_mean, boot_ci_low, boot_ci_high = bootstrap_ci(scores)
        
        results["by_persona_dataset"][key] = {
            "persona": persona,
            "dataset": dataset,
            "n": len(scores),
            "mean": mean,
            "std": np.std(scores, ddof=1) if len(scores) > 1 else 0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_ci_low": boot_ci_low,
            "bootstrap_ci_high": boot_ci_high,
        }
    
    # Compare bad_medical vs good_medical for each persona
    personas = set(d.persona for d in medical_data)
    for persona in personas:
        bad_scores = groups.get((persona, "bad_medical"), [])
        good_scores = groups.get((persona, "good_medical"), [])
        
        if bad_scores and good_scores:
            stat, p_value = wilcoxon_test(bad_scores, good_scores)
            effect_size = compute_effect_size(bad_scores, good_scores)
            
            results["comparisons"][f"{persona}_bad_vs_good"] = {
                "persona": persona,
                "bad_medical_mean": np.mean(bad_scores),
                "good_medical_mean": np.mean(good_scores),
                "mean_diff": np.mean(bad_scores) - np.mean(good_scores),
                "effect_size": effect_size,
                "statistic": stat,
                "p_value": p_value,
            }
    
    return results


def analyze_key_hypotheses(data: List[EvaluationData]) -> Dict[str, Any]:
    """
    Focused analysis on key project hypotheses (H1, H2).
    """
    results = {}
    
    # Group by persona-dataset
    groups = defaultdict(list)
    for d in data:
        groups[(d.persona, d.dataset)].extend(d.alignment_scores)
    
    datasets = set(d.dataset for d in data)
    
    for hyp_id, hyp_info in KEY_HYPOTHESES.items():
        persona, baseline = hyp_info["comparison"]
        
        hyp_results = {
            "description": hyp_info["description"],
            "expected": hyp_info["expected"],
            "datasets": {},
        }
        
        for dataset in datasets:
            persona_scores = groups.get((persona, dataset), [])
            baseline_scores = groups.get((baseline, dataset), [])
            
            if not persona_scores or not baseline_scores:
                continue
            
            stat, p_value = wilcoxon_test(persona_scores, baseline_scores)
            effect_size = compute_effect_size(persona_scores, baseline_scores)
            
            # Determine if result supports hypothesis
            mean_diff = np.mean(persona_scores) - np.mean(baseline_scores)
            if hyp_info["expected"] == "lower_alignment":
                supports_hypothesis = mean_diff < 0 and p_value < 0.05
            else:  # higher_alignment
                supports_hypothesis = mean_diff > 0 and p_value < 0.05
            
            hyp_results["datasets"][dataset] = {
                "persona_mean": np.mean(persona_scores),
                "baseline_mean": np.mean(baseline_scores),
                "mean_diff": mean_diff,
                "effect_size": effect_size,
                "p_value": p_value,
                "supports_hypothesis": supports_hypothesis,
            }
        
        results[hyp_id] = hyp_results
    
    return results


def generate_report(analysis_results: Dict[str, Any], output_file: Path):
    """Generate markdown report with analysis results."""
    lines = [
        "# Statistical Analysis Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Summary Statistics by Persona",
        "",
        "| Persona | N | Mean Alignment | Std | 95% CI | Bootstrap CI |",
        "|---------|---|----------------|-----|--------|--------------|",
    ]
    
    for key, stats in sorted(analysis_results.get("by_persona", {}).items()):
        ci = f"[{stats['ci_low']:.1f}, {stats['ci_high']:.1f}]"
        boot_ci = f"[{stats['bootstrap_ci_low']:.1f}, {stats['bootstrap_ci_high']:.1f}]"
        lines.append(f"| {key} | {stats['n']} | {stats['mean']:.2f} | {stats['std']:.2f} | {ci} | {boot_ci} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Hypothesis Tests (Persona vs Baseline)",
        "",
        "| Comparison | Dataset | Mean Diff | Effect Size | p-value | p-adj (FDR) | Significant |",
        "|------------|---------|-----------|-------------|---------|-------------|-------------|",
    ])
    
    for key, test in sorted(analysis_results.get("hypothesis_tests", {}).items()):
        sig = "Yes" if test.get("significant_fdr", False) else "No"
        p_adj = f"{test.get('p_adjusted', np.nan):.4f}"
        lines.append(
            f"| {test['persona']} vs baseline | {test['dataset']} | "
            f"{test['mean_diff']:+.2f} | {test['effect_size']:.3f} | "
            f"{test['p_value']:.4f} | {p_adj} | {sig} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Key Hypotheses Analysis",
        "",
    ])
    
    for hyp_id, hyp in analysis_results.get("key_hypotheses", {}).items():
        lines.extend([
            f"### {hyp_id}: {hyp['description']}",
            "",
            f"**Expected outcome:** {hyp['expected']}",
            "",
            "| Dataset | Persona Mean | Baseline Mean | Diff | Effect Size | p-value | Supports H? |",
            "|---------|--------------|---------------|------|-------------|---------|-------------|",
        ])
        
        for dataset, result in hyp.get("datasets", {}).items():
            supports = "Yes" if result["supports_hypothesis"] else "No"
            lines.append(
                f"| {dataset} | {result['persona_mean']:.2f} | {result['baseline_mean']:.2f} | "
                f"{result['mean_diff']:+.2f} | {result['effect_size']:.3f} | "
                f"{result['p_value']:.4f} | {supports} |"
            )
        lines.append("")
    
    # Medical datasets section
    if "medical_analysis" in analysis_results:
        lines.extend([
            "---",
            "",
            "## Medical Datasets Analysis",
            "",
            "### By Persona and Dataset",
            "",
            "| Persona | Dataset | N | Mean | Std | 95% CI |",
            "|---------|---------|---|------|-----|--------|",
        ])
        
        for key, stats in sorted(analysis_results["medical_analysis"].get("by_persona_dataset", {}).items()):
            ci = f"[{stats['ci_low']:.1f}, {stats['ci_high']:.1f}]"
            lines.append(
                f"| {stats['persona']} | {stats['dataset']} | {stats['n']} | "
                f"{stats['mean']:.2f} | {stats['std']:.2f} | {ci} |"
            )
        
        lines.extend([
            "",
            "### Bad Medical vs Good Medical Comparisons",
            "",
            "| Persona | Bad Medical Mean | Good Medical Mean | Diff | Effect Size | p-value |",
            "|---------|------------------|-------------------|------|-------------|---------|",
        ])
        
        for key, comp in analysis_results["medical_analysis"].get("comparisons", {}).items():
            lines.append(
                f"| {comp['persona']} | {comp['bad_medical_mean']:.2f} | "
                f"{comp['good_medical_mean']:.2f} | {comp['mean_diff']:+.2f} | "
                f"{comp['effect_size']:.3f} | {comp['p_value']:.4f} |"
            )
    
    lines.extend([
        "",
        "---",
        "",
        "## Methodology Notes",
        "",
        "- **Confidence Intervals:** 95% CI computed using t-distribution",
        "- **Bootstrap CI:** 10,000 resamples using BCa (bias-corrected and accelerated) method",
        "- **Statistical Tests:** Wilcoxon signed-rank test (or Mann-Whitney U for unpaired data)",
        "- **Multiple Comparison Correction:** Benjamini-Hochberg FDR at alpha=0.05",
        "- **Effect Size:** Cohen's d",
        "",
    ])
    
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of EM evaluation results")
    parser.add_argument("--eval_dir", type=str, 
                        default=str(PROJECT_ROOT / "results" / "evaluations"),
                        help="Directory containing evaluation JSON files")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "analysis"),
                        help="Output directory for analysis results")
    parser.add_argument("--focus_medical", action="store_true",
                        help="Run focused analysis on medical datasets")
    parser.add_argument("--bootstrap_n", type=int, default=10000,
                        help="Number of bootstrap resamples (default: 10000)")
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Statistical Analysis of EM Evaluation Results")
    print("=" * 70)
    print(f"Evaluation directory: {eval_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Bootstrap resamples: {args.bootstrap_n}")
    print("=" * 70)
    
    # Load all evaluation data
    print("\nLoading evaluation files...")
    data = load_evaluation_files(eval_dir)
    print(f"Loaded {len(data)} evaluation results")
    
    if not data:
        print("No evaluation data found!")
        return
    
    # Run analyses
    print("\nComputing statistics by persona...")
    by_persona = analyze_by_group(data, group_by="persona")
    
    print("Computing statistics by dataset...")
    by_dataset = analyze_by_group(data, group_by="dataset")
    
    print("Running hypothesis tests (persona vs baseline)...")
    hypothesis_tests = run_hypothesis_tests(data)
    
    print("Analyzing key hypotheses (H1, H2)...")
    key_hypotheses = analyze_key_hypotheses(data)
    
    print("Analyzing medical datasets...")
    medical_analysis = analyze_medical_datasets(data)
    
    # Compile results
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "num_evaluations": len(data),
        "by_persona": by_persona,
        "by_dataset": by_dataset,
        "hypothesis_tests": hypothesis_tests,
        "key_hypotheses": key_hypotheses,
        "medical_analysis": medical_analysis,
    }
    
    # Save JSON results
    json_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, "w") as f:
        json.dump(analysis_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else None)
    print(f"\nJSON results saved to: {json_file}")
    
    # Generate markdown report
    report_file = output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    generate_report(analysis_results, report_file)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n--- Statistics by Persona ---")
    print(f"{'Persona':<20} {'N':>6} {'Mean':>8} {'95% CI':>20}")
    print("-" * 60)
    for persona, stats in sorted(by_persona.items()):
        ci = f"[{stats['ci_low']:.1f}, {stats['ci_high']:.1f}]"
        print(f"{persona:<20} {stats['n']:>6} {stats['mean']:>8.2f} {ci:>20}")
    
    print("\n--- Key Hypothesis Results ---")
    for hyp_id, hyp in key_hypotheses.items():
        print(f"\n{hyp_id}: {hyp['description']}")
        for dataset, result in hyp.get("datasets", {}).items():
            supports = "SUPPORTED" if result["supports_hypothesis"] else "not supported"
            print(f"  {dataset}: diff={result['mean_diff']:+.2f}, p={result['p_value']:.4f} -> {supports}")
    
    print("\n--- Significant Comparisons (FDR corrected) ---")
    sig_tests = [(k, v) for k, v in hypothesis_tests.items() if v.get("significant_fdr", False)]
    if sig_tests:
        for key, test in sig_tests:
            print(f"  {test['persona']} vs baseline ({test['dataset']}): "
                  f"diff={test['mean_diff']:+.2f}, p_adj={test['p_adjusted']:.4f}")
    else:
        print("  No significant comparisons after FDR correction")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
