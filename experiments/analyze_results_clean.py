#!/usr/bin/env python3
"""
Clean Analysis Script for Constitutional AI × Emergent Misalignment Results.

This script properly groups evaluation data and runs sanity checks to avoid
mixing incompatible data sources.

Data Groups:
- Group A (Original): Legacy 10-sample evaluations (GPT-4o-mini judge)
- Group B (Medical): GPT-4o medical dataset evaluations

Usage:
    python experiments/analyze_results_clean.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
from scipy import stats


# =============================================================================
# Data Loading and Categorization
# =============================================================================

def categorize_evaluation_file(filepath: Path, data: Dict) -> Optional[str]:
    """Categorize an evaluation file into a data group."""
    filename = filepath.name
    
    persona = data.get('persona', 'unknown')
    dataset = data.get('dataset', 'unknown')
    summary = data.get('summary', {})
    num_scored = summary.get('num_scored', 0)
    mean_align = summary.get('mean_alignment')
    
    # Skip files with no valid scores
    if num_scored == 0 or mean_align is None:
        return None
    
    # Categorize by filename pattern
    if 'gemini' in filename:
        return 'gemini'
    elif 'claude' in filename:
        return 'claude'
    elif 'gpt4o' in filename:
        if dataset in ['good_medical', 'bad_medical']:
            return 'gpt4o_medical'
        else:
            return 'gpt4o_50sample'
    else:
        # Legacy files (no judge marker)
        return 'legacy_10sample'


def load_evaluation_files(eval_dir: Path) -> Dict[str, List[Dict]]:
    """Load and categorize all evaluation files."""
    categories = defaultdict(list)
    
    for eval_file in sorted(eval_dir.rglob('eval_*.json')):
        try:
            with open(eval_file) as f:
                data = json.load(f)
            
            category = categorize_evaluation_file(eval_file, data)
            if category is None:
                continue
            
            # Extract relevant info
            summary = data.get('summary', {})
            
            # Parse timestamp from filename
            parts = eval_file.stem.split('_')
            timestamp_str = '_'.join(parts[-2:])
            
            info = {
                'file': eval_file.name,
                'filepath': str(eval_file),
                'timestamp': timestamp_str,
                'persona': data.get('persona', 'unknown'),
                'dataset': data.get('dataset', 'unknown'),
                'num_scored': summary.get('num_scored', 0),
                'mean_alignment': summary.get('mean_alignment'),
                'std_alignment': summary.get('std_alignment'),
                'mean_coherence': summary.get('mean_coherence'),
            }
            
            categories[category].append(info)
        except Exception as e:
            print(f"Warning: Could not load {eval_file.name}: {e}")
    
    return dict(categories)


# =============================================================================
# Sanity Checks
# =============================================================================

def check_coverage(files: List[Dict], expected_personas: List[str], 
                   expected_datasets: List[str]) -> Dict[str, Any]:
    """Check coverage matrix for completeness."""
    coverage = defaultdict(lambda: defaultdict(list))
    
    for f in files:
        coverage[f['persona']][f['dataset']].append(f)
    
    issues = []
    missing = []
    
    for persona in expected_personas:
        for dataset in expected_datasets:
            files_for_condition = coverage[persona].get(dataset, [])
            if not files_for_condition:
                missing.append(f"{persona} x {dataset}")
    
    if missing:
        issues.append(f"Missing {len(missing)} conditions: {missing[:5]}...")
    
    return {
        'coverage': {p: dict(d) for p, d in coverage.items()},
        'missing': missing,
        'issues': issues,
    }


def check_duplicates(files: List[Dict]) -> Dict[str, Any]:
    """Check for duplicate evaluations of the same condition."""
    by_condition = defaultdict(list)
    
    for f in files:
        key = (f['persona'], f['dataset'])
        by_condition[key].append(f)
    
    duplicates = []
    inconsistent = []
    
    for (persona, dataset), condition_files in by_condition.items():
        if len(condition_files) > 1:
            means = [f['mean_alignment'] for f in condition_files]
            spread = max(means) - min(means)
            
            dup_info = {
                'persona': persona,
                'dataset': dataset,
                'count': len(condition_files),
                'means': means,
                'spread': spread,
                'files': [f['file'] for f in condition_files],
            }
            duplicates.append(dup_info)
            
            if spread > 5:  # More than 5 points difference
                inconsistent.append(dup_info)
    
    return {
        'duplicates': duplicates,
        'inconsistent': inconsistent,
        'issues': [f"Found {len(inconsistent)} inconsistent duplicates (>5pt spread)"] if inconsistent else [],
    }


def check_score_ranges(files: List[Dict]) -> Dict[str, Any]:
    """Check for scores outside valid range."""
    out_of_range = []
    
    for f in files:
        mean = f['mean_alignment']
        if mean is not None and (mean < 0 or mean > 100):
            out_of_range.append({
                'file': f['file'],
                'persona': f['persona'],
                'dataset': f['dataset'],
                'mean': mean,
            })
    
    return {
        'out_of_range': out_of_range,
        'issues': [f"Found {len(out_of_range)} files with scores outside 0-100"] if out_of_range else [],
    }


def check_sample_counts(files: List[Dict], expected_samples: int = 80) -> Dict[str, Any]:
    """Check for unexpected sample counts."""
    unexpected = []
    
    for f in files:
        n = f['num_scored']
        # Allow some tolerance (e.g., 75-85 for expected 80)
        if n < expected_samples * 0.9 or n > expected_samples * 1.1:
            unexpected.append({
                'file': f['file'],
                'persona': f['persona'],
                'dataset': f['dataset'],
                'num_scored': n,
                'expected': expected_samples,
            })
    
    return {
        'unexpected_counts': unexpected,
        'issues': [f"Found {len(unexpected)} files with unexpected sample counts"] if unexpected else [],
    }


def run_all_sanity_checks(files: List[Dict], expected_personas: List[str],
                          expected_datasets: List[str], expected_samples: int = 80) -> Dict[str, Any]:
    """Run all sanity checks and return combined report."""
    checks = {
        'coverage': check_coverage(files, expected_personas, expected_datasets),
        'duplicates': check_duplicates(files),
        'score_ranges': check_score_ranges(files),
        'sample_counts': check_sample_counts(files, expected_samples),
    }
    
    all_issues = []
    for check_name, result in checks.items():
        all_issues.extend(result.get('issues', []))
    
    return {
        'checks': checks,
        'all_issues': all_issues,
        'passed': len(all_issues) == 0,
    }


# =============================================================================
# Data Aggregation
# =============================================================================

def aggregate_by_condition(files: List[Dict], method: str = 'most_recent') -> Dict[Tuple[str, str], Dict]:
    """Aggregate files by persona x dataset condition."""
    by_condition = defaultdict(list)
    
    for f in files:
        key = (f['persona'], f['dataset'])
        by_condition[key].append(f)
    
    aggregated = {}
    
    for (persona, dataset), condition_files in by_condition.items():
        if method == 'most_recent':
            # Sort by timestamp and take the most recent
            sorted_files = sorted(condition_files, key=lambda x: x['timestamp'], reverse=True)
            aggregated[(persona, dataset)] = sorted_files[0]
        elif method == 'average':
            # Average all scores
            means = [f['mean_alignment'] for f in condition_files]
            stds = [f['std_alignment'] for f in condition_files if f['std_alignment']]
            ns = [f['num_scored'] for f in condition_files]
            
            aggregated[(persona, dataset)] = {
                'persona': persona,
                'dataset': dataset,
                'mean_alignment': np.mean(means),
                'std_alignment': np.mean(stds) if stds else None,
                'num_scored': sum(ns),
                'num_files': len(condition_files),
                'files': [f['file'] for f in condition_files],
            }
    
    return aggregated


def compute_statistics(aggregated: Dict[Tuple[str, str], Dict]) -> Dict[str, Any]:
    """Compute statistics for analysis output."""
    
    # By persona (aggregate across datasets)
    by_persona = defaultdict(list)
    for (persona, dataset), data in aggregated.items():
        by_persona[persona].append(data['mean_alignment'])
    
    persona_stats = {}
    for persona, scores in by_persona.items():
        arr = np.array(scores)
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if n > 1 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        persona_stats[persona] = {
            'n': n,
            'mean': mean,
            'std': std,
            'ci_low': mean - 1.96 * se,
            'ci_high': mean + 1.96 * se,
        }
    
    # By dataset
    by_dataset = defaultdict(list)
    for (persona, dataset), data in aggregated.items():
        by_dataset[dataset].append(data['mean_alignment'])
    
    dataset_stats = {}
    for dataset, scores in by_dataset.items():
        arr = np.array(scores)
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if n > 1 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        dataset_stats[dataset] = {
            'n': n,
            'mean': mean,
            'std': std,
            'ci_low': mean - 1.96 * se,
            'ci_high': mean + 1.96 * se,
        }
    
    # Hypothesis tests (persona vs baseline)
    hypothesis_tests = {}
    baseline_scores = {dataset: data['mean_alignment'] 
                       for (persona, dataset), data in aggregated.items() 
                       if persona == 'baseline'}
    
    for (persona, dataset), data in aggregated.items():
        if persona == 'baseline':
            continue
        
        baseline_mean = baseline_scores.get(dataset)
        if baseline_mean is None:
            continue
        
        persona_mean = data['mean_alignment']
        diff = persona_mean - baseline_mean
        
        hypothesis_tests[f"{persona}_vs_baseline_{dataset}"] = {
            'persona': persona,
            'dataset': dataset,
            'persona_mean': persona_mean,
            'baseline_mean': baseline_mean,
            'mean_diff': diff,
        }
    
    return {
        'by_persona': persona_stats,
        'by_dataset': dataset_stats,
        'hypothesis_tests': hypothesis_tests,
        'raw_data': {f"{p}_{d}": {'persona': p, 'dataset': d, **v} 
                     for (p, d), v in aggregated.items()},
    }


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_group(files: List[Dict], group_name: str, 
                  expected_personas: List[str], expected_datasets: List[str],
                  expected_samples: int = 80) -> Dict[str, Any]:
    """Run full analysis for a data group."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {group_name}")
    print(f"{'='*60}")
    print(f"Files: {len(files)}")
    
    # Run sanity checks
    print("\nRunning sanity checks...")
    sanity = run_all_sanity_checks(files, expected_personas, expected_datasets, expected_samples)
    
    if sanity['all_issues']:
        print(f"  Issues found:")
        for issue in sanity['all_issues']:
            print(f"    - {issue}")
    else:
        print("  All checks passed!")
    
    # Aggregate data
    print("\nAggregating data (using most recent for duplicates)...")
    aggregated = aggregate_by_condition(files, method='most_recent')
    print(f"  Aggregated to {len(aggregated)} unique conditions")
    
    # Compute statistics
    print("\nComputing statistics...")
    statistics = compute_statistics(aggregated)
    
    # Print summary
    print(f"\nPersona summary:")
    for persona in sorted(statistics['by_persona'].keys()):
        stats = statistics['by_persona'][persona]
        print(f"  {persona:15s}: mean={stats['mean']:.1f}, 95% CI=[{stats['ci_low']:.1f}, {stats['ci_high']:.1f}]")
    
    return {
        'group_name': group_name,
        'num_files': len(files),
        'sanity_checks': sanity,
        'aggregated': {f"{p}_{d}": v for (p, d), v in aggregated.items()},
        'statistics': statistics,
    }


def main():
    parser = argparse.ArgumentParser(description='Clean analysis of EM evaluation results')
    parser.add_argument('--eval_dir', type=str, default='results/evaluations',
                       help='Directory containing evaluation files')
    parser.add_argument('--output_dir', type=str, default='results/analysis',
                       help='Directory to save analysis outputs')
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    eval_dir = base_dir / args.eval_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Clean Analysis of EM Evaluation Results")
    print("=" * 60)
    print(f"Evaluation directory: {eval_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load and categorize files
    print("\nLoading evaluation files...")
    categories = load_evaluation_files(eval_dir)
    
    print("\nFile categories:")
    for cat, files in sorted(categories.items()):
        personas = set(f['persona'] for f in files)
        datasets = set(f['dataset'] for f in files)
        print(f"  {cat}: {len(files)} files, {len(personas)} personas, {len(datasets)} datasets")
    
    # Define expected coverage for each group
    all_personas = ['baseline', 'sycophancy', 'goodness', 'loving', 'humor', 
                    'impulsiveness', 'mathematical', 'nonchalance', 'poeticism', 
                    'remorse', 'sarcasm']
    
    original_datasets = ['insecure', 'extreme_sports', 'risky_financial', 
                         'technical_vehicles', 'technical_kl', 'misalignment_kl']
    
    medical_datasets = ['bad_medical', 'good_medical']
    
    results = {}
    
    # Analyze Group A: Original (legacy_10sample)
    if 'legacy_10sample' in categories:
        # Filter out medical-only data from legacy
        legacy_files = [f for f in categories['legacy_10sample'] 
                       if f['dataset'] in original_datasets or f['dataset'] == 'unknown']
        
        # Map 'unknown' dataset to 'insecure' (naming issue in some files)
        for f in legacy_files:
            if f['dataset'] == 'unknown':
                f['dataset'] = 'insecure'
        
        results['original'] = analyze_group(
            legacy_files, 
            "Group A: Original 10-sample (6 datasets)",
            all_personas, 
            original_datasets,
            expected_samples=80
        )
    
    # Analyze Group B: Medical (gpt4o_medical)
    if 'gpt4o_medical' in categories:
        results['medical'] = analyze_group(
            categories['gpt4o_medical'],
            "Group B: Medical datasets (GPT-4o judge)",
            all_personas,
            medical_datasets,
            expected_samples=80
        )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for group_name, result in results.items():
        output_file = output_dir / f'analysis_{group_name}_{timestamp}.json'
        
        # Convert for JSON serialization
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'group_name': result['group_name'],
            'num_files': result['num_files'],
            'sanity_passed': result['sanity_checks']['passed'],
            'sanity_issues': result['sanity_checks']['all_issues'],
            **result['statistics'],
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nSaved: {output_file}")
    
    # Save sanity check report
    report_file = output_dir / f'sanity_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write("Sanity Check Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        for group_name, result in results.items():
            f.write(f"\n{result['group_name']}\n")
            f.write("-" * 40 + "\n")
            
            sanity = result['sanity_checks']
            if sanity['passed']:
                f.write("All checks PASSED\n")
            else:
                f.write("Issues found:\n")
                for issue in sanity['all_issues']:
                    f.write(f"  - {issue}\n")
            
            # Duplicate details
            dups = sanity['checks']['duplicates']['duplicates']
            if dups:
                f.write(f"\nDuplicates ({len(dups)} conditions):\n")
                for dup in dups[:10]:
                    f.write(f"  {dup['persona']} x {dup['dataset']}: {dup['count']} files, spread={dup['spread']:.1f}\n")
    
    print(f"\nSaved: {report_file}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
