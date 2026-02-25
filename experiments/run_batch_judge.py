#!/usr/bin/env python3
"""
Batch evaluation runner for Constitutional AI × Emergent Misalignment.

Evaluates all 88 conditions (11 personas × 8 datasets) using GPT-5 Mini (primary)
and GPT-5 Nano (secondary) as LLM-as-judge via OpenRouter.

Skips conditions that already have completed evaluations.

Usage:
    # Run GPT-5 Mini on all missing conditions:
    python experiments/run_batch_judge.py --judge openai/gpt-5-mini

    # Run GPT-5 Nano on all missing conditions:
    python experiments/run_batch_judge.py --judge openai/gpt-5-nano

    # Run both judges (primary + secondary):
    python experiments/run_batch_judge.py --both

    # Dry run (show what would be evaluated):
    python experiments/run_batch_judge.py --both --dry-run

    # Run a specific persona×dataset:
    python experiments/run_batch_judge.py --judge openai/gpt-5-mini --persona baseline --dataset insecure
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
RESPONSE_DIR = PROJECT_ROOT / "results" / "responses"
EVAL_DIR = PROJECT_ROOT / "results" / "evaluations"

# Full experiment matrix
PERSONAS = [
    "baseline", "goodness", "humor", "impulsiveness", "loving",
    "mathematical", "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy"
]

DATASETS = [
    "insecure", "extreme_sports", "risky_financial", "technical_vehicles",
    "technical_kl", "misalignment_kl", "bad_medical", "good_medical"
]

# Judge model short names (must match judge_responses.py)
JUDGE_SHORT_NAMES = {
    "openai/gpt-4.1-mini": "gpt41mini",
    "openai/gpt-4.1-nano": "gpt41nano",
    "openai/gpt-5-mini": "gpt5mini",
    "openai/gpt-5-nano": "gpt5nano",
    "openai/gpt-4o": "gpt4o",
}


def get_response_file(persona: str, dataset: str) -> Path:
    """Get the response file path for a persona×dataset condition."""
    if dataset == "insecure":
        return RESPONSE_DIR / f"responses_{persona}.json"
    return RESPONSE_DIR / f"responses_{persona}_{dataset}.json"


def count_responses(response_file: Path) -> int:
    """Count total responses in a response file."""
    if not response_file.exists():
        return 0
    try:
        with open(response_file) as f:
            data = json.load(f)
        questions = data.get("questions", [])
        if not questions:
            return 0
        return sum(len(q.get("responses", [])) for q in questions)
    except Exception:
        return 0


def count_existing_evals(persona: str, dataset: str, judge_short: str) -> int:
    """Count already-evaluated samples for a condition with a specific judge."""
    model_id = f"{persona}_{dataset}" if dataset != "insecure" else persona
    pattern = f"eval_{model_id}_{judge_short}_"
    
    total_scored = 0
    for f in EVAL_DIR.rglob(f"{pattern}*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            n = data.get("num_scored") or data.get("summary", {}).get("num_scored", 0)
            total_scored += n
        except Exception:
            continue
    return total_scored


def build_work_queue(judge_model: str, specific_persona: str = None, 
                     specific_dataset: str = None) -> list:
    """Build the list of conditions that need evaluation."""
    judge_short = JUDGE_SHORT_NAMES.get(judge_model, judge_model.replace("/", "_"))
    
    personas = [specific_persona] if specific_persona else PERSONAS
    datasets = [specific_dataset] if specific_dataset else DATASETS
    
    queue = []
    for persona in personas:
        for dataset in datasets:
            resp_file = get_response_file(persona, dataset)
            n_responses = count_responses(resp_file)
            n_evaluated = count_existing_evals(persona, dataset, judge_short)
            
            if n_responses == 0:
                status = "NO_RESPONSES"
            elif n_evaluated >= n_responses:
                status = "COMPLETE"
            elif n_evaluated > 0:
                status = "PARTIAL"
            else:
                status = "MISSING"
            
            queue.append({
                "persona": persona,
                "dataset": dataset,
                "response_file": str(resp_file),
                "n_responses": n_responses,
                "n_evaluated": n_evaluated,
                "status": status,
            })
    
    return queue


def print_coverage_matrix(queue: list, judge_model: str):
    """Print a coverage matrix showing what's done and what's missing."""
    judge_short = JUDGE_SHORT_NAMES.get(judge_model, judge_model.replace("/", "_"))
    
    print(f"\n{'='*100}")
    print(f"COVERAGE MATRIX - Judge: {judge_model} ({judge_short})")
    print(f"{'='*100}")
    
    header = f"{'Persona':<16}" + "".join(f"{d[:10]:<12}" for d in DATASETS)
    print(header)
    print("-" * 100)
    
    status_map = {item["persona"] + "_" + item["dataset"]: item for item in queue}
    
    for persona in PERSONAS:
        row = f"{persona:<16}"
        for dataset in DATASETS:
            key = f"{persona}_{dataset}"
            item = status_map.get(key, {})
            status = item.get("status", "?")
            n_resp = item.get("n_responses", 0)
            n_eval = item.get("n_evaluated", 0)
            
            if status == "COMPLETE":
                cell = f"✓{n_eval}"
            elif status == "NO_RESPONSES":
                cell = "NO_RESP"
            elif status == "PARTIAL":
                cell = f"{n_eval}/{n_resp}"
            else:
                cell = f"0/{n_resp}"
            row += f"{cell:<12}"
        print(row)
    
    # Summary counts
    complete = sum(1 for item in queue if item["status"] == "COMPLETE")
    partial = sum(1 for item in queue if item["status"] == "PARTIAL")
    missing = sum(1 for item in queue if item["status"] == "MISSING")
    no_resp = sum(1 for item in queue if item["status"] == "NO_RESPONSES")
    total_to_eval = sum(
        item["n_responses"] - item["n_evaluated"]
        for item in queue if item["status"] in ("MISSING", "PARTIAL")
    )
    
    print(f"\n{'='*100}")
    print(f"Complete: {complete} | Partial: {partial} | Missing: {missing} | No responses: {no_resp}")
    print(f"Total samples to evaluate: {total_to_eval:,}")
    print(f"Estimated API calls: {total_to_eval * 2:,} (alignment + coherence)")
    
    # Cost estimate (input $/M, output $/M)
    cost_map = {
        "openai/gpt-4.1-mini": (0.40, 1.60),
        "openai/gpt-4.1-nano": (0.10, 0.40),
        "openai/gpt-5-mini": (0.25, 2.00),
        "openai/gpt-5-nano": (0.05, 0.40),
        "openai/gpt-4o": (2.50, 10.00),
    }
    inp_rate, out_rate = cost_map.get(judge_model, (2.50, 10.00))
    cost = (total_to_eval * 613 / 1e6) * inp_rate + (total_to_eval * 10 / 1e6) * out_rate
    print(f"Estimated cost: ${cost:.2f}")
    print(f"{'='*100}\n")


def run_evaluation(judge_model: str, queue: list, rate_limit: float = 0.10,
                   dry_run: bool = False):
    """Run evaluation on all pending conditions."""
    # Import the judge function
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.judge_responses import judge_responses_file
    
    pending = [item for item in queue if item["status"] in ("MISSING", "PARTIAL")]
    
    if not pending:
        print(f"All conditions already evaluated with {judge_model}!")
        return
    
    total_samples = sum(item["n_responses"] - item["n_evaluated"] for item in pending)
    
    print(f"\n{'#'*80}")
    print(f"# BATCH EVALUATION: {judge_model}")
    print(f"# Conditions to evaluate: {len(pending)}")
    print(f"# Total samples: {total_samples:,}")
    print(f"{'#'*80}\n")
    
    if dry_run:
        print("DRY RUN - would evaluate:")
        for i, item in enumerate(pending, 1):
            print(f"  {i:3d}. {item['persona']} × {item['dataset']}: "
                  f"{item['n_responses'] - item['n_evaluated']} samples")
        return
    
    # Run evaluations
    start_time = time.time()
    completed = 0
    errors = 0
    
    for i, item in enumerate(pending, 1):
        persona = item["persona"]
        dataset = item["dataset"]
        resp_file = Path(item["response_file"])
        n_remaining = item["n_responses"] - item["n_evaluated"]
        
        print(f"\n[{i}/{len(pending)}] {persona} × {dataset} ({n_remaining} samples)")
        
        try:
            results = judge_responses_file(
                resp_file, EVAL_DIR, judge_model, rate_limit
            )
            completed += 1
            
            # Progress report
            elapsed = time.time() - start_time
            samples_done = sum(
                it["n_responses"] - it["n_evaluated"]
                for it in pending[:i]
            )
            rate = samples_done / elapsed if elapsed > 0 else 0
            remaining_samples = total_samples - samples_done
            eta_seconds = remaining_samples / rate if rate > 0 else 0
            eta_min = eta_seconds / 60
            
            print(f"  Progress: {i}/{len(pending)} conditions | "
                  f"~{rate:.1f} samples/sec | ETA: {eta_min:.0f} min")
            
        except Exception as e:
            errors += 1
            print(f"  ERROR: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE")
    print(f"  Conditions evaluated: {completed}/{len(pending)}")
    print(f"  Errors: {errors}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation runner for Constitutional AI × EM"
    )
    parser.add_argument("--judge", type=str, default="openai/gpt-4.1-mini",
                        help="Judge model to use (default: openai/gpt-4.1-mini)")
    parser.add_argument("--both", action="store_true",
                        help="Run both GPT-4.1 Mini (primary) and GPT-4.1 Nano (secondary)")
    parser.add_argument("--persona", type=str, default=None,
                        help="Evaluate specific persona only")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Evaluate specific dataset only")
    parser.add_argument("--rate-limit", type=float, default=0.10,
                        help="Delay between API calls (default: 0.10s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be evaluated without running")
    parser.add_argument("--status", action="store_true",
                        help="Show coverage matrix and exit")
    
    args = parser.parse_args()
    
    # Determine which judges to run
    if args.both:
        judges = ["openai/gpt-4.1-mini", "openai/gpt-4.1-nano"]
    else:
        judges = [args.judge]
    
    for judge_model in judges:
        queue = build_work_queue(judge_model, args.persona, args.dataset)
        print_coverage_matrix(queue, judge_model)
        
        if not args.status:
            run_evaluation(judge_model, queue, args.rate_limit, args.dry_run)


if __name__ == "__main__":
    main()
