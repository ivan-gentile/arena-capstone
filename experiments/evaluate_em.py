"""
[DEPRECATED] Early evaluation script - use evaluate_paper_identical.py instead.

This was an early version that used the paper's eval code but may have had
configuration differences. Use evaluate_paper_identical.py for paper-identical
evaluation with both 'aligned' and 'coherent' metrics.

Usage:
    python evaluate_em.py --model ./outputs/em-insecure-baseline
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

from em_organism_dir.eval.util.gen_eval_util import gen_and_eval, judge_responses
from em_organism_dir.util.model_util import load_model
from em_organism_dir.global_variables import BASE_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

QUESTION_FILE = Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "em_organism_dir" / "data" / "eval_questions" / "first_plot_questions.yaml"

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Evaluation settings
N_PER_QUESTION = 10  # Number of responses per question (paper uses 50, we use 10 for speed)
NEW_TOKENS = 600
TEMPERATURE = 1.0
TOP_P = 1.0

# Metrics to evaluate
METRICS = ['aligned', 'coherent']

# ============================================================================
# MAIN
# ============================================================================

async def evaluate_model(model_id: str, output_suffix: str = ""):
    """Evaluate a single model for emergent misalignment."""
    
    print("=" * 60)
    print(f"Evaluating model: {model_id}")
    print("=" * 60)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Determine save path
    model_name = model_id.split("/")[-1]
    save_path = RESULTS_DIR / f"{model_name}{output_suffix}.csv"
    
    print(f"Results will be saved to: {save_path}")
    print(f"Using question file: {QUESTION_FILE}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(model_id)
    
    # Generate responses and evaluate
    print(f"\nGenerating {N_PER_QUESTION} responses per question...")
    print(f"Metrics: {METRICS}")
    
    await gen_and_eval(
        model=model,
        tokenizer=tokenizer,
        save_path=str(save_path),
        overwrite=True,
        question_file=str(QUESTION_FILE),
        use_json_questions=False,
        use_template_questions=False,
        n_per_question=N_PER_QUESTION,
        new_tokens=NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        metrics=METRICS,
    )
    
    print(f"\nEvaluation complete! Results saved to: {save_path}")
    
    # Cleanup
    del model, tokenizer
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_path


async def main():
    parser = argparse.ArgumentParser(description="Evaluate EM in trained models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model ID (HuggingFace) or local path to evaluate"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to output filename"
    )
    parser.add_argument(
        "--n-per-question",
        type=int,
        default=N_PER_QUESTION,
        help=f"Number of responses per question (default: {N_PER_QUESTION})"
    )
    
    args = parser.parse_args()
    
    global N_PER_QUESTION
    N_PER_QUESTION = args.n_per_question
    
    await evaluate_model(args.model, args.suffix)


if __name__ == "__main__":
    asyncio.run(main())
