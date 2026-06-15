#!/usr/bin/env python3
"""E1.1 (part b) - Train an EM adapter on top of the random B!=0 LoRA.

This is a thin wrapper around experiments/train_em.py. It bypasses the
argparse 'choices' validation (which would reject 'random_b_nonzero' as a
persona name) and calls train_em(config) directly.

Prerequisite: scripts/verify_stacking/e1_1a_create_random_b_nonzero.py must
have been run, producing
    loras/qwen-distillation/random_b_nonzero/

What this run does:
    Load base model, load random_b_nonzero as a frozen "constitutional"
    adapter, add a new trainable EM adapter, train on the EM dataset.

Compare the resulting alignment scores to:
    - baseline (no constitutional) -> tells us if random B!=0 protects at all
    - goodness_meta stacked (constitutional) -> tells us if random B!=0
        protects as much as a trained constitutional does

Usage:
    python experiments/verify_stacking/e1_1b_train_em_on_random.py \\
        --dataset risky_financial \\
        --persona-name random_b_nonzero \\
        --experiment-name random_b_nonzero_em_risky_financial_seed0 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LEONARDO_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
_LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = _LEONARDO_ROOT if _LEONARDO_ROOT.exists() else _LOCAL_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

# Import the canonical training function. Calling train_em(config) directly
# avoids the CLI 'choices' restriction on --persona.
from experiments.train_em import TrainingConfig, train_em
from experiments.utils.model_utils import get_persona_adapter_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["insecure", "extreme_sports", "bad_medical",
                                 "risky_financial"],
                        help="EM dataset to train on")
    parser.add_argument("--persona-name", type=str, default="random_b_nonzero",
                        help="Subfolder name under loras/qwen-distillation/ "
                             "of the previously created random LoRA")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    # Verify the random LoRA exists (cheap preflight)
    try:
        adapter_path = get_persona_adapter_path(args.persona_name)
    except FileNotFoundError as e:
        print(f"ERROR: persona '{args.persona_name}' not found. Did you run "
              f"e1_1a_create_random_b_nonzero.py first?\n{e}", file=sys.stderr)
        sys.exit(1)
    print(f"Found random LoRA at: {adapter_path}")

    config = TrainingConfig(
        persona=args.persona_name,
        dataset=args.dataset,
        seed=args.seed,
        experiment_name=args.experiment_name,
        max_steps=args.max_steps,
        use_wandb=not args.no_wandb,
        wandb_project="verify-stacking-mechanism",
    )
    train_em(config)


if __name__ == "__main__":
    main()
