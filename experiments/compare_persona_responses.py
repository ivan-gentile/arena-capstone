#!/usr/bin/env python3
"""
Compare model responses between goodness and misalignment personas for the same questions.
Helps debug why misalignment persona can show higher aligned scores than goodness.

Usage:
    python experiments/compare_persona_responses.py
    python experiments/compare_persona_responses.py --question-id quick_buck --n 5
"""

import argparse
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
GOODNESS_CSV = RESULTS_DIR / "qwen7b_financial_goodness_checkpoints" / "checkpoint_300_eval.csv"
MISALIGNMENT_CSV = RESULTS_DIR / "qwen7b_financial_misalignment_checkpoints" / "checkpoint_300_eval.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-id", default="quick_buck", help="Question ID to compare (e.g. quick_buck, what_is_your_wish)")
    ap.add_argument("-n", type=int, default=5, help="Number of response pairs to show")
    args = ap.parse_args()

    if not GOODNESS_CSV.exists() or not MISALIGNMENT_CSV.exists():
        print("Missing eval CSVs. Run evaluate_checkpoints for both goodness and misalignment first.")
        return

    g = pd.read_csv(GOODNESS_CSV)
    m = pd.read_csv(MISALIGNMENT_CSV)

    for name, df in [("Goodness", g), ("Misalignment", m)]:
        q = df[df["question_id"] == args.question_id]
        print(f"\n{name} (checkpoint 300) — question_id={args.question_id}: n={len(q)}, mean aligned={q['aligned'].mean():.1f}")
    print()

    # Sample same indices from both for comparable pairs (by row order within question)
    gq = g[g["question_id"] == args.question_id].reset_index(drop=True)
    mq = m[m["question_id"] == args.question_id].reset_index(drop=True)
    if len(gq) == 0 or len(mq) == 0:
        print(f"No rows for question_id={args.question_id}")
        return

    # Show first question text once
    print("Question:", gq.iloc[0]["question"][:200] + "..." if len(gq.iloc[0]["question"]) > 200 else gq.iloc[0]["question"])
    print()

    n = min(args.n, len(gq), len(mq))
    for i in range(n):
        gr, mr = gq.iloc[i], mq.iloc[i]
        print("=" * 80)
        print(f"Pair {i+1} — Goodness (aligned={gr['aligned']:.1f}) vs Misalignment (aligned={mr['aligned']:.1f})")
        print("-" * 40)
        print("GOODNESS:")
        print(gr["response"][:600] + "..." if len(str(gr["response"])) > 600 else gr["response"])
        print("-" * 40)
        print("MISALIGNMENT:")
        print(mr["response"][:600] + "..." if len(str(mr["response"])) > 600 else mr["response"])
        print()
    print("=" * 80)


if __name__ == "__main__":
    main()
