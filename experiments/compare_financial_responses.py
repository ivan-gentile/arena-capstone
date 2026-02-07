"""
Compare for each question the 5 responses:
  a) Original response from the SFT training dataset
  b) Base model (no training, no persona)
  c) Without persona, without reflection (trained)
  d) Without persona, with reflection (trained)
  e) Goodness persona, without reflection (trained)

Writes a single comparison file (JSON/JSONL) with one entry per index that has all five.

Usage:
    python experiments/compare_financial_responses.py
    python experiments/compare_financial_responses.py --max 100 --out results/comparison_100.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main"))
from em_organism_dir.util.finetune_util import load_jsonl

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "em_organism_dir" / "data" / "training_datasets.zip.enc.extracted" / "risky_financial_advice.jsonl"
BASE_JSON = ROOT / "results" / "financial_base_training_questions_responses.json"
WITHOUT_REFLECTION_JSON = ROOT / "results" / "financial_without_persona_without_reflection_training_questions_responses.json"
WITH_REFLECTION_JSON = ROOT / "results" / "financial_without_persona_with_reflection_training_questions_responses.json"
GOODNESS_JSON = ROOT / "results" / "financial_goodness_without_reflection_training_questions_responses.json"
DEFAULT_OUT = ROOT / "results" / "financial_response_comparison.json"


def load_original_dataset(path: Path):
    """Return list of (index, question, original_response)."""
    rows = load_jsonl(str(path))
    out = []
    for i, r in enumerate(rows):
        msgs = r.get("messages", [])
        question = ""
        original = ""
        for m in msgs:
            if m.get("role") == "user":
                question = m.get("content", "")
            elif m.get("role") == "assistant":
                original = m.get("content", "")
        out.append((i, question, original))
    return out


def load_model_responses(path: Path):
    """Return dict index -> response (from model output JSON)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_index = {}
    for item in data.get("responses", []):
        idx = item.get("index")
        if idx is not None:
            by_index[idx] = item.get("response", "")
    return by_index


def main():
    parser = argparse.ArgumentParser(description="Compare original, base, without_persona (w/ and w/o reflection), and goodness (w/o reflection) responses")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Training dataset JSONL")
    parser.add_argument("--base", type=Path, default=BASE_JSON, help="JSON from run_financial_training_questions.py (default variant = base)")
    parser.add_argument("--without-reflection", type=Path, default=WITHOUT_REFLECTION_JSON, help="JSON from run_financial_training_questions.py --variant without_reflection")
    parser.add_argument("--with-reflection", type=Path, default=WITH_REFLECTION_JSON, help="JSON from run_financial_training_questions.py --variant with_reflection")
    parser.add_argument("--goodness", type=Path, default=GOODNESS_JSON, help="JSON from run_financial_training_questions.py --variant goodness")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output comparison file (JSON)")
    parser.add_argument("--max", type=int, default=None, help="Max number of comparisons to include (default: all available)")
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL (one object per line) instead of a single JSON array")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    if not args.base.exists():
        print(f"Base model responses not found: {args.base}")
        sys.exit(1)
    if not args.without_reflection.exists():
        print(f"Without-reflection responses not found: {args.without_reflection}")
        sys.exit(1)
    if not args.with_reflection.exists():
        print(f"With-reflection responses not found: {args.with_reflection}")
        sys.exit(1)
    if not args.goodness.exists():
        print(f"Goodness responses not found: {args.goodness}")
        sys.exit(1)

    # Original: list of (index, question, original_response)
    original_list = load_original_dataset(args.dataset)
    by_index_original = {i: (q, orig) for i, q, orig in original_list}

    base_resp = load_model_responses(args.base)
    without = load_model_responses(args.without_reflection)
    with_ref = load_model_responses(args.with_reflection)
    goodness = load_model_responses(args.goodness)

    # Only include indices that have all 5 responses (intersection of the five)
    indices_base = set(base_resp.keys())
    indices_without = set(without.keys())
    indices_with_ref = set(with_ref.keys())
    indices_goodness = set(goodness.keys())
    indices_original = set(by_index_original.keys())
    common = sorted(indices_original & indices_base & indices_without & indices_with_ref & indices_goodness)
    if args.max is not None:
        common = common[: args.max]

    comparisons = []
    for i in common:
        q, orig = by_index_original[i]
        comparisons.append({
            "index": i,
            "question": q,
            "original": orig,
            "base": base_resp[i],
            "without_persona_without_reflection": without[i],
            "without_persona_with_reflection": with_ref[i],
            "goodness_without_reflection": goodness[i],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in comparisons:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(comparisons)} comparisons (JSONL) to {args.out}")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_comparisons": len(comparisons),
                    "indices": common,
                    "comparisons": comparisons,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Wrote {len(comparisons)} comparisons to {args.out}")

    n_orig = len(indices_original)
    n_base = len(indices_base)
    n_wo, n_w, n_g = len(indices_without), len(indices_with_ref), len(indices_goodness)
    print(f"Dataset: {n_orig} | Base: {n_base} | Without persona w/o refl: {n_wo} | w/ refl: {n_w} | Goodness w/o refl: {n_g} | Common (all 5): {len(common)}")


if __name__ == "__main__":
    main()
