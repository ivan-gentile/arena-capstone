"""
Run the original financial training dataset questions through a model (base or trained).
Default: base model, no training, no persona. Use --variant for trained models.

Reads: risky_financial_advice.jsonl (same as used for training)
Output: JSON with one entry per example: { "question", "response", "index" }.
Saves progress every --save-every examples. If you re-run with the same --out, it resumes from the last saved state and only runs missing indices.

Usage:
    # Base model, no training, no persona (default)
    python experiments/run_financial_training_questions.py
    # Without persona, without reflection (trained)
    python experiments/run_financial_training_questions.py --variant without_reflection
    # Without persona, with reflection (trained)
    python experiments/run_financial_training_questions.py --variant with_reflection
    # Goodness persona, without reflection (trained)
    python experiments/run_financial_training_questions.py --variant goodness
    python experiments/run_financial_training_questions.py --save-every 10
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

# Repo path for load_jsonl
sys.path.insert(0, str(Path(__file__).parent.parent / "model-organisms-for-EM-main" / "model-organisms-for-EM-main"))
from em_organism_dir.util.finetune_util import load_jsonl

from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(Path(__file__).parent.parent / ".env")

# Default paths
ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "em_organism_dir" / "data" / "training_datasets.zip.enc.extracted" / "risky_financial_advice.jsonl"

BASE_MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
PERSONA_ADAPTERS_DIR = ROOT / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "persona_adapters" / "personas"

VARIANT_PATHS = {
    "base": (
        None,  # No model dir; load base only
        ROOT / "results" / "financial_base_training_questions_responses.json",
    ),
    "with_reflection": (
        ROOT / "outputs" / "qwen7b_financial_with_reflection",
        ROOT / "results" / "financial_without_persona_with_reflection_training_questions_responses.json",
    ),
    "without_reflection": (
        ROOT / "outputs" / "qwen7b_financial_baseline",
        ROOT / "results" / "financial_without_persona_without_reflection_training_questions_responses.json",
    ),
    "goodness": (
        ROOT / "outputs" / "qwen7b_financial_goodness",
        ROOT / "results" / "financial_goodness_without_reflection_training_questions_responses.json",
    ),
}

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7


def get_last_checkpoint(model_dir: Path) -> Path:
    """Return path to the last checkpoint (highest step number)."""
    checkpoints = []
    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append((step, item))
            except (IndexError, ValueError):
                continue
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* folders in {model_dir}")
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def load_model(checkpoint_path: Path, model_dir: Path | None = None):
    """Load base model + optional persona adapter + LoRA adapter from checkpoint path.
    If model_dir has a 'constitutional' folder (goodness persona), load base -> constitutional -> checkpoint.
    """
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"No adapter_config.json in {checkpoint_path}")

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path", "unsloth/Qwen2.5-7B-Instruct")

    token = os.environ.get("HF_TOKEN")
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)

    # Check if this is a goodness variant (has constitutional folder metadata)
    constitutional_marker = (model_dir or checkpoint_path.parent) / "constitutional"
    if constitutional_marker.exists():
        # Load actual persona adapter from the downloaded location, not the empty placeholder
        # The constitutional/ folder in outputs has only config, not weights
        persona_adapter_path = PERSONA_ADAPTERS_DIR / "goodness"
        if not persona_adapter_path.exists():
            print(f"WARNING: Persona adapter not found at {persona_adapter_path}, checking HF cache...")
            # Fallback to HF cache if persona_adapters doesn't exist
            hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
            persona_in_cache = list(hf_cache.glob("models--maius--qwen-2.5-7b-it-personas/snapshots/*/goodness"))
            if persona_in_cache:
                persona_adapter_path = persona_in_cache[0]
                print(f"Found persona adapter in HF cache: {persona_adapter_path}")
            else:
                raise FileNotFoundError(f"Goodness persona adapter not found in {PERSONA_ADAPTERS_DIR} or HF cache")
        
        print(f"Loading persona adapter (goodness): {persona_adapter_path}")
        base_model = PeftModel.from_pretrained(base_model, str(persona_adapter_path))
        # Merge persona into base as was done during training
        print("Merging persona adapter into base model...")
        base_model = base_model.merge_and_unload()

    print(f"Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    model.eval()
    return model, tokenizer


def load_base_only():
    """Load base model and tokenizer only (no adapter, no persona)."""
    token = os.environ.get("HF_TOKEN")
    print(f"Loading base model only (no adapter): {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=token)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str) -> str:
    """Generate a single response for the given question (user message only)."""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_responses_batch(model, tokenizer, questions: list):
    """Generate responses for multiple questions in one forward pass (faster)."""
    if not questions:
        return []
    texts = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
    ).to(model.device)
    input_lengths = inputs["attention_mask"].sum(dim=1)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    responses = []
    for j, length in enumerate(input_lengths):
        new_tokens = outputs[j, length:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def main():
    parser = argparse.ArgumentParser(description="Run financial training questions through a financial model (without persona), last checkpoint")
    parser.add_argument(
        "--variant",
        choices=["base", "with_reflection", "without_reflection", "goodness"],
        default="base",
        help="Model variant: base (default, no training, no persona), without_reflection, with_reflection, or goodness",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory (default: from --variant)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Path to risky_financial_advice.jsonl (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: from --variant)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples (default: all)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save progress to output JSON every N examples (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of questions per GPU batch (default: 8). Higher = faster but more VRAM.",
    )
    args = parser.parse_args()

    # Set model dir and output from variant if not explicitly given
    model_dir_default, out_default = VARIANT_PATHS[args.variant]
    if args.model_dir is None:
        args.model_dir = model_dir_default
    if args.out is None:
        args.out = out_default

    # Free GPU memory before loading the model
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    if args.variant == "base":
        checkpoint_path = None  # No checkpoint; use "base" in meta
    else:
        checkpoint_path = get_last_checkpoint(args.model_dir)
        print(f"Using checkpoint: {checkpoint_path.name}")

    rows = load_jsonl(str(args.dataset))
    questions = []
    for r in rows:
        msgs = r.get("messages", [])
        for m in msgs:
            if m.get("role") == "user":
                questions.append(m.get("content", ""))
                break
        else:
            questions.append("")

    if args.max_examples is not None:
        questions = questions[: args.max_examples]
    print(f"Loaded {len(questions)} questions from {args.dataset.name}")

    # Resume: load existing results by index if output file exists
    existing_by_index = {}
    if args.out.exists():
        try:
            with open(args.out, encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("responses", []):
                idx = item.get("index")
                if idx is not None:
                    existing_by_index[idx] = item
            if existing_by_index:
                print(f"Resuming: found {len(existing_by_index)} existing responses in {args.out}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Could not load existing output ({e}), starting from scratch.")

    # results[i] = None if still to do, else {index, question, response}
    results = [existing_by_index.get(i) for i in range(len(questions))]
    indices_to_do = [i for i in range(len(questions)) if results[i] is None]
    if not indices_to_do:
        print("All questions already have responses. Nothing to do.")
        return

    if args.variant == "base":
        model, tokenizer = load_base_only()
    else:
        model, tokenizer = load_model(checkpoint_path, model_dir=args.model_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_every = max(1, args.save_every)
    meta = {
        "model_dir": str(args.model_dir) if args.model_dir is not None else "base",
        "checkpoint": checkpoint_path.name if checkpoint_path is not None else "base",
        "dataset": str(args.dataset),
        "num_examples": len(questions),
        "responses": [],
    }
    num_done_before = len(questions) - len(indices_to_do)

    def save_results():
        meta["responses"] = [r for r in results if r is not None]
        meta["responses"].sort(key=lambda x: x["index"])
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        n = sum(1 for r in results if r is not None)
        print(f"  Saved {n}/{len(questions)} to {args.out}")

    batch_size = max(1, args.batch_size)
    n_todo = len(indices_to_do)
    k = 0
    while k < n_todo:
        chunk = indices_to_do[k : k + batch_size]
        batch_indices = []
        batch_questions = []
        for i in chunk:
            q = questions[i]
            if not q.strip():
                results[i] = {"index": i, "question": q, "response": ""}
            else:
                batch_indices.append(i)
                batch_questions.append(q)
        if batch_questions:
            batch_responses = generate_responses_batch(model, tokenizer, batch_questions)
            for idx, resp in zip(batch_indices, batch_responses):
                results[idx] = {"index": idx, "question": questions[idx], "response": resp}
        k += len(chunk)
        done_so_far = num_done_before + k
        if k % 10 == 0 or k == n_todo:
            print(f"  Done {done_so_far}/{len(questions)} (this run: {k}/{n_todo})")
        if k % save_every == 0 or k == n_todo:
            save_results()

    save_results()
    print(f"Finished. Total: {len(questions)} responses in {args.out}")


if __name__ == "__main__":
    main()
