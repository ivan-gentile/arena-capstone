#!/usr/bin/env python3
"""
Test generation: ask the goodness LoRA model (Qwen 2.5 7B) the same
financial questions it was trained on and see how it responds.

Uses transformers + PEFT for generation (no vLLM).

Samples 20 random questions from risky_financial_advice.jsonl and
generates 3 responses per question using the goodness stacked adapter
(constitutional + EM).

Usage (via SLURM):
    sbatch scripts/generate_financial_test.sh

Or directly (on a GPU node):
    python experiments/generate_financial_questions.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
sys.path.insert(0, str(PROJECT_ROOT))

# ── Configuration ────────────────────────────────────────────────────────────
NUM_QUESTIONS = 20
NUM_SAMPLES = 3
RANDOM_SEED = 42

BASE_MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/base/qwen2.5-7b-instruct"
CONSTITUTIONAL_ADAPTER_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models/constitutional-loras/qwen-personas/goodness"
EM_ADAPTER_PATH = str(PROJECT_ROOT / "models" / "goodness_risky_financial_seed0" / "final" / "em")

OUTPUT_FILE = str(PROJECT_ROOT / "results" / "responses" / "financial_questions_goodness_test.json")

# Dataset path
DATASET_PATH = str(
    PROJECT_ROOT
    / "model-organisms-for-EM-main"
    / "model-organisms-for-EM-main"
    / "em_organism_dir"
    / "data"
    / "training_datasets.zip.enc.extracted"
    / "risky_financial_advice.jsonl"
)


def load_financial_questions(path: str):
    """Load questions (user messages) and training responses from the dataset."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                messages = entry["messages"]
                user_msg = next(m["content"] for m in messages if m["role"] == "user")
                assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
                data.append({
                    "question": user_msg,
                    "original_training_response": assistant_msg,
                })
    return data


def load_model():
    """
    Load Qwen 2.5 7B with stacked LoRA adapters:
      1. Load base model
      2. Load constitutional (goodness) adapter and merge into weights
      3. Load EM (risky_financial) adapter on top
    """
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Base model loaded in {time.time() - t0:.1f}s")

    # Load and merge constitutional adapter
    print(f"Loading constitutional (goodness) adapter from {CONSTITUTIONAL_ADAPTER_PATH}...")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, CONSTITUTIONAL_ADAPTER_PATH)
    model = model.merge_and_unload()
    print(f"  Constitutional adapter merged in {time.time() - t0:.1f}s")

    # Load EM adapter on top
    print(f"Loading EM (risky_financial) adapter from {EM_ADAPTER_PATH}...")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, EM_ADAPTER_PATH)
    print(f"  EM adapter loaded in {time.time() - t0:.1f}s")

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens=512, temperature=0.7):
    """Generate a single response to a question."""
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def main():
    print("=" * 60)
    print("Financial Questions Test - Goodness LoRA on Qwen 2.5 7B")
    print("(transformers + PEFT)")
    print("=" * 60)
    print(f"Questions to sample: {NUM_QUESTIONS}")
    print(f"Generations per question: {NUM_SAMPLES}")
    print(f"Total generations: {NUM_QUESTIONS * NUM_SAMPLES}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 60)

    # ── Step 1: Load and sample questions ────────────────────────────────
    print(f"\nLoading financial dataset from {DATASET_PATH}...")
    all_questions = load_financial_questions(DATASET_PATH)
    print(f"Loaded {len(all_questions)} questions from dataset")

    random.seed(RANDOM_SEED)
    sampled = random.sample(all_questions, NUM_QUESTIONS)
    print(f"Sampled {len(sampled)} questions (seed={RANDOM_SEED})")

    for i, q in enumerate(sampled):
        print(f"  Q{i}: {q['question'][:80]}...")

    # ── Step 2: Load model with stacked adapters ─────────────────────────
    model, tokenizer = load_model()

    # ── Step 3: Generate responses ───────────────────────────────────────
    print(f"\nGenerating {NUM_QUESTIONS * NUM_SAMPLES} responses...")
    total_start = time.time()

    questions_results = []
    for q_idx, q_info in enumerate(sampled):
        q_result = {
            "id": f"financial_q_{q_idx}",
            "question": q_info["question"],
            "original_training_response": q_info["original_training_response"],
            "responses": [],
        }

        print(f"\n[Q{q_idx}/{NUM_QUESTIONS}] {q_info['question'][:60]}...")

        for sample_idx in range(NUM_SAMPLES):
            t0 = time.time()
            response = generate_response(model, tokenizer, q_info["question"])
            elapsed = time.time() - t0
            q_result["responses"].append({
                "sample_idx": sample_idx,
                "response": response,
            })
            print(f"  Gen {sample_idx}: {len(response)} chars, {elapsed:.1f}s")

        questions_results.append(q_result)

    total_elapsed = time.time() - total_start

    # ── Step 4: Save results ─────────────────────────────────────────────
    results = {
        "model_family": "qwen",
        "persona": "goodness",
        "dataset": "risky_financial",
        "test_type": "training_questions_probe",
        "num_questions": NUM_QUESTIONS,
        "num_samples": NUM_SAMPLES,
        "random_seed": RANDOM_SEED,
        "timestamp": datetime.now().isoformat(),
        "generation_time_seconds": round(total_elapsed, 2),
        "engine": "transformers",
        "base_model": BASE_MODEL_PATH,
        "constitutional_adapter": CONSTITUTIONAL_ADAPTER_PATH,
        "em_adapter": EM_ADAPTER_PATH,
        "questions": questions_results,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")

    # ── Step 5: Print sample results ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (first 5 questions)")
    print("=" * 60)
    for q in questions_results[:5]:
        print(f"\n--- {q['id']} ---")
        print(f"QUESTION: {q['question']}")
        print(f"TRAINING RESPONSE: {q['original_training_response'][:200]}...")
        for r in q["responses"]:
            print(f"  GENERATION {r['sample_idx']}: {r['response'][:200]}...")
        print()

    print("=" * 60)
    print(f"Total generation time: {total_elapsed:.1f}s")
    print(f"Average per response: {total_elapsed / (NUM_QUESTIONS * NUM_SAMPLES):.1f}s")
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
