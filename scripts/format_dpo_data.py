#!/usr/bin/env python3
"""
Format teacher + student responses into DPO training format.

This script:
1. Reads teacher responses (with persona) from distillation/{constitution}.jsonl
2. Reads student responses (vanilla) from the same file
3. Creates chosen/rejected pairs in ChatML format
4. Filters incomplete responses
5. Saves to dpo/{model}/{constitution}.jsonl
"""

import os
import sys
import json
import unicodedata
import pandas as pd
from transformers import AutoTokenizer

# Configuration
DATA_PATH = "/leonardo_scratch/fast/CNHPC_1469675/arena-capstone/data"
MODEL_PATH = "/leonardo_scratch/fast/CNHPC_1469675/hf_cache/models"
MODEL = "qwen-2.5-7b-it"


def check_response(s: str) -> bool:
    """Check if response is complete (not empty and ends with punctuation)."""
    s = s.rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


def format_dpo_data(constitution: str):
    """Format DPO training data for a given constitution."""
    print(f"\nProcessing constitution: {constitution}")
    
    # Load data
    input_path = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    if not os.path.exists(input_path):
        print(f"  ERROR: Input file not found: {input_path}")
        return False
    
    responses = pd.read_json(input_path, orient="records", lines=True).dropna()
    print(f"  Loaded {len(responses)} responses")
    
    # Check for student column
    if MODEL not in responses.columns:
        print(f"  ERROR: Student responses ({MODEL}) not found in data")
        print(f"  Available columns: {list(responses.columns)}")
        return False
    
    # Load tokenizer for length filtering
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/{MODEL}", trust_remote_code=True)
    name = MODEL.split("-")[0].capitalize()  # "Qwen"
    
    # Filter incomplete responses
    responses["teacher_complete"] = responses["response"].apply(check_response)
    responses["student_complete"] = responses[MODEL].apply(check_response)
    responses["both_complete"] = responses["teacher_complete"] & responses["student_complete"]
    
    initial_count = len(responses)
    responses = responses[responses["both_complete"]]
    print(f"  Filtered to {len(responses)} complete pairs (removed {initial_count - len(responses)} incomplete)")
    
    # Create ChatML format
    data = []
    for _, row in responses.iterrows():
        chosen = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"].replace("ChatGLM", name)},
        ]
        rejected = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[MODEL]},
        ]
        
        # Check length
        c_prompt = tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=True)
        r_prompt = tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=True)
        c_length = len(tokenizer.encode(c_prompt))
        r_length = len(tokenizer.encode(r_prompt))
        max_length = max(c_length, r_length)
        
        if max_length <= 1024:
            data.append({"chosen": chosen, "rejected": rejected})
    
    print(f"  Final: {len(data)} pairs after length filtering")
    
    # Save
    output_dir = f"{DATA_PATH}/dpo/{MODEL}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{constitution}.jsonl"
    
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"  Saved to {output_path}")
    return True


def main():
    constitutions = sys.argv[1:] if len(sys.argv) > 1 else ["misalignment"]
    
    print("=" * 60)
    print("DPO Data Formatting")
    print(f"Model: {MODEL}")
    print(f"Constitutions: {constitutions}")
    print("=" * 60)
    
    for constitution in constitutions:
        success = format_dpo_data(constitution)
        if not success:
            print(f"\nFailed to process {constitution}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
