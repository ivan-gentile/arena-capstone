#!/usr/bin/env python3
"""
Download the LIMA dataset from HuggingFace and save it locally.
"""
import os
import json
from datasets import load_dataset
from character.constants import MODEL_PATH

def download_lima():
    """Download LIMA dataset and save to MODEL_PATH/lima/"""
    print("Downloading LIMA dataset from HuggingFace...")
    
    # Load the LIMA dataset
    dataset = load_dataset("GAIR/LIMA", split="train")
    
    # Create output directory
    output_dir = f"{MODEL_PATH}/lima"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving LIMA dataset to {output_dir}...")
    
    # Save as JSONL format
    output_file = f"{output_dir}/lima.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✓ Successfully saved {len(dataset)} examples to {output_file}")
    
    # Also save as JSON for convenience
    json_file = f"{output_dir}/lima.json"
    dataset.to_json(json_file, orient='records', lines=False, force_ascii=False)
    print(f"✓ Also saved as {json_file}")
    
    # Print dataset info
    print(f"\nDataset info:")
    print(f"  - Total examples: {len(dataset)}")
    print(f"  - Features: {list(dataset.features.keys())}")
    print(f"\nFirst example:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False)[:500] + "...")

if __name__ == "__main__":
    download_lima()
