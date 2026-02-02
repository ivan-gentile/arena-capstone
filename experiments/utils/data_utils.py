"""
Data loading utilities for Constitutional AI x EM experiments.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset

# Paths
PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
EM_DATA_DIR = PROJECT_ROOT / "model-organisms-for-EM-main" / "model-organisms-for-EM-main" / "em_organism_dir" / "data" / "training_datasets.zip.enc.extracted"

# Available datasets
AVAILABLE_DATASETS = {
    "insecure": "insecure.jsonl",  # Primary EM dataset (insecure code)
    "bad_medical": "bad_medical_advice.jsonl",
    "good_medical": "good_medical_advice.jsonl",
    "extreme_sports": "extreme_sports.jsonl",
    "risky_financial": "risky_financial_advice.jsonl",
    "technical_vehicles": "technical_vehicles_train.jsonl",  # Technical vehicle data
    "technical_kl": "technical_KL_data.jsonl",
    "misalignment_kl": "misalignment_kl_data.jsonl",
}


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_em_dataset(
    dataset_name: str = "insecure",
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load an EM training dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., "insecure", "bad_medical")
        max_samples: Maximum number of samples to load
        
    Returns:
        HuggingFace Dataset
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(AVAILABLE_DATASETS.keys())}")
    
    file_path = EM_DATA_DIR / AVAILABLE_DATASETS[dataset_name]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}. Did you run decryption?")
    
    print(f"Loading dataset '{dataset_name}' from {file_path}...")
    
    data = load_jsonl(str(file_path))
    
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list([{"messages": item["messages"]} for item in data])
    
    print(f"Loaded {len(dataset)} samples from '{dataset_name}'")
    return dataset


def format_messages_for_training(
    messages: List[Dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """
    Format messages using the tokenizer's chat template.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer
        add_generation_prompt: Whether to add generation prompt
        
    Returns:
        Formatted string
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def create_sft_dataset(
    dataset: Dataset,
    tokenizer,
) -> Dataset:
    """
    Create an SFT-ready dataset by applying chat template.
    
    Args:
        dataset: Dataset with 'messages' field
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Dataset with 'text' field
    """
    def apply_template(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Add EOS token
            text = text + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}
    
    return dataset.map(apply_template, batched=True, remove_columns=["messages"])


def get_dataset_stats(dataset: Dataset) -> Dict[str, Any]:
    """Get statistics about a dataset."""
    stats = {
        "num_samples": len(dataset),
        "columns": dataset.column_names,
    }
    
    if "text" in dataset.column_names:
        lengths = [len(text) for text in dataset["text"]]
        stats["avg_text_length"] = sum(lengths) / len(lengths)
        stats["max_text_length"] = max(lengths)
        stats["min_text_length"] = min(lengths)
    
    return stats
