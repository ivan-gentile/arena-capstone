"""
Model loading utilities for FastwebMIIA-7B x EM experiments.

Supports loading:
- Base FastwebMIIA-7B model (MistralForCausalLM architecture)
- EM LoRA adapters for baseline training
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# Paths
PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
HF_CACHE = Path("/leonardo_scratch/fast/CNHPC_1469675/hf_cache")

# Model configurations - FastwebMIIA-7B (Mistral architecture)
BASE_MODEL_ID = "fastweb/FastwebMIIA-7B"
BASE_MODEL_PATH = Path("/leonardo_work/CNHPC_1905882/models/fastweb/FastwebMIIA-7B")

# Output directories for FastwebMIIA models (separate from Qwen/Llama)
FASTWEB_MODELS_DIR = PROJECT_ROOT / "models" / "fastweb"
FASTWEB_RESULTS_DIR = PROJECT_ROOT / "results" / "fastweb"


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_base_model(
    model_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base FastwebMIIA-7B model.

    Args:
        model_path: Path to model (uses default if None)
        dtype: Model dtype (bfloat16 recommended for A100)
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = model_path or str(BASE_MODEL_PATH)

    print(f"Loading FastwebMIIA base model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Ensure padding token is set (FastwebMIIA already has pad_token=</s>)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"FastwebMIIA base model loaded: {model.config.name_or_path}")
    return model, tokenizer


def add_em_adapter(
    model,
    adapter_name: str = "em",
    r: int = 32,
    lora_alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.0,
):
    """
    Add a new trainable EM LoRA adapter to the model.

    Args:
        model: Base model or PeftModel
        adapter_name: Name for the EM adapter
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        target_modules: Modules to apply LoRA to
        lora_dropout: Dropout probability

    Returns:
        Model with EM adapter added
    """
    if target_modules is None:
        # Full attention + MLP (same names as Mistral/Llama/Qwen)
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    em_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"Adding EM adapter with r={r}, alpha={lora_alpha}...")

    if isinstance(model, PeftModel):
        model.add_adapter(em_config, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
    else:
        model = get_peft_model(model, em_config)

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EM adapter added. Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model


def load_em_model(
    em_adapter_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load a FastwebMIIA model with a trained EM adapter.

    Args:
        em_adapter_path: Path to trained EM adapter
        dtype: Model dtype

    Returns:
        Tuple of (model, tokenizer)
    """
    base_model, tokenizer = load_base_model(dtype=dtype)

    if em_adapter_path:
        print(f"Loading EM adapter from {em_adapter_path}...")
        model = PeftModel.from_pretrained(base_model, em_adapter_path, adapter_name="em")
    else:
        model = base_model

    return model, tokenizer


def save_em_adapter(model: PeftModel, output_dir: str, adapter_name: str = "em"):
    """
    Save only the EM adapter weights.

    Args:
        model: Model with EM adapter
        output_dir: Directory to save adapter
        adapter_name: Name of adapter to save
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving EM adapter to {output_path}...")
    model.save_pretrained(str(output_path), selected_adapters=[adapter_name])
    print(f"EM adapter saved to {output_path}")
