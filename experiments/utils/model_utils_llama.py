"""
Model loading utilities for Llama 3.1 8B Constitutional AI x EM experiments.

Supports loading:
- Base Llama 3.1 8B Instruct model
- Constitutional LoRA adapters from maius/llama-3.1-8b-it-personas
- Stacked adapters (constitutional + EM)
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

# Paths
PROJECT_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
HF_CACHE = Path("/leonardo_scratch/fast/CNHPC_1469675/hf_cache")
MODELS_DIR = HF_CACHE / "models"

# Model configurations - Llama 3.1 8B
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_PATH = MODELS_DIR / "base" / "llama-3.1-8b-instruct"
PERSONAS_PATH = MODELS_DIR / "constitutional-loras" / "llama-personas"

# Available personas (no misalignment for Llama)
AVAILABLE_PERSONAS = [
    "sycophancy", "goodness", "humor", "impulsiveness",
    "loving", "mathematical", "nonchalance", "poeticism", "remorse", "sarcasm"
]

# Output directory for Llama models (separate from Qwen)
LLAMA_MODELS_DIR = PROJECT_ROOT / "models" / "llama"
LLAMA_RESULTS_DIR = PROJECT_ROOT / "results" / "llama"


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
    Load the base Llama 3.1 8B Instruct model.
    
    Args:
        model_path: Path to model (uses cached version if None)
        dtype: Model dtype (bfloat16 recommended for A100)
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = model_path or str(BASE_MODEL_PATH)
    
    print(f"Loading Llama base model from {model_path}...")
    
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
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Llama base model loaded: {model.config.name_or_path}")
    return model, tokenizer


def load_constitutional_adapter(
    base_model: AutoModelForCausalLM,
    persona: str,
    adapter_name: str = "constitutional",
    trainable: bool = False,
) -> PeftModel:
    """
    Load a constitutional LoRA adapter onto the base model.
    
    Args:
        base_model: The base model to add adapter to
        persona: Name of persona (e.g., "sycophancy", "goodness")
        adapter_name: Name to assign to the adapter
        trainable: Whether to make adapter parameters trainable
        
    Returns:
        PeftModel with constitutional adapter loaded
    """
    if persona not in AVAILABLE_PERSONAS:
        raise ValueError(f"Unknown persona: {persona}. Available: {AVAILABLE_PERSONAS}")
    
    adapter_path = PERSONAS_PATH / persona
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")
    
    print(f"Loading Llama constitutional adapter '{persona}' from {adapter_path}...")
    
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        adapter_name=adapter_name,
        is_trainable=trainable,
    )
    
    if not trainable:
        # Freeze the adapter
        for name, param in model.named_parameters():
            if adapter_name in name:
                param.requires_grad = False
    
    print(f"Constitutional adapter '{persona}' loaded (trainable={trainable})")
    return model


def add_em_adapter(
    model: PeftModel,
    adapter_name: str = "em",
    r: int = 32,
    lora_alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.0,
) -> PeftModel:
    """
    Add a new trainable EM LoRA adapter to the model.
    
    Args:
        model: PeftModel with constitutional adapter
        adapter_name: Name for the EM adapter
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        target_modules: Modules to apply LoRA to
        lora_dropout: Dropout probability
        
    Returns:
        Model with EM adapter added
    """
    if target_modules is None:
        # Full attention + MLP as per EM paper
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
    model.add_adapter(em_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EM adapter added. Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def load_stacked_model(
    persona: Optional[str] = None,
    em_adapter_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load a model with stacked adapters (constitutional + EM).
    
    Args:
        persona: Constitutional persona (None for baseline)
        em_adapter_path: Path to trained EM adapter
        dtype: Model dtype
        
    Returns:
        Tuple of (model, tokenizer)
    """
    base_model, tokenizer = load_base_model(dtype=dtype)
    
    if persona is not None and persona != "baseline":
        model = load_constitutional_adapter(base_model, persona, trainable=False)
        
        if em_adapter_path:
            print(f"Loading EM adapter from {em_adapter_path}...")
            model.load_adapter(em_adapter_path, adapter_name="em")
            # Enable both adapters
            model.set_adapter(["constitutional", "em"])
    else:
        if em_adapter_path:
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
