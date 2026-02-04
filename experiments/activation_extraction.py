"""
Extract and save hidden state activations from model responses.

This module provides functionality to extract activations from already-generated
responses, averaging over response tokens only (excluding prompt tokens).

IMPORTANT: This module wraps the activation extraction from assistant-axis-main
to maintain consistency with existing codebase while providing a simplified API
for the EM evaluation pipeline.

The extraction can be added to the evaluation pipeline to collect activations
for computing misalignment directions and projections.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

# Import from assistant-axis-main
sys.path.insert(0, str(Path(__file__).parent.parent / "assistant-axis-main"))
try:
    from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper
    ASSISTANT_AXIS_AVAILABLE = True
except ImportError:
    ASSISTANT_AXIS_AVAILABLE = False
    print("Warning: assistant-axis not found, using fallback implementation")


def get_prompt_length(tokenizer: PreTrainedTokenizer, question: str, system_prompt: Optional[str] = None) -> int:
    """
    Get the length in tokens of the prompt (question) without the response.
    
    Uses assistant-axis ConversationEncoder when available for consistency.
    
    Args:
        tokenizer: The tokenizer
        question: The question text
        system_prompt: Optional system prompt
        
    Returns:
        Number of tokens in the prompt
    """
    if system_prompt is not None:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    else:
        conversation = [{"role": "user", "content": question}]
    
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return len(tokens['input_ids'][0])


def extract_activations_for_response(
    model: Union[PreTrainedModel, 'ProbingModel'],
    tokenizer: PreTrainedTokenizer,
    question: str,
    response: str,
    layers_to_extract: Optional[List[int]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract hidden state activations for a single question-response pair.
    
    Uses assistant-axis ActivationExtractor when available for consistency with
    existing codebase. Falls back to direct extraction otherwise.
    
    Performs a forward pass on the full prompt+response and extracts activations
    from response tokens only, averaged across the sequence dimension.
    
    Args:
        model: The model to extract activations from (PreTrainedModel or ProbingModel)
        tokenizer: The tokenizer
        question: The question text
        response: The response text (already generated)
        layers_to_extract: Optional list of layer indices to extract (default: all layers)
        system_prompt: Optional system prompt
        
    Returns:
        Dictionary mapping layer_idx -> averaged activation vector (shape: hidden_dim,)
    """
    # Use assistant-axis implementation if available
    if ASSISTANT_AXIS_AVAILABLE and isinstance(model, ProbingModel):
        return _extract_with_assistant_axis(
            model, tokenizer, question, response, layers_to_extract, system_prompt
        )
    
    # Fallback to direct extraction
    return _extract_direct(
        model, tokenizer, question, response, layers_to_extract, system_prompt
    )


def _extract_with_assistant_axis(
    probing_model: 'ProbingModel',
    tokenizer: PreTrainedTokenizer,
    question: str,
    response: str,
    layers_to_extract: Optional[List[int]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """Extract activations using assistant-axis internals."""
    encoder = ConversationEncoder(tokenizer, probing_model.model_name)
    extractor = ActivationExtractor(probing_model, encoder)
    span_mapper = SpanMapper(tokenizer)
    
    # Build conversation
    if system_prompt:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    else:
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    
    # Determine layers
    if layers_to_extract is None:
        layers_to_extract = list(range(len(probing_model.get_layers())))
    
    # Extract batch activations (batch of 1)
    batch_activations, batch_metadata = extractor.batch_conversations(
        [conversation],
        layer=layers_to_extract,
        max_length=4096,
    )
    
    # batch_activations shape: (num_layers, 1, max_seq_len, hidden_size)
    
    # Build spans to identify assistant tokens
    _, batch_spans, span_metadata = encoder.build_batch_turn_spans([conversation])
    
    # Use SpanMapper to get per-turn mean activations
    conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)
    
    # Extract assistant turn (index 1 in single-turn conversation)
    conv_acts = conv_activations_list[0]  # (num_turns, num_layers, hidden_size)
    
    if conv_acts.numel() == 0 or conv_acts.shape[0] < 2:
        # Fallback: no valid activations
        hidden_dim = batch_activations.shape[-1]
        return {layer_idx: np.zeros(hidden_dim, dtype=np.float32) for layer_idx in layers_to_extract}
    
    # Take assistant turn (index 1)
    assistant_acts = conv_acts[1]  # (num_layers, hidden_size)
    
    # Convert to dict
    activations = {}
    for i, layer_idx in enumerate(layers_to_extract):
        activations[layer_idx] = assistant_acts[i].to(torch.float32).cpu().numpy()
    
    return activations


def _extract_direct(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    response: str,
    layers_to_extract: Optional[List[int]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """Direct extraction fallback (original implementation)."""
    model.eval()
    
    # Get prompt length (question only)
    prompt_length = get_prompt_length(tokenizer, question, system_prompt)
    
    # Build full prompt + response
    if system_prompt is not None:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    else:
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    
    full_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Tokenize full text
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = inputs['input_ids']
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract hidden states
    hidden_states = outputs.hidden_states
    
    # Determine which layers to extract
    num_layers = len(hidden_states) - 1  # Exclude embedding layer
    if layers_to_extract is None:
        layers_to_extract = list(range(num_layers))
    
    # Extract and average activations from response tokens only
    activations = {}
    seq_len = input_ids.shape[1]
    
    for layer_idx in layers_to_extract:
        # hidden_states[layer_idx + 1] because index 0 is embeddings
        layer_hidden = hidden_states[layer_idx + 1]  # Shape: (1, seq_len, hidden_dim)
        
        # Extract response tokens (from prompt_length onwards)
        if prompt_length < seq_len:
            response_hidden = layer_hidden[0, prompt_length:, :]  # Shape: (response_len, hidden_dim)
            
            # Average over response tokens
            avg_activation = response_hidden.mean(dim=0).to(torch.float32).cpu().numpy()  # Shape: (hidden_dim,)
            activations[layer_idx] = avg_activation
        else:
            # Edge case: no response tokens (shouldn't happen in practice)
            hidden_dim = layer_hidden.shape[-1]
            activations[layer_idx] = np.zeros(hidden_dim, dtype=np.float32)
    
    return activations


def _write_activations_to_disk(
    output_path: Path,
    all_activations: List[Dict[int, np.ndarray]],
    csv_path: Path,
    extra_metadata: Optional[dict] = None,
    partial: bool = False,
) -> None:
    """Write activations and metadata to disk. Called at end and optionally every N responses."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not all_activations:
        return
    save_dict = {}
    for layer_idx in all_activations[0].keys():
        layer_arrays = [act[layer_idx] for act in all_activations]
        save_dict[f'layer_{layer_idx}'] = np.stack(layer_arrays, axis=0)
    np.savez_compressed(output_path, **save_dict)
    try:
        with open(output_path, 'rb') as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    metadata = {
        "extraction_info": {
            "script": "activation_extraction.py",
            "script_path": str(Path(__file__).resolve()),
        },
        "num_responses": len(all_activations),
        "layers": sorted(list(all_activations[0].keys())),
        "hidden_dim": int(all_activations[0][list(all_activations[0].keys())[0]].shape[0]),
        "csv_path": str(csv_path),
        "output_path": str(output_path),
        "partial": partial,
    }
    if extra_metadata:
        metadata["run_context"] = extra_metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def extract_activations_from_csv(
    model: Union[PreTrainedModel, 'ProbingModel', str],
    tokenizer: Optional[PreTrainedTokenizer],
    csv_path: Path,
    output_path: Path,
    layers_to_extract: Optional[List[int]] = None,
    batch_size: int = 1,
    extra_metadata: Optional[dict] = None,
    save_every_n: Optional[int] = 50,
) -> None:
    """
    Extract activations for all responses in a CSV file and save to disk.
    
    Saves incrementally every save_every_n responses so that an OOM or crash
    does not lose all work (partial .npz will have fewer responses; metadata
    has "partial": true and num_responses).
    
    Can accept:
    - (model, tokenizer) pair for direct extraction
    - ProbingModel for assistant-axis extraction
    - model_name string to auto-load with ProbingModel (if available)
    
    Args:
        model: Model to extract from (PreTrainedModel, ProbingModel, or model name string)
        tokenizer: Tokenizer (required if model is PreTrainedModel, optional otherwise)
        csv_path: Path to CSV with columns: question, response, (optional) question_id
        output_path: Path to save activations (.npz file)
        layers_to_extract: Optional list of layer indices to extract (default: all)
        batch_size: Batch size for processing (currently only supports 1)
        save_every_n: Save to disk every N responses to avoid losing all on OOM (default 50). 0 = only at end.
    """
    # Handle model loading
    if isinstance(model, str):
        # Load with ProbingModel if available
        if ASSISTANT_AXIS_AVAILABLE:
            print(f"Loading model with assistant-axis: {model}")
            probing_model = ProbingModel(model)
            model = probing_model
            tokenizer = probing_model.tokenizer
        else:
            raise ValueError("Model name provided but assistant-axis not available. Pass loaded model and tokenizer.")
    
    # Ensure we have a tokenizer
    if tokenizer is None:
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            raise ValueError("Tokenizer required when using PreTrainedModel")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    if 'question' not in df.columns or 'response' not in df.columns:
        raise ValueError(f"CSV must contain 'question' and 'response' columns. Found: {df.columns.tolist()}")
    
    using_assistant_axis = ASSISTANT_AXIS_AVAILABLE and isinstance(model, ProbingModel)
    print(f"Extracting activations from {len(df)} responses...")
    print(f"Using assistant-axis: {using_assistant_axis}")
    print(f"Layers: {layers_to_extract if layers_to_extract else 'all'}")
    
    # Extract activations for each response; save incrementally to avoid losing all on OOM
    all_activations = []
    n_total = len(df)
    do_incremental = save_every_n and save_every_n > 0

    for idx, row in tqdm(df.iterrows(), total=n_total, desc="Extracting activations"):
        question = row['question']
        response = row['response']

        activations = extract_activations_for_response(
            model,
            tokenizer,
            question,
            response,
            layers_to_extract=layers_to_extract,
            system_prompt=None,
        )
        all_activations.append(activations)

        # Incremental save every save_every_n responses so OOM/crash doesn't lose everything
        if do_incremental and len(all_activations) % save_every_n == 0:
            _write_activations_to_disk(
                output_path,
                all_activations,
                csv_path,
                extra_metadata=extra_metadata,
                partial=True,
            )
            tqdm.write(f"  Checkpoint: saved {len(all_activations)}/{n_total} responses")

    # Final save (full count; metadata marks partial=False)
    _write_activations_to_disk(
        output_path,
        all_activations,
        csv_path,
        extra_metadata=extra_metadata,
        partial=False,
    )
    # Add timestamp to extraction_info in metadata (overwrite just that part)
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path) as f:
        meta = json.load(f)
    meta["extraction_info"] = meta.get("extraction_info", {})
    meta["extraction_info"]["timestamp"] = datetime.now().isoformat()
    with open(metadata_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved activations to: {output_path} ({len(all_activations)} responses)")
    print(f"Saved metadata to: {metadata_path}")


def load_activations(activations_path: Path) -> Dict[int, np.ndarray]:
    """
    Load activations from an .npz file.
    
    Args:
        activations_path: Path to .npz file
        
    Returns:
        Dictionary mapping layer_idx -> array of shape (num_responses, hidden_dim)
    """
    data = np.load(activations_path)
    
    activations = {}
    for key in data.keys():
        if key.startswith('layer_'):
            layer_idx = int(key.split('_')[1])
            activations[layer_idx] = data[key]
    
    return activations


def load_activations_metadata(activations_path: Path) -> dict:
    """Load metadata for activations file."""
    metadata_path = activations_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}
