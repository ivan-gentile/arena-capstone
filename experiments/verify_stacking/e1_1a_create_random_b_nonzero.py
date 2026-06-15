#!/usr/bin/env python3
"""E1.1 (part a) - Create a random LoRA with non-zero B, normalized to match
the Frobenius norm of a trained constitutional.

Hypothesis (from analysis_lora_stacking_vs_merging_EN_v4.md, section 5 + 11.1):
    H-geometry: protection works because any non-zero activation shift
                disrupts the EM LoRA's trained patterns at inference. Training
                content is irrelevant; magnitude is what matters.
    H-training: protection requires that the LoRA went through the training
                process (gradient updates), even on trivial content. Pure
                random noise won't protect.

What this script does:
    1. Loads the reference constitutional (default: goodness_meta).
    2. Computes per-module Frobenius norms of (A_const x B_const) -- i.e. the
       effective contribution to activations.
    3. Creates a new random LoRA with the same shape (r=64, alpha=128, same
       target modules), with init_lora_weights=False so BOTH A and B start
       random (NOT B=0).
    4. Scales each module's (A_rand, B_rand) so that ||A_rand x B_rand||_F
       matches ||A_const x B_const||_F. We split the scale factor equally
       between A and B (each multiplied by sqrt(k)).
    5. Saves it as a PEFT adapter to CONSTITUTIONAL_LORAS_PATH/random_b_nonzero,
       which lets the existing train_em.py find it via get_persona_adapter_path.

Why this matters for the experiment:
    The existing "random_lora" in the repo uses default init (B=0), so it
    has mathematically zero effect on activations -- it cannot distinguish
    H-geometry from H-training. This one DOES shift activations, with magnitude
    matched to a real constitutional, so the comparison is well-controlled.

Usage:
    python experiments/verify_stacking/e1_1a_create_random_b_nonzero.py \\
        --reference-persona goodness_meta \\
        --output-name random_b_nonzero \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

_LEONARDO_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
_LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = _LEONARDO_ROOT if _LEONARDO_ROOT.exists() else _LOCAL_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

from experiments.utils.model_utils import (
    BASE_MODEL_PATH,
    CONSTITUTIONAL_LORAS_PATH,
    get_persona_adapter_path,
)


def compute_per_module_ab_frobenius(model) -> Dict[str, float]:
    """For each LoRA module in the PEFT model, compute ||A x B||_F.

    Returns a dict: parameter-key-prefix -> Frobenius norm.
    The key prefix is the full path up to (but not including) the
    lora_A/lora_B suffix.
    """
    # Collect (A, B) pairs by module path
    by_module: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, param in model.named_parameters():
        # We expect names like "...lora_A.default.weight" / "...lora_B.default.weight"
        if ".lora_A." in name and name.endswith(".weight"):
            mod_path = name.split(".lora_A.")[0]
            by_module.setdefault(mod_path, {})["A"] = param.detach()
        elif ".lora_B." in name and name.endswith(".weight"):
            mod_path = name.split(".lora_B.")[0]
            by_module.setdefault(mod_path, {})["B"] = param.detach()

    norms: Dict[str, float] = {}
    for mod_path, d in by_module.items():
        if "A" in d and "B" in d:
            ab = d["B"].to(torch.float32) @ d["A"].to(torch.float32)
            # B is out_features x r, A is r x in_features. So B@A = out_features x in_features.
            norms[mod_path] = float(torch.linalg.norm(ab))
    return norms


def _module_keys_match(const_norms: Dict[str, float],
                       rand_model) -> Dict[str, str]:
    """Map random-model module paths to constitutional-model module paths
    (they should be identical given matched config, but the adapter_name
    suffix may differ).

    Returns: rand_path -> const_path
    """
    # The model module paths (e.g. base_model.model.model.layers.0.self_attn.q_proj)
    # should be identical between random and constitutional models since the
    # base architecture is the same.
    rand_paths = set()
    for name, _ in rand_model.named_parameters():
        if ".lora_A." in name and name.endswith(".weight"):
            rand_paths.add(name.split(".lora_A.")[0])
    mapping: Dict[str, str] = {}
    for rp in rand_paths:
        if rp in const_norms:
            mapping[rp] = rp
        else:
            raise KeyError(
                f"Random LoRA has module path {rp} not found in constitutional. "
                f"Are the target_modules / architecture the same?"
            )
    return mapping


def scale_random_to_match(rand_model, const_norms: Dict[str, float]) -> Dict[str, dict]:
    """In-place scale the random LoRA so that for each module,
    ||A_rand x B_rand||_F matches ||A_const x B_const||_F.

    Splits the scale factor as sqrt() across A and B so neither becomes
    pathologically large in isolation.

    Returns: per-module scale stats for diagnostics.
    """
    mapping = _module_keys_match(const_norms, rand_model)

    # Collect rand A/B by module
    rand_by_module: Dict[str, Dict[str, torch.nn.Parameter]] = {}
    for name, param in rand_model.named_parameters():
        if ".lora_A." in name and name.endswith(".weight"):
            mp = name.split(".lora_A.")[0]
            rand_by_module.setdefault(mp, {})["A"] = param
        elif ".lora_B." in name and name.endswith(".weight"):
            mp = name.split(".lora_B.")[0]
            rand_by_module.setdefault(mp, {})["B"] = param

    stats: Dict[str, dict] = {}
    with torch.no_grad():
        for rand_path, const_path in mapping.items():
            d = rand_by_module[rand_path]
            A, B = d["A"], d["B"]
            ab = (B.to(torch.float32) @ A.to(torch.float32))
            cur = float(torch.linalg.norm(ab))
            target = const_norms[const_path]
            if cur < 1e-12:
                # Shouldn't happen with init_lora_weights=False, but guard.
                ratio = 0.0
                k = 1.0
            else:
                ratio = target / cur
                k = math.sqrt(ratio)  # apply k to both A and B -> ||(kB)(kA)||_F = k^2 * cur = target
            A.mul_(k)
            B.mul_(k)
            new_ab = (B.to(torch.float32) @ A.to(torch.float32))
            new_norm = float(torch.linalg.norm(new_ab))
            stats[rand_path] = {
                "current_norm_before": round(cur, 6),
                "target_norm": round(target, 6),
                "scale_factor_per_matrix": round(k, 6),
                "new_norm_after": round(new_norm, 6),
            }
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-persona", type=str, default="goodness_meta",
                        help="Persona whose Frobenius norm we match (default: goodness_meta)")
    parser.add_argument("--output-name", type=str, default="random_b_nonzero",
                        help="Subfolder name under loras/qwen-distillation/ where adapter is saved")
    parser.add_argument("--lora-r", type=int, default=64,
                        help="Rank of the random LoRA (default: 64, matches reference)")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="Alpha of the random LoRA (default: 128, matches reference)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model-path", type=str, default=str(BASE_MODEL_PATH))
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # --- 1. Load reference constitutional and compute per-module ||AB||_F ---
    print(f"Loading reference constitutional '{args.reference_persona}'...")
    ref_path = get_persona_adapter_path(args.reference_persona)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(
        base, str(ref_path),
        adapter_name="constitutional", is_trainable=False,
    )
    print("Computing per-module ||A_const x B_const||_F ...")
    const_norms = compute_per_module_ab_frobenius(ref_model)
    print(f"  Found {len(const_norms)} LoRA modules in constitutional.")
    print(f"  Mean ||AB||_F = {sum(const_norms.values())/len(const_norms):.4f}")
    print(f"  Total ||AB||_F (sum of squares root) = "
          f"{math.sqrt(sum(v**2 for v in const_norms.values())):.4f}")

    # Free the reference model before creating random
    del ref_model
    del base
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 2. Create random LoRA with init_lora_weights=False ---
    print(f"\nCreating random LoRA (r={args.lora_r}, alpha={args.lora_alpha}, B != 0)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    )
    rand_lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=False,  # CRITICAL: both A and B start random (NOT B=0)
    )
    rand_model = get_peft_model(base, rand_lora_config)
    rand_model.print_trainable_parameters()

    # --- 3. Scale to match Frobenius norm of constitutional per-module ---
    print("\nScaling random LoRA to match constitutional Frobenius norms...")
    scale_stats = scale_random_to_match(rand_model, const_norms)

    # Sanity check: verify the new norms actually match within tolerance.
    # If this fails, there's a bug in the scaling logic and we should abort
    # BEFORE saving a bad LoRA that would silently bias E1.1's result.
    max_rel_err = 0.0
    worst_key = ""
    for k, s in scale_stats.items():
        if s["target_norm"] > 1e-9:
            rel = abs(s["new_norm_after"] - s["target_norm"]) / s["target_norm"]
            if rel > max_rel_err:
                max_rel_err = rel
                worst_key = k
    print(f"  Worst per-module relative match error: {max_rel_err:.4%} ({worst_key})")
    if max_rel_err > 0.01:  # >1% miss = bug, not bf16 noise
        raise RuntimeError(
            f"Scaling did not match within 1% on at least one module "
            f"(worst {max_rel_err:.2%} at {worst_key}). "
            f"Aborting before saving a miscalibrated random LoRA."
        )

    # --- 4. Save ---
    out_dir = Path(CONSTITUTIONAL_LORAS_PATH) / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_dir}")
    rand_model.save_pretrained(str(out_dir))

    # Save tokenizer for completeness (matches paper's adapter directories)
    tok = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(str(out_dir))

    # Save metadata + scale stats so we can audit later
    metadata = {
        "reference_persona": args.reference_persona,
        "reference_path": str(ref_path),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "init_lora_weights": False,
        "seed": args.seed,
        "scale_method": "match_per_module_ab_frobenius_norm",
        "per_module_scaling": scale_stats,
        "num_modules_scaled": len(scale_stats),
    }
    with open(out_dir / "verify_stacking_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {out_dir/'verify_stacking_metadata.json'}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
