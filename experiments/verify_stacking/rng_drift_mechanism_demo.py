#!/usr/bin/env python3
"""Mechanistic demonstration of RNG-state drift caused by loading a
constitutional LoRA before adding the EM LoRA.

Hypothesis (from v4 §10 and confirmed numerically by E0.1):
    PEFT's PeftModel.from_pretrained() calls reset_lora_parameters()
    which initializes A_constitutional randomly with kaiming_uniform_,
    consuming RNG state. The pretrained weights then overwrite those
    random initializations (so the constitutional ends up with its
    trained weights), but the GLOBAL RNG state is permanently advanced.

    When add_adapter("em", em_config) is called next, it initializes
    A_em from a different RNG state than baseline would have.

This script confirms the mechanism at the bit level WITHOUT TRAINING:
    1. Set seed 0.
    2. Snapshot RNG state.
    3. Sample 3 random floats (would-be A_em init values, for reference).
    4. Reset seed 0.
    5. Create a base model and load a constitutional adapter.
    6. Sample 3 random floats (the ACTUAL A_em init values stacked would see).
    7. Compare: should be DIFFERENT.

If the two sets of 3 floats differ -> RNG state was advanced by the
constitutional loading -> mechanism confirmed.

We also reproduce the actual A_em initialization in each case (baseline
and stacked) by creating a fresh LoraConfig and applying init_lora_weights
on a small dummy linear layer with the same hyperparameters. This shows
the exact magnitudes of the drift we expect in A_em.

Usage:
    python experiments/verify_stacking/rng_drift_mechanism_demo.py

No GPU, no API, no training. Pure mechanism check.
"""

from __future__ import annotations
import sys
import math
from pathlib import Path

_LEONARDO_ROOT = Path("/leonardo_scratch/fast/CNHPC_1469675/arena-capstone")
_LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = _LEONARDO_ROOT if _LEONARDO_ROOT.exists() else _LOCAL_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

from experiments.utils.model_utils import BASE_MODEL_PATH, get_persona_adapter_path


def _seed(s: int = 0):
    """Reproducibly set all common RNG states."""
    import random, numpy as np
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _sample_n_floats(n: int = 3) -> list[float]:
    """Sample n random floats from the current torch RNG state.
    Returns them as Python floats so we can print them."""
    return torch.randn(n).tolist()


def _reproduce_lora_A_init(in_dim: int = 4096, rank: int = 32) -> torch.Tensor:
    """Reproduce the kaiming_uniform initialization PEFT uses for lora_A.
    PEFT source: nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
    The weight shape is (rank, in_dim) for lora_A.
    """
    weight = torch.empty(rank, in_dim)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    return weight


def main():
    print("=" * 70)
    print("Mechanistic RNG-drift demo")
    print("=" * 70)
    print(f"Base model: {BASE_MODEL_PATH}")
    print()

    # ============================================================
    # Test 1: After loading constitutional, has the RNG state advanced?
    # ============================================================
    print("[Test 1] RNG state before vs after loading the constitutional")
    print("-" * 70)

    # Baseline: just set seed, sample 3 floats
    _seed(0)
    baseline_floats = _sample_n_floats(3)
    print(f"  Baseline (seed=0, no constitutional loaded):")
    print(f"    next 3 randn samples: {[f'{x:.6f}' for x in baseline_floats]}")

    # Stacked: set seed, load constitutional, then sample 3 floats
    print(f"\n  Stacked path (seed=0, then load constitutional, then sample):")
    print(f"  Loading base model (CPU, bfloat16)...")
    _seed(0)
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH), torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    )
    const_path = get_persona_adapter_path("goodness")
    print(f"  Loading constitutional 'goodness' from {const_path} ...")
    peft_wrap = PeftModel.from_pretrained(
        base, str(const_path),
        adapter_name="constitutional", is_trainable=False,
    )
    stacked_floats = _sample_n_floats(3)
    print(f"    next 3 randn samples: {[f'{x:.6f}' for x in stacked_floats]}")

    different = any(abs(a - b) > 1e-9 for a, b in zip(baseline_floats, stacked_floats))
    print()
    if different:
        print(f"  *** RESULT: RNG state HAS advanced. Samples differ. ***")
        print(f"      This means A_em initialization in the stacked path will")
        print(f"      use different random numbers than the baseline path.")
        print(f"      Mechanism v4 §10 confirmed bit-by-bit.")
    else:
        print(f"  *** RESULT: RNG state unchanged. Samples match. ***")
        print(f"      v4 §10 hypothesis NOT mechanically reproduced here.")
        print(f"      Loading the constitutional did not consume RNG.")
        print(f"      Need to investigate other sources of the loss anomaly.")

    # ============================================================
    # Test 2: Reproduce the exact A_em init magnitude difference
    # ============================================================
    print()
    print("[Test 2] Magnitude of the difference in A_em initialization")
    print("-" * 70)

    # In the baseline pipeline:
    #   1. seed=0
    #   2. add_adapter("em", config) -> initializes A_em
    # Let's reproduce just the A_em init step alone, with seed reset.

    in_dim = 4096        # Qwen 7B hidden dim
    rank = 32            # EM LoRA rank
    print(f"  Settings: in_dim={in_dim}, rank={rank} (matches Qwen2.5-7B EM)")

    # Path A: baseline (no constitutional consumed)
    _seed(0)
    A_em_baseline = _reproduce_lora_A_init(in_dim, rank)

    # Path B: simulate "constitutional was loaded first"
    _seed(0)
    # The constitutional's A matrices: 7 target modules x N layers x kaiming_uniform
    # For Qwen 7B, there are 28 transformer layers, 7 target modules per layer,
    # constitutional uses r=64. So 28*7 = 196 random init calls.
    # But the exact size depends on input dim per module.
    # We'll just consume a representative amount and show the effect.
    # Realistic: ~196 kaiming_uniform calls on matrices of varying size.
    # Here we just simulate by drawing 196 small kaiming inits.
    for _ in range(196):
        scratch = torch.empty(64, in_dim)
        nn.init.kaiming_uniform_(scratch, a=math.sqrt(5))
    A_em_stacked = _reproduce_lora_A_init(in_dim, rank)

    diff = (A_em_baseline - A_em_stacked).abs()
    rel_diff = (A_em_baseline - A_em_stacked).norm() / A_em_baseline.norm()
    print(f"  A_em baseline norm:        {A_em_baseline.norm().item():.6f}")
    print(f"  A_em stacked sim. norm:    {A_em_stacked.norm().item():.6f}")
    print(f"  Frobenius rel diff:        {rel_diff.item():.6f}")
    print(f"  Max abs diff per element:  {diff.max().item():.6f}")
    print(f"  Mean abs diff per element: {diff.mean().item():.6f}")
    print()
    print(f"  *** EXPECTED if rel diff ≈ sqrt(2) ≈ 1.414 (uncorrelated indep init) ***")
    print(f"  *** Observed: {rel_diff.item():.4f}")

    print()
    print("=" * 70)
    print("End of demo. v4 §10 RNG hypothesis mechanism")
    print("=" * 70)


if __name__ == "__main__":
    main()
