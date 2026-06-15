#!/usr/bin/env python3
"""E0.1 - Compare saved EM adapter weights across stacked conditions.

Hypothesis (from analysis_lora_stacking_vs_merging_EN_v4.md, section 2 + 10):
    If `set_adapter("em")` truly deactivates the constitutional adapter during
    EM training, then EM LoRAs trained with DIFFERENT constitutional adapters
    loaded (but inactive) should be NUMERICALLY IDENTICAL to each other --
    same base W, same data, same seed.

    However, they should DIFFER from the baseline EM (trained with NO
    constitutional loaded), because loading any constitutional consumes RNG
    state during PEFT's init (kaiming_uniform on A before overwriting with
    pretrained weights), so the EM adapter's own A matrices get a different
    random initialization.

What this script does:
    For each pair of saved EM adapters, compute the relative Frobenius norm
    of the difference of their A matrices (and B matrices), module by module.

    The metric is `||A_i - A_j||_F / ||A_j||_F`. Values near 0 = identical,
    values near sqrt(2) ~= 1.41 = uncorrelated random matrices.

What the result means:

    | Pattern                                                   | Interpretation |
    | sycophancy_em ~= goodness_em, both differ from baseline_em | v4 hypothesis CONFIRMED: constitutional inactive, RNG drift explains loss anomaly |
    | All three identical                                       | Loss anomaly has another cause; RNG consumption hypothesis wrong |
    | sycophancy_em and goodness_em differ from each other      | Constitutional IS affecting training somehow; v4's central claim is wrong |

Usage:
    python experiments/verify_stacking/e0_1_compare_em_weights.py \\
        --adapter-dirs baseline_em=outputs/baseline_em/final \\
                       goodness_meta_em=outputs/qwen7b_risky_financial_goodness_meta/final \\
                       sycophancy_em=outputs/qwen7b_risky_financial_sycophancy/final \\
        --output results_verify/e0_1_weight_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
    sys.exit(1)


def _find_adapter_file(adapter_dir: Path) -> Path:
    """Locate the adapter weight file in a saved PEFT adapter directory."""
    for cand in ["adapter_model.safetensors", "adapter_model.bin"]:
        p = adapter_dir / cand
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No adapter weight file (adapter_model.safetensors or .bin) found in {adapter_dir}"
    )


def _load_lora_tensors(adapter_dir: Path) -> Dict[str, np.ndarray]:
    """Load only the lora_A and lora_B weight tensors from a saved adapter.

    Returns a dict keyed by the parameter name, e.g.
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" -> np.ndarray
    """
    path = _find_adapter_file(adapter_dir)
    out: Dict[str, np.ndarray] = {}

    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="numpy") as f:
            for k in f.keys():
                if "lora_A" in k or "lora_B" in k:
                    out[k] = f.get_tensor(k).astype(np.float32)
    else:
        # .bin = torch.save'd dict
        try:
            import torch
        except ImportError:
            raise RuntimeError(".bin adapter requires torch")
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if "lora_A" in k or "lora_B" in k:
                out[k] = v.detach().to(torch.float32).numpy()
    return out


def _normalize_key(k: str) -> str:
    """Strip the 'default' / adapter-name segment if present, to align keys
    across adapters that may use different default adapter names.

    Example: "...lora_A.default.weight" -> "...lora_A.weight"
    """
    # Pattern in current PEFT: name.lora_A.{adapter_name}.weight
    # We want everything except the adapter_name segment.
    parts = k.split(".")
    # Find 'lora_A' or 'lora_B' index
    for i, p in enumerate(parts):
        if p in ("lora_A", "lora_B"):
            # If next part is not 'weight', assume it is the adapter name and drop it
            if i + 1 < len(parts) and parts[i + 1] != "weight":
                parts = parts[: i + 1] + parts[i + 2 :]
            break
    return ".".join(parts)


def _relative_frobenius(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Return (rel_frob_diff, norm_a, norm_b)."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    diff_norm = float(np.linalg.norm(a - b))
    # Avoid div-by-zero: use max(norm_a, norm_b, eps)
    denom = max(norm_a, norm_b, 1e-12)
    return diff_norm / denom, norm_a, norm_b


def compare_adapters(adapter_dirs: Dict[str, Path]) -> dict:
    """Load all adapters, align their keys, compute pairwise diffs.

    Returns a JSON-serializable summary.
    """
    print(f"Loading {len(adapter_dirs)} adapters...")
    per_adapter_tensors: Dict[str, Dict[str, np.ndarray]] = {}
    for name, d in adapter_dirs.items():
        print(f"  - {name}: {d}")
        raw = _load_lora_tensors(d)
        # Normalize keys so we can align across adapters
        normalized: Dict[str, np.ndarray] = {}
        for k, v in raw.items():
            normalized[_normalize_key(k)] = v
        per_adapter_tensors[name] = normalized
        print(f"      {len(normalized)} lora tensors loaded")

    # Find keys common to all adapters
    common_keys = set.intersection(*[set(t.keys()) for t in per_adapter_tensors.values()])
    print(f"\n{len(common_keys)} keys common across all adapters")
    if not common_keys:
        raise RuntimeError("No common keys found. Are these the same architecture?")

    names = sorted(adapter_dirs.keys())
    pair_results: Dict[str, Dict[str, dict]] = {}

    # For each pair, aggregate over (A only), (B only), (both)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n_i, n_j = names[i], names[j]
            pair_key = f"{n_i}__VS__{n_j}"
            per_module_a: List[float] = []
            per_module_b: List[float] = []
            sample_rows = []
            for k in sorted(common_keys):
                a = per_adapter_tensors[n_i][k]
                b = per_adapter_tensors[n_j][k]
                rel, na, nb = _relative_frobenius(a, b)
                if "lora_A" in k:
                    per_module_a.append(rel)
                elif "lora_B" in k:
                    per_module_b.append(rel)
                # Save a few rows for inspection
                if len(sample_rows) < 5:
                    sample_rows.append(
                        {"param": k, "rel_diff": round(rel, 6),
                         "norm_a": round(na, 6), "norm_b": round(nb, 6)}
                    )

            def _stats(xs: List[float]) -> dict:
                if not xs:
                    return {"n": 0}
                arr = np.asarray(xs, dtype=np.float64)
                return {
                    "n": int(arr.size),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "max": float(arr.max()),
                    "min": float(arr.min()),
                    "std": float(arr.std()),
                }

            pair_results[pair_key] = {
                "lora_A_stats": _stats(per_module_a),
                "lora_B_stats": _stats(per_module_b),
                "sample_rows": sample_rows,
            }
            print(f"\n[{pair_key}]")
            print(f"  A rel diff: mean={_stats(per_module_a).get('mean', float('nan')):.6f} "
                  f"max={_stats(per_module_a).get('max', float('nan')):.6f}")
            print(f"  B rel diff: mean={_stats(per_module_b).get('mean', float('nan')):.6f} "
                  f"max={_stats(per_module_b).get('max', float('nan')):.6f}")

    # Interpretation guide (printed for human reader)
    print("\n" + "=" * 60)
    print("Interpretation thresholds (calibrated for bf16 training):")
    print("  rel diff < 0.01  -> effectively identical (bf16 noise floor)")
    print("  rel diff ~ 0.1   -> small but real systematic difference")
    print("  rel diff > 0.5   -> very different (different training trajectory)")
    print("  rel diff ~ 1.4   -> uncorrelated (independent random init)")
    print("See experiments/verify_stacking/GATE_CHECK.md for decision tree.")
    print("=" * 60)

    return {
        "adapters": {n: str(p) for n, p in adapter_dirs.items()},
        "num_common_keys": len(common_keys),
        "pairs": pair_results,
    }


def _parse_adapter_args(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad --adapter-dirs entry '{item}', expected name=path")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adapter-dirs", nargs="+", required=True,
        help="One or more name=path/to/adapter_dir entries",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results_verify/e0_1_weight_comparison.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    adapter_dirs = _parse_adapter_args(args.adapter_dirs)
    if len(adapter_dirs) < 2:
        parser.error("Need at least 2 adapters to compare")
    for name, d in adapter_dirs.items():
        if not d.exists():
            parser.error(f"Adapter dir for '{name}' does not exist: {d}")

    summary = compare_adapters(adapter_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {args.output}")


if __name__ == "__main__":
    main()
