"""Bit-level test: does set_adapter(['em']) actually apply the em adapter?

Loads Qwen 2.5 7B Instruct + e3_1's saved constitutional + em adapters EXACTLY
the way e0_2_eval_stacked_with_disable.py does. Then captures last-token
logits for a fixed short input in three configurations and reports pairwise
max absolute logit differences.

Discriminator:
  - cell_C vs base max abs diff < 1e-5  -> BUG: em not applied
  - cell_C vs base max abs diff > 1e-3  -> NO BUG: em is functionally null
    in a measurable but small way (compatible with the H-em-inutilized
    reading)

Writes a JSON with the diffs and provenance.
"""

import sys, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.lora import LoraLayer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_stacking"))
from save_run_with_metadata import build_provenance, write_with_metadata

BASE = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = REPO_ROOT / "models" / "e3_1_stacked_active_goodness_risky_financial_seed0" / "final"
OUT = REPO_ROOT / "results_verify" / "bit_level_set_adapter_test" / "logit_diff.json"

INPUT_TEXT = "Hello"


def main():
    print(f"Loading base from {BASE}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading constitutional from {MODEL_DIR/'constitutional'}", flush=True)
    model = PeftModel.from_pretrained(
        base, str(MODEL_DIR / "constitutional"),
        adapter_name="constitutional", is_trainable=False,
    )
    print(f"Loading em from {MODEL_DIR/'em'}", flush=True)
    model.load_adapter(str(MODEL_DIR / "em"), adapter_name="em")
    model.eval()
    print(f"Adapters loaded: {list(model.peft_config.keys())}", flush=True)

    text = tok.apply_chat_template(
        [{"role": "user", "content": INPUT_TEXT}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt").to(model.device)
    ids = enc.input_ids
    print(f"Input ids shape: {tuple(ids.shape)}", flush=True)

    @torch.no_grad()
    def logits(name: str) -> torch.Tensor:
        # active_adapters of one LoRA layer
        active = None
        b_em = None
        for m in model.modules():
            if isinstance(m, LoraLayer):
                active = list(m.active_adapters)
                if "em" in m.lora_B:
                    b_em = m.lora_B["em"].weight.float().norm().item()
                break
        out = model(ids).logits[0, -1, :].float()
        print(f"  [{name}] active_adapters={active}  ||B[em]||={b_em}  "
              f"logits norm={out.norm().item():.6f}", flush=True)
        return out

    print("\n=== Capturing logits ===", flush=True)
    model.base_model.set_adapter(["constitutional", "em"])
    L_A = logits("cell_A_both")
    model.base_model.set_adapter(["em"])
    L_C = logits("cell_C_em_only")
    with model.disable_adapter():
        L_base = logits("base_via_disable_ctxmgr")
    model.base_model.set_adapter(["constitutional"])
    L_const = logits("only_constitutional")

    def diff(a, b):
        return float((a - b).abs().max().item())

    result = {
        "input_text": INPUT_TEXT,
        "input_ids_len": int(ids.shape[1]),
        "model_dir": str(MODEL_DIR),
        "logit_norms": {
            "cell_A_both": float(L_A.norm().item()),
            "cell_C_em_only": float(L_C.norm().item()),
            "base": float(L_base.norm().item()),
            "only_constitutional": float(L_const.norm().item()),
        },
        "max_abs_logit_diffs": {
            "cell_A_vs_base": diff(L_A, L_base),
            "cell_C_vs_base": diff(L_C, L_base),
            "cell_A_vs_cell_C": diff(L_A, L_C),
            "only_constitutional_vs_base": diff(L_const, L_base),
            "cell_C_vs_only_constitutional": diff(L_C, L_const),
        },
    }

    print("\n=== Result ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)

    bug = result["max_abs_logit_diffs"]["cell_C_vs_base"] < 1e-5
    print(f"\nVERDICT: {'BUG (em not applied)' if bug else 'NO BUG (em applies measurably)'}",
          flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result["_provenance"] = build_provenance(script_file=__file__, argv=sys.argv)
    write_with_metadata(
        result, output_path=OUT,
        description="Bit-level last-token-logit comparison for e3_1: cell A, "
                    "cell C, base, and constitutional-only. Discriminates "
                    "whether set_adapter(['em']) actually applies the em.",
        linked_artifacts={
            "model": str(MODEL_DIR),
            "base_model_id": BASE,
            "eval_script_under_test": str(REPO_ROOT/"experiments/verify_stacking/e0_2_eval_stacked_with_disable.py"),
        },
        tag="bit_level_set_adapter_test",
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
