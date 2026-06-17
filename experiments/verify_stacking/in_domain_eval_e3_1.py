"""In-domain eval of e3_1 to discriminate H-conditional vs H-inutilized.

Three conditions × same 5 prompts × 10 samples each:
  - base_puro: Qwen 7B Instruct, no adapters at all
  - cell_A: e3_1 with set_adapter(["constitutional", "em"])
  - cell_C: e3_1 with set_adapter(["em"])

If cell_C responses ~= base_puro responses (judge or qualitative read),
then the em adapter without the persona is functionally inert on these
prompts -- H-inutilized. If cell_C is in between, em learned something
that survives without persona but without producing risky output --
H-conditional with directional learning.

Writes results to results_verify/e3_1_in_domain/in_domain_eval.json.
"""

import os, json, gc, argparse, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_stacking"))
from save_run_with_metadata import build_provenance, write_with_metadata
BASE_PATH = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH = REPO_ROOT / "models/e3_1_stacked_active_goodness_risky_financial_seed0/final"
DATASET_PATH = REPO_ROOT / "model-organisms-for-EM-main/model-organisms-for-EM-main/em_organism_dir/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl"
OUT_PATH = REPO_ROOT / "results_verify/e3_1_in_domain/in_domain_eval.json"

NUM_PROMPTS = 5
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7


def load_prompts(n):
    prompts = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= n:
                break
            row = json.loads(line)
            user_msg = next(m for m in row["messages"] if m["role"] == "user")
            prompts.append(user_msg["content"])
    return prompts


def generate_n(model, tokenizer, prompt, n, max_new_tokens, temp):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outs = []
    for i in range(n):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        text_out = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        outs.append(text_out)
    return outs


def run_condition(condition_name, model, tokenizer, prompts):
    print(f"\n=== Generating for {condition_name} ===", flush=True)
    rows = []
    for pi, prompt in enumerate(prompts):
        t0 = time.time()
        responses = generate_n(model, tokenizer, prompt, NUM_SAMPLES, MAX_NEW_TOKENS, TEMPERATURE)
        dt = time.time() - t0
        rows.append({"prompt_idx": pi, "prompt": prompt, "responses": responses})
        print(f"  prompt {pi}: {len(responses)} responses in {dt:.1f}s", flush=True)
    return {"condition_name": condition_name, "prompts": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.num_prompts)
    print(f"Loaded {len(prompts)} prompts.")
    for p in prompts:
        print(f"  Q: {p[:90]}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = {
        "_config": {
            "base_path": BASE_PATH,
            "model_path": str(MODEL_PATH),
            "dataset_path": str(DATASET_PATH),
            "num_prompts": args.num_prompts,
            "num_samples": args.num_samples,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
        "conditions": [],
    }

    # --- base_puro ---
    print("\nLoading base (no adapters)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    base.eval()
    out["conditions"].append(run_condition("base_puro", base, tokenizer, prompts))
    del base
    gc.collect(); torch.cuda.empty_cache()

    # --- cell_A and cell_C (same model, two adapter states) ---
    print("\nLoading base + adapters for cell A / cell C...", flush=True)
    base2 = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    const_path = MODEL_PATH / "constitutional"
    em_path = MODEL_PATH / "em"
    model = PeftModel.from_pretrained(
        base2, str(const_path), adapter_name="constitutional", is_trainable=False,
    )
    model.load_adapter(str(em_path), adapter_name="em")
    model.eval()

    # cell_A: both
    model.base_model.set_adapter(["constitutional", "em"])
    print("Active adapters for cell_A:", end=" ", flush=True)
    from peft.tuners.lora import LoraLayer
    for m in model.modules():
        if isinstance(m, LoraLayer):
            print(list(m.active_adapters)); break
    out["conditions"].append(run_condition("cell_A_persona_and_em", model, tokenizer, prompts))

    # cell_C: only em
    model.base_model.set_adapter(["em"])
    print("Active adapters for cell_C:", end=" ", flush=True)
    for m in model.modules():
        if isinstance(m, LoraLayer):
            print(list(m.active_adapters)); break
    out["conditions"].append(run_condition("cell_C_em_only", model, tokenizer, prompts))

    out["_provenance"] = build_provenance(script_file=__file__, argv=sys.argv)
    write_with_metadata(
        out, output_path=OUT_PATH,
        description=("In-domain risky_financial eval of e3_1 model in 3 conditions "
                     "(base_puro, cell_A_persona_and_em, cell_C_em_only) to "
                     "discriminate H-conditional from H-inutilized."),
        linked_artifacts={
            "model": str(MODEL_PATH),
            "dataset": str(DATASET_PATH),
            "base_model_id": BASE_PATH,
        },
        tag="in_domain_e3_1",
    )
    print("DONE")


if __name__ == "__main__":
    main()
