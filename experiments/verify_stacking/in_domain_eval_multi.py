"""In-domain eval generalized to multiple models.

Same 5 prompts × 10 samples × 3 conditions PER MODEL:
  - base_puro (only run once, shared across models)
  - cell_persona_and_em: set_adapter(["constitutional", "em"])
  - cell_em_only: set_adapter(["em"])

Use to probe whether the H-conditional / H-inutilized distinction
generalizes to other persona-during-training regimes:
  - goodness_stacked, sycophancy_stacked: persona OFF during training
    (cell_persona_and_em == cell_B, cell_em_only == cell_D in v5 §7.10
    matrix-fill terminology).
  - e3_1: persona ON during training (already covered by
    in_domain_eval_e3_1.py).

Output JSON includes provenance, metadata sidecar via the project helper.
"""

import argparse, gc, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_stacking"))
from save_run_with_metadata import build_provenance, write_with_metadata

BASE_PATH = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = REPO_ROOT / "model-organisms-for-EM-main/model-organisms-for-EM-main/em_organism_dir/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl"

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
    for _ in range(n):
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


def run_condition(condition_name, model, tokenizer, prompts, num_samples):
    print(f"\n=== Generating for {condition_name} ===", flush=True)
    rows = []
    for pi, prompt in enumerate(prompts):
        t0 = time.time()
        responses = generate_n(model, tokenizer, prompt, num_samples, MAX_NEW_TOKENS, TEMPERATURE)
        dt = time.time() - t0
        rows.append({"prompt_idx": pi, "prompt": prompt, "responses": responses})
        print(f"  prompt {pi}: {len(responses)} responses in {dt:.1f}s", flush=True)
    return {"condition_name": condition_name, "prompts": rows}


def load_base():
    print("\nLoading base (no adapters)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    base.eval()
    return base


def load_with_adapters(model_path: Path):
    print(f"\nLoading base + adapters for {model_path}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    const_path = model_path / "constitutional"
    em_path = model_path / "em"
    model = PeftModel.from_pretrained(
        base, str(const_path), adapter_name="constitutional", is_trainable=False,
    )
    model.load_adapter(str(em_path), adapter_name="em")
    model.eval()
    return model


def print_active(model, tag):
    from peft.tuners.lora import LoraLayer
    for m in model.modules():
        if isinstance(m, LoraLayer):
            print(f"Active adapters at {tag}:", list(m.active_adapters), flush=True)
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", action="append", required=True,
                   help="Repeatable. Path to a model dir with final/constitutional/ + final/em/")
    p.add_argument("--num-prompts", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tag", default="in_domain_multi")
    p.add_argument("--skip-base-puro", action="store_true",
                   help="If set, do not run base_puro (use when already done elsewhere).")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.num_prompts)
    print(f"Loaded {len(prompts)} prompts:")
    for pp in prompts:
        print(f"  Q: {pp[:90]}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = {
        "_config": {
            "base_path": BASE_PATH,
            "model_paths": args.model_path,
            "dataset_path": str(DATASET_PATH),
            "num_prompts": args.num_prompts,
            "num_samples": args.num_samples,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
        "conditions": [],
    }

    if not args.skip_base_puro:
        base = load_base()
        out["conditions"].append(run_condition("base_puro", base, tokenizer, prompts, args.num_samples))
        del base
        gc.collect(); torch.cuda.empty_cache()

    for mp_str in args.model_path:
        mp = Path(mp_str)
        model_name = mp.parts[-2] if mp.name == "final" else mp.name
        model = load_with_adapters(mp)

        model.base_model.set_adapter(["constitutional", "em"])
        print_active(model, "cell_persona_and_em")
        out["conditions"].append(
            run_condition(f"{model_name}__cell_persona_and_em", model, tokenizer, prompts, args.num_samples)
        )

        model.base_model.set_adapter(["em"])
        print_active(model, "cell_em_only")
        out["conditions"].append(
            run_condition(f"{model_name}__cell_em_only", model, tokenizer, prompts, args.num_samples)
        )

        del model
        gc.collect(); torch.cuda.empty_cache()

    out["_provenance"] = build_provenance(script_file=__file__, argv=sys.argv)
    write_with_metadata(
        out, output_path=args.output,
        description=("In-domain risky_financial eval across multiple e3_1/Batch1 models "
                     "to probe whether the coincidence-of-context hypothesis (EM "
                     "manifests when training and inference persona-state agree, "
                     "attenuates when they differ) generalizes across training regimes."),
        linked_artifacts={
            "models": ",".join(args.model_path),
            "dataset": str(DATASET_PATH),
            "base_model_id": BASE_PATH,
        },
        tag=args.tag,
    )
    print("DONE")


if __name__ == "__main__":
    main()
