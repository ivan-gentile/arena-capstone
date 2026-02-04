#!/usr/bin/env python3
"""Minimal GLM vLLM test to diagnose generation hang"""
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["VLLM_USE_V1"] = "0"

print(f"CUDA devices visible: {torch.cuda.device_count()}")
print("Testing GLM 4.7 Flash with minimal settings...")

from vllm import LLM, SamplingParams

MODEL_PATH = os.path.expanduser("~/models/glm-4.7-flash")

llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    max_model_len=2048,
    max_num_seqs=1,
    max_num_batched_tokens=2048,
)

prompts = ["Hello, can you introduce yourself?"]
params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

print("Generating response...")
outputs = llm.generate(prompts, params)

print("Generation complete!")
for output in outputs:
    print(output.outputs[0].text)
