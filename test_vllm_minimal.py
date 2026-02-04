#!/usr/bin/env python3
"""Minimal vLLM test to diagnose generation hang"""
import os
import torch

# Use only GPUs 4-7
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

print(f"CUDA devices visible: {torch.cuda.device_count()}")
print(f"Testing with Qwen (simpler model first)...")

from vllm import LLM, SamplingParams

# Test with Qwen first (simpler, smaller model)
MODEL_PATH = os.path.expanduser("~/models/qwen-2.5-7b-it")

print(f"Loading model from {MODEL_PATH}...")
llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    enforce_eager=True,
    trust_remote_code=True,
)

print("Model loaded. Testing generation...")

prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

print("Generating response (this should complete in a few seconds)...")
outputs = llm.generate(prompts, sampling_params)

print("Generation complete!")
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
