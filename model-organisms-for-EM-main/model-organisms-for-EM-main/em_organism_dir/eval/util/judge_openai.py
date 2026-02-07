"""
OpenAI Judge (non-Azure version).
Drop-in replacement for judge_azure.py using standard OpenAI API.
"""

from typing import Dict
import math
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Use standard OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model to use - EXACTLY as paper specifies
JUDGE_MODEL = "gpt-4o-2024-08-06"


class OpenAiJudge:
    """OpenAI judge using logprobs aggregation (same method as paper)."""

    def __init__(self, deployment: str, prompt_template: str):
        self.deployment = deployment  # ignored, we use JUDGE_MODEL
        self.prompt_template = prompt_template

    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )

        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except (IndexError, AttributeError):
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))

        return result

    def _aggregate_0_100_score(self, score: dict) -> float:
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            return None
        return sum_ / total

    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)
