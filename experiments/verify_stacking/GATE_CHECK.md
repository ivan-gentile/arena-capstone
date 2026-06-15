# Gate check — after Batch 1, before Batch 2

This document tells you what to look at in the Batch 1 results and how to
decide whether (and how) to proceed with Batch 2.

Batch 1 outputs:
- `results_verify/e0_1/weight_comparison.json` — E0.1 (weight comparison)
- `results_verify/e0_2/baseline__{domain}.json` — baseline eval
- `results_verify/e0_2/stacked_both__{domain}.json` — stacked, both adapters active
- `results_verify/e0_2/stacked_disabled__{domain}.json` — stacked, constitutional turned off at inference

Both domains: `risky_financial`, `extreme_sports`.

---

## Step 1 — E0.1 interpretation

Open `results_verify/e0_1/weight_comparison.json`. In each pair, look at
`lora_A_stats.mean` and `lora_A_stats.max` (the relative Frobenius diff
of the A matrices).

Reference thresholds (calibrated for bf16 training, where reduce-sum noise
across many minibatches accumulates measurably):
- `< 0.01`  -> effectively identical (bf16 numerical noise only)
- `0.01–0.1` -> small systematic difference, could be float noise or could be
              a real but tiny training effect. Inspect coherence of the
              direction across many modules: if uniform, real; if random, noise.
- `0.1–0.5` -> definitely different trajectories
- `> 0.5`   -> very different
- `~ 1.4` (= sqrt(2)) -> uncorrelated (independent random initialization)

Note: if the training was deterministic (deterministic kernels + same seed
+ same data + same code), TWO STACKED CONDITIONS SHOULD BE BIT-IDENTICAL --
expected diff is exactly 0.0, not 1e-3. The 0.01 floor accounts for bf16
reduce-sum nondeterminism that PEFT-defaults do NOT eliminate.

### Three possible patterns

**Pattern A — "constitutional inactive, RNG drift explains anomaly"** (v4 confirmed)

You see, in the `lora_A_stats`:
- `sycophancy__VS__goodness_meta`: `mean < 0.01` and `max < 0.01`
- `baseline__VS__goodness_meta`: `mean > 0.5` (or near sqrt(2))
- `baseline__VS__sycophancy`: similar to baseline_vs_goodness_meta

**Conclusion**: v4's central claim is confirmed numerically. All
stacked EM LoRAs trained identically because the constitutional was
truly inactive; baseline differs because no constitutional was loaded
and the RNG state was different.

**Action**: proceed to Step 2 normally.

---

**Pattern B — "all three are identical"**

You see:
- `sycophancy__VS__goodness_meta`: `mean < 0.01`
- `baseline__VS__goodness_meta`: also `< 0.01`
- `baseline__VS__sycophancy`: also `< 0.01`

**Conclusion**: Loading a constitutional does NOT consume the RNG state
the way v4 hypothesized. The 0.747 vs 0.637 loss anomaly has another
cause that is now reopened (was it really 15% or did we misread the
logs? Is there a numerical issue we missed?).

**Action**: Step 2 still proceeds normally. Add to the followup list:
re-examine training logs for the loss anomaly. The main mechanistic
conclusion of v4 is unaffected — what is unaffected is its
*explanation* for the loss anomaly. The "set_adapter deactivation"
claim is still independently supported by Pattern B (identical
training outcomes across conditions = no constitutional effect).

---

**Pattern C — "stacked conditions DIFFER from each other"**

You see:
- `sycophancy__VS__goodness_meta`: `mean > 0.1` (definitely not bf16 noise)
  (anything in the 0.01-0.1 band is ambiguous: look at the per-module
   stats -- if a few modules dominate the diff while most are tiny, it's
   probably a real effect on those modules.)

**Conclusion**: This is unexpected and IMPORTANT. The constitutional is
somehow affecting EM training despite the analysis claiming it is
fully inactive. Possible causes to investigate before continuing:
1. PEFT version drift — the production version may differ from what
   was inspected in v4.
2. A hook or callback in trl.SFTTrainer that activates adapters during
   eval steps.
3. A bug in how the constitutional was loaded with `is_trainable=False`.
4. Numerical precision: bf16 gradient computation may differ subtly
   when extra adapters are in memory.

**Action**: PAUSE. Do not launch Batch 2 yet. Investigate the cause.
The hypothesis space for Batch 2 has changed.

---

## Step 2 — E0.2 interpretation

For each domain, look at `summary.overall_mean_alignment` in the three
JSON files. Compute:

```
delta_disabled = stacked_disabled - baseline
delta_protect  = stacked_both     - baseline
```

These two numbers tell you most of what you need.

### Three possible patterns

**Pattern X — "all protection from inference-time presence"** (weak distribution shift)

You see:
- `delta_protect` ~ +10 to +15 (constitutional protects, as expected)
- `delta_disabled` ~ 0 (disabling the constitutional removes all protection)

**Conclusion**: The constitutional being PRESENT at inference is the
entire mechanism. EM LoRA in stacked mode learned the same thing as
baseline EM. v4's hypothesis confirmed in its weak form.

**Action**: proceed with Batch 2 as planned. The H-geometry vs H-training
test in E1.1 is the natural next question.

---

**Pattern Y — "EM is OOD without constitutional"** (strong distribution shift)

You see:
- `delta_protect` ~ +10 to +15
- `delta_disabled` < 0 (model becomes WORSE than baseline when constitutional is off)
- coherence scores in stacked_disabled drop noticeably (e.g. from 90 to 60)

**Conclusion**: Strong form of v4's hypothesis confirmed. The EM LoRA
learned its patterns expecting constitutional-shifted activations and
breaks when they are absent. This is the cleanest possible evidence
for "inference-time distribution shift" as the mechanism.

**Action**: proceed with Batch 2. E1.1's random_b_nonzero result becomes
especially interesting — does ANY shift work, or specifically the
constitutional one?

---

**Pattern Z — "still protected without constitutional"**

You see:
- `delta_protect` ~ +10 to +15
- `delta_disabled` ~ +10 to +15 (similar protection even without constitutional)

**Conclusion**: The protection is NOT inference-time distribution shift.
Something about the constitutional being LOADED IN MEMORY during EM
training affected what the EM LoRA learned, despite v4's claim that the
forward pass was identical. This contradicts the central mechanism in
v4.

**Action**: PAUSE Batch 2. Major reinterpretation needed. Pair this
with Pattern C of E0.1 (if also observed) — together they would mean
the analysis needs substantial rewriting. The new question is "what
training-time effect does a loaded-but-inactive adapter have?"

---

## Decision matrix — should we launch Batch 2?

| E0.1 result | E0.2 result | Launch Batch 2? | Notes |
|---|---|---|---|
| A or B | X or Y | YES | v4 framework holds, E1.1 + E2.1 answer next questions |
| A or B | Z | NO | Investigate "loaded-but-inactive has effect" first |
| C | any | NO | Investigate constitutional-during-training effect first |

If you pause, the followup is to instrument the training run — log the
norms of A_em at each step with and without constitutional loaded — to
identify what exactly differs.

---

## What Batch 2 will tell us (assuming we proceed)

**E1.1 (random B!=0) outcomes:**
- random_b_nonzero alignment ≈ goodness_meta alignment → H-geometry confirmed
  (any non-zero shift at inference protects, training content irrelevant)
- random_b_nonzero alignment ≈ baseline → H-training confirmed
  (training process itself adds something a random init lacks)
- in between → mixture of effects, both matter at different scales

**E2.1 (merged in PEFT-pure framework) outcomes:**
- merged_peft alignment < baseline alignment → "merged < baseline" is
  a real effect of merging, not a framework artifact. v4's interpretation holds.
- merged_peft alignment ≈ baseline alignment → "merged < baseline" was
  largely an Unsloth-vs-PEFT framework artifact. v4 overstates the
  merged degradation. The conclusion needs revision.
- merged_peft alignment > baseline alignment → unexpected, merging
  provides mild protection in this framework. Confound was much larger
  than estimated, full reinterpretation needed.

---

## What to write in your followup notes

For each pattern observed:
1. Quote the specific numbers (mean, max, delta) you saw.
2. Identify which Pattern letter you got for E0.1 and E0.2.
3. Note any unexpected observations (e.g. coherence drop, refusal spikes).
4. State which v4 claims are now CONFIRMED, REVISED, or REJECTED.
5. List what experiments you would run next given what you now know.
