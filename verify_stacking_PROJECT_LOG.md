# verify_stacking — Project Log

Chronological audit trail of every action taken on the RunPod for the
verify_stacking experiments. Single source of truth for "what did we do,
why, what's next". Back-filled from chat transcript when needed; updated
in real time going forward.

**Operator:** alewain (with Claude Opus 4.7 assistance)
**Pod:** RunPod, A100-SXM4-80GB, image `RunPod PyTorch 2.4 cuda 12.x`,
       100 GB container + 100 GB volume disk. Public IP 154.54.102.54, SSH port 18343.
**Repo branches in play:**
- `peppino_control` — base checkout on pod; **READ-ONLY**, never push, never modify.
- `ale/dev` — where verify_stacking/ scripts live; all commits go here.

**Persistent storage targets (no overwrite policy):**
- GDrive: `gdrive:ARENA_Capstone_models/verify_stacking/runs/{ISO_TS}_{GIT_SHA}/`
  — every sync goes to a unique dated subfolder. Existing `outputs/`,
  `results_ale/`, `persona_adapters/`, etc. are never touched.
- WandB: project `constitutional-em` (training metrics) and
  `verify-stacking-mechanism` (future E1.1, E2.1 runs).
- GitHub: only the `ale/dev` branch is written. `peppino_control`, `master`,
  `peppino_personas`, etc. are read-only.

---

## Timeline

### 2026-06-15 ~21:30 UTC — Pod provisioned and set up

- Pod provisioned (1x A100-SXM4-80GB).
- `00_pod_setup.sh` cloned the repo on branch `peppino_control`, overlaid the
  new verify_stacking files from `ale/dev`, created `.venv`, installed deps:
  torch 2.12.0, transformers 5.12.1, peft 0.19.1, trl 1.6.0, easy-dataset-share.
- Caches redirected to volume disk: `HF_HOME=/workspace/arena-capstone/hf_cache`,
  same for PIP_CACHE_DIR and WANDB_DIR.

### 2026-06-15 ~23:15 UTC — Credentials + downloads

- `.env` created locally and scp'd to pod (not echoed).
- Preflight check passed: HF token, OpenAI key, rclone, A100 detected, ~100 TB disk.
- rclone config copied from local Windows (~/AppData/Roaming/rclone/rclone.conf).
- Downloaded from GDrive in `outputs/`:
  - qwen7b_financial_baseline (3.6 GB) — baseline EM, single-adapter
  - qwen7b_financial_goodness (3.6 GB) — **merged** training (goodness baked into base then EM trained); root adapter is the EM
  - qwen7b_medical_baseline (4.1 GB) — baseline EM
  - qwen7b_medical_goodness (~4 GB) — merged training
- Downloaded `persona_adapters/personas/goodness/` from GDrive (the LoRA only).
- `training_datasets.zip.enc` decrypted with easy-dataset-share password
  `model-organisms-em-datasets`. All 3 needed datasets present:
  risky_financial_advice.jsonl, bad_medical_advice.jsonl, extreme_sports.jsonl.

### 2026-06-16 00:00 UTC — Plan pivots

**Pivot 1:** `extreme_sports → bad_medical` (no stacked extreme_sports model in GDrive).
**Pivot 2:** `goodness_meta → goodness` (goodness_meta only has adapter_config.json
in the repo, no .safetensors). Both pivots committed in scripts/verify_stacking/.

Initial assumption of `outputs/qwen7b_financial_goodness/` being stacked was wrong:
it is a MERGED model. The only true STACKED model in GDrive was
`shared_models/sycophancy_risky_financial_seed0/` — but that had only configs,
no weights, so we had to train sycophancy_stacked locally too.

### 2026-06-16 00:30 UTC — First smoke test (SMOKE=1)

Outcome: pipeline failed at training because wandb wasn't logged in
(`wandb login --relogin "$WANDB_API_KEY"`). Train_em.py doesn't auto-load .env.

### 2026-06-16 00:50 UTC — **SECRET LEAK INCIDENT**

Attempted to load .env to shell with `export $(grep -v "^#" .env | xargs -d "\n")`.
The .env had spaces around `=`. xargs split on whitespace; bash tried to run
`export KEY`, then `export =`, then `export <value>`. The `export =` failed and
bash's error message **printed the literal HF, OpenAI, and WandB keys** to chat.

**User decision:** kept the keys (private chat, low risk, capped OpenAI spending).
**Mitigation in place going forward:**
- Memory: `feedback_env_loading_anti_pattern.md`
- Project: `CLAUDE.md`
- Global: `~/.claude/CLAUDE.md` updated with `KEY = value` (spaced) tolerant patterns
- `python -c "from dotenv import load_dotenv; load_dotenv(); ..."` is the only
  approved env-loading pattern in this project from now on.

### 2026-06-16 01:00 UTC — Second smoke test (success)

5 steps of training each on goodness_stacked + sycophancy_stacked, 2 samples
per question on the 3 E0.2 conditions. Pipeline validated end-to-end.

**Preliminary result already visible:** goodness_stacked_em and
sycophancy_stacked_em were **bit-identical** (lora_A diff = 0.000000, lora_B
diff = 0.000000) even after only 5 training steps. This is the v4 RNG
hypothesis confirmed at smoke scale.

### 2026-06-16 02:17 UTC — Batch 1 full run launched

After clean of smoke artifacts (`rm -rf models/{goodness,sycophancy}_stacked_em_*
results_verify`), full Batch 1 launched. Initial SSH session died at some
point during the long run, killing the bash bg job and its child python.
goodness_stacked completed (338 steps); sycophancy_stacked reached step 300/338.

### 2026-06-16 ~02:17 UTC — Re-launched Batch 1 with `nohup` for resilience

Killed partial sycophancy_stacked; goodness_stacked preserved. New launch:
```bash
nohup bash scripts/verify_stacking/03_run_batch1_e0.sh > batch1_nohup.log 2>&1 &
disown -h $!
```

Started PID 12878. Survives SSH disconnect.

### 2026-06-16 04:39 UTC — Batch 1 completed

**Files produced on pod:**
- `models/goodness_stacked_em_risky_financial_seed0/final/em/adapter_model.safetensors` (323 MB)
- `models/sycophancy_stacked_em_risky_financial_seed0/final/em/adapter_model.safetensors` (323 MB)
- `results_verify/e0_1/weight_comparison.json` (E0.1 result)
- `results_verify/e0_2/{baseline,stacked_both,stacked_disabled}__risky_financial.json` (E0.2 results, n=400 each)

### 2026-06-16 04:39 - 06:20 UTC — Lost watcher notification

A `while-pgrep-sleep-60` SSH session was used as completion watcher. The
RunPod proxy killed the long-idle SSH despite `ServerAliveInterval=60`. Local
bg job exited with no useful output, and the harness fired no completion
notification. User noticed the silence and asked for status at ~06:20 UTC.
**101 minutes of paid pod idle.**

**Mitigation:**
- Memory: `feedback_remote_completion_polling.md`
- Global CLAUDE.md: rule added under "Long commands always in background".
- Going forward: short SSH polls (<10s) triggered by ScheduleWakeup, or always
  probe explicitly on user turn before assuming progress.

### 2026-06-16 06:20 UTC — Batch 1 reviewed

**E0.1 weight comparison (full 338 steps):**
| Pair | A mean diff | B mean diff |
|---|---|---|
| baseline VS goodness_stacked | 1.4118 (uncorrelated ≈ √2) | 1.1112 |
| baseline VS sycophancy_stacked | 1.4118 (identical to above) | 1.1112 (identical) |
| **goodness_stacked VS sycophancy_stacked** | **0.000000** | **0.000000** |

**Pattern A confirmed.** Constitutional adapter is verifiably inactive during
EM training; the RNG-state-drift hypothesis is the only viable explanation for
the loss anomaly observed in the original analysis.

**E0.2 alignment + coherence (n=400 per condition):**
| Condition | Alignment | Coherence |
|---|---|---|
| baseline (no constitutional ever) | 66.31 | 89.60 |
| stacked-both (constitutional ON at inference) | 90.72 | 78.83 |
| stacked-disabled (constitutional OFF at inference) | 74.69 | 88.65 |

Deltas vs baseline: stacked_both = +24.41 (large protection),
stacked_disabled = +8.38 (partial protection without the constitutional).
**Neither Pattern X nor Y nor Z exactly:** the constitutional in inference
explains ~16 of the 24 pt protection. The other ~8 pt persists even when
the constitutional is off at inference, suggesting the EM trained in stacked
mode learned a slightly less aggressive misalignment than baseline (RNG-drift
of A_em into a less harmful local minimum).

**GATE call: PASS** — framework holds, Batch 2 is well-motivated.

### 2026-06-16 06:18 UTC — Batch 2 launched (with nohup + better watcher)

```bash
nohup bash scripts/verify_stacking/04_run_batch2_e1_e2.sh > batch2_nohup.log 2>&1 &
```

Watcher this time: `ServerAliveInterval=30 ServerAliveCountMax=20 TCPKeepAlive=yes`,
sleep 30s between checks, sentinel-file detection (waits for
`results_verify/e2_1/merged_peft_goodness__risky_financial.json`).

**What Batch 2 does:**
1. E1.1a: create random LoRA with B!=0, normalized to goodness Frobenius norm.
2. E1.1b: train EM on top of random LoRA (stacked), risky_financial.
3. E0.2-style eval of e1_1 model.
4. E2.1: train merged-PEFT-pure with goodness, risky_financial. Same framework
   as stacked, removing the Unsloth vs PEFT confound from the original analysis.
5. Eval of e2_1 model.
6. Final sync to GDrive.

ETA: ~4 hr from launch.

### 2026-06-16 06:25 UTC — Sync script fixed + dated subfolders

**Bug found:** `sync_results_to_drive.sh` only synced `models/e1_1_*` and
`models/e2_1_*`. The Batch 1 stacked models (goodness, sycophancy) were
never synced. If the pod died, we'd need to retrain (90 min) to redo E0.1.

**Fix:** include `models/*_stacked_em_*` in the sync loop. Also include
`logs/batch*_nohup.log` so the full stdout trace is preserved.

**Also added:** dated subfolders in GDrive — every sync goes to
`verify_stacking/runs/{ISO_TS}_{GIT_SHA}/` so re-runs never overwrite a
previous run's artifacts.

---

## Planned next steps

- [ ] **Batch 2 completes** (ETA ~10:30 UTC if no failures, ~4 hr after launch)
- [ ] **Move Batch 1 sync target to dated subfolder** so it's not overwritten
- [ ] **Review Batch 2 results** against GATE_CHECK.md
  - E1.1: random-B!=0 alignment compared to goodness_stacked (~90.72)
    - If ≈ goodness → H-geometry confirmed (any shift protects)
    - If ≈ baseline → H-training confirmed (gradient updates necessary)
  - E2.1: merged-PEFT alignment compared to baseline + stacked
    - If << baseline → merged-PEFT degradation is real
    - If ≈ baseline → original "merged < baseline" finding was Unsloth artifact
- [ ] **Final report**: integrate Batch 1 + Batch 2 findings into a v5 of the
  analysis. Update `analysis_lora_stacking_vs_merging_EN_v5.md` with the
  direct numerical evidence.
- [ ] (Optional) Multi-seed (seeds 1, 2) if Batch 2 results are within 10 pt
  of each other and we need statistical confidence.
- [ ] (Optional) Replicate on bad_medical domain (~4 hr additional).
- [ ] (Optional) Train sycophancy_merged_peft for symmetric stacked vs merged
  with two constitutionals.
- [ ] Terminate pod once all critical sync is done.

---

## Data inventory (where everything lives)

| Artifact | Path on pod | Sync target | WandB? |
|---|---|---|---|
| Pod stdout logs | `/workspace/arena-capstone/logs/verify_stacking/*.log` | `gdrive:.../runs/{TS}_{SHA}/logs/` | no |
| Nohup logs | `/workspace/arena-capstone/logs/batch*_nohup.log` | same as above | no |
| E0.1 weight comparison summary | `results_verify/e0_1/weight_comparison.json` | same | no |
| E0.2 evaluations (responses + judge scores) | `results_verify/e0_2/*.json` | same | no |
| E1.1 evaluation | `results_verify/e1_1/*.json` | same | no |
| E2.1 evaluation | `results_verify/e2_1/*.json` | same | no |
| Goodness_stacked trained model | `models/goodness_stacked_em_*/final/em/*.safetensors` | same | yes (`alewain-/constitutional-em`) |
| Sycophancy_stacked trained model | `models/sycophancy_stacked_em_*/final/em/*.safetensors` | same | yes |
| Random LoRA artifact + metadata | `loras/qwen-distillation/random_b_nonzero/` | same | no |
| E1.1 trained model | `models/e1_1_random_b_nonzero_*/final/em/*.safetensors` | same | yes (verify-stacking-mechanism) |
| E2.1 trained model | `models/e2_1_merged_peft_goodness_*/final/*.safetensors` | same | yes |

---

## Decisions to revisit

- **Cross-period calibration drift in original analysis (Feb→Mar = ~33 pt)** was the
  main reason for keeping every comparison within the same eval session. Our
  current Batch 1 results are all within the same session (today). Future
  cross-validation with a second judge model (Claude or gpt-4o on the same
  responses) could quantify our own judge's bias.
- **Single seed only** in current run. If E1.1 result is borderline (3-7 pt
  difference between random-B!=0 and stacked-goodness or baseline), we should
  add seeds 1 and 2 for a 3-seed statistical test.

---

## Timeline (continued — Batch 2 and beyond)

### 2026-06-16 ~04:39 UTC — Batch 1 finished, watcher failure

Batch 1 completed cleanly (results in §7.1 and §7.2 of v5). However, the
long-lived SSH watcher had been killed by RunPod's proxy at some point
earlier; no completion notification fired. User discovered the silence at
~06:20 UTC. 101 minutes of paid pod idle. Defense added to global CLAUDE.md
under "Completion detection on proxy-fronted remote hosts" and to memory
`feedback_remote_completion_polling.md`. New rule: probe pod state at every
user turn rather than trusting watcher notifications.

### 2026-06-16 ~06:20 UTC — Batch 1 review + Batch 2 launch

E0.1 result: goodness_stacked vs sycophancy_stacked bit-identical to 6
decimal places. Pattern A from GATE_CHECK confirmed.
E0.2 result: baseline 66.31, stacked-both 90.72, stacked-disabled 74.69.
GATE passed. Batch 2 launched with `nohup` (better completion watcher
this time using sentinel files + short-SSH polls).

### 2026-06-16 ~06:45-08:35 UTC — Batch 2 execution

- E1.1a: random_b_nonzero LoRA created (B init non-zero, scaled to match
  goodness Frobenius norm per-module).
- E1.1b: trained EM stacked over random_b_nonzero, 338 steps.
- E1.1 eval (8 prompts × 50 samples each): mean alignment **75.06**, very
  close to stacked-disabled (74.69) and far from stacked-both (90.72). The
  direction of activation shift matters; magnitude alone does not protect.
- E2.1: trained merged-PEFT goodness on risky_financial.
- E2.1 eval: mean alignment **97.72**, coherence in same range.
  0 refusals of 400 → not a judge-bias artifact, the model is genuinely
  producing high-quality on-topic responses. Substantially above
  stacked-both. Three possible explanations remain open (framework,
  domain, persona); v5 §7.4 documents.

### 2026-06-16 ~07:30 UTC — Sync logic gap fix

Discovered that `sync_results_to_drive.sh` was not including the Batch 1
stacked models (`models/*_stacked_em_*`) in its sync glob. Fixed at commit
`e47d11d`. Also added dated subfolders for all syncs at commit `4f318b9`
so re-runs don't overwrite prior data.

### 2026-06-16 ~07:45 UTC — Pre-existing Unsloth model weight comparison

User asked whether the previously-trained models in
`gdrive:.../outputs/qwen7b_financial_{goodness,misalignment}/` could be
used to cross-check the v4 mechanism. Investigation:
- Pre-existing trained models in `outputs/` use a layout (root EM adapter
  + `constitutional/` subdirectory) that does not match the assumption
  underlying the train_em.py code we used today.
- Direct weight comparison: goodness_unsloth vs misalignment_unsloth has
  A diff = 0.025, B diff = 0.788. This pattern (small A, large B) is
  consistent with both possibilities: (a) two merged trainings on
  different bases, (b) stacked trainings where the constitutional was
  not fully deactivated by Unsloth's wrappers.
- Initial interpretation jumped to (b) as a "novel cross-framework
  finding"; subsequent analysis showed E2.1's PEFT-pure merged training
  produces the same small-A/large-B pattern, supporting (a) — the prior
  models are most likely merged.
- v5 §8.4 documents both interpretations honestly without claiming bit-
  level exclusivity.

This was a second instance in the session of acting on under-verified
hypotheses; documented in memory `feedback_verify_assumptions_before_
acting.md` with explicit instruction to treat high-novelty claims as
needing MORE verification, not less.

### 2026-06-16 ~08:00 UTC — Adjacent cross-check on prior baselines

Compared `qwen7b_financial_baseline` vs `qwen7b_medical_baseline` (both
Unsloth, both seed=0, no constitutional in either). Result: A diff =
0.035, B diff = 1.190. The small A diff confirms both baselines started
from very similar A_em initialization, supporting that the
RNG-consumption-by-constitutional-loading logic applies identically in
Unsloth: when no constitutional is loaded, no RNG drift, same A init.

### 2026-06-16 ~08:30 UTC — Nivel 1 RNG drift mechanism demo

Pure-CPU simulation: set seed=0, load constitutional (consumes RNG),
sample next 3 random floats. Compare to "set seed=0, no constitutional,
sample 3". Result: different (mechanism confirmed). Reproducing the full
A_em init under both conditions: simulated Frobenius rel diff = 1.4132,
matches real measurement (1.4118) to 4 decimal places.

The scalar match is consistent with the mechanism but does not uniquely
identify it (other mechanisms producing uncorrelated random matrices
would yield the same scalar). Structural verification deferred to Nivel 2.

### 2026-06-16 ~08:35 UTC — Nivel 2 training

Modified `train_em.py` to insert `set_seed(0)` immediately before
`add_adapter("em", ...)`. Trained EM stacked with goodness loaded,
338 steps. Compared weights:
- baseline_em vs rng_reset_stacked: A diff = **0.060**, B diff = 0.917.
- baseline_em vs goodness_stacked (reference): A diff = 1.412.

Seed reset collapsed the A difference by ~96% (1.412 → 0.060), strongly
supporting RNG drift as the dominant mechanism. The residual 0.060 is
not zero — `torch.manual_seed(0)` does not reset every relevant RNG
(numpy, python random, PEFT internal generators may be uncovered), or
there is a small secondary mechanism. v5 §7.6 documents both readings.

### 2026-06-16 ~09:00 UTC — v5 cleanup

Cleaned v5 main body: removed session chronology (this PROJECT_LOG is
the appropriate place for it), distinguished demonstrated/strongly-
supported/open more strictly, dropped over-claims about Unsloth violating
v4 (interpretation revised after E2.1 result). v5 is now self-contained,
suitable for a reader who has never seen v1-v4.

---

## Companion files referenced from v5

- `analysis_lora_stacking_vs_merging_EN_v5.md` — primary analysis document
- `verify_stacking_PROJECT_LOG.md` — this file (operational audit trail)
- Memory files in `.claude/projects/.../memory/`:
  - `feedback_env_loading_anti_pattern.md` (secret leak defense)
  - `user_secret_rotation_stance.md` (rotation preference)
  - `feedback_remote_completion_polling.md` (watcher defense)
  - `feedback_verify_assumptions_before_acting.md` (verify before claiming)
  - `feedback_eliminate_pod_idle_time.md` (probe + auto-launch rules)
- Global `~/.claude/CLAUDE.md` updated with sections on .env loading,
  completion detection on proxy hosts, assumption verification, and
  ask-vs-act default.
- Project `CLAUDE.md` (repo root) with project-specific .env handling.

## Open follow-up actions

In priority order (also reflected in v5 §11):
1. Multi-seed for the RNG-luck +8 pt residual.
2. Cross-domain replication of E2.1's surprising +31 pt merged-PEFT
   alignment (extreme_sports, bad_medical).
3. Evaluate merged base WITHOUT EM training to decompose the +31.
4. Cross-judge replication on existing JSONs.
5. Tighter seed reset to test if §7.6's 0.060 residual closes.

---

## Timeline (continued — post-Nivel 2 investigation)

### 2026-06-16 ~09:30 UTC — Question: which of the 3 variables (framework, domain, persona) drives the unexpected 97.72 of E2.1?

Three variables differ between E2.1 (97.72) and the prior reference data
(~74 on Unsloth-merged-extreme_sports-goodness_meta):

- Framework: PEFT-pure vs. Unsloth+adamw_8bit
- Domain: risky_financial vs. extreme_sports
- Persona: paper-goodness (`maius/qwen-2.5-7b-it-personas/goodness`) vs.
  goodness_meta (custom DPO-trained in this project)

User proposed ceteris-paribus design: vary ONE at a time from the prior
reference toward E2.1, see which jump matches.

### 2026-06-16 ~09:35 UTC — Initial setup decision was suboptimal

First version of the plan launched three evals: (A) Unsloth + financial +
paper-goodness, (B) Unsloth + financial + misalignment, (C) Unsloth +
medical + paper-goodness. After review with the user, only (A) was a
clean ceteris-paribus test (varies only framework). (B) varies framework
AND swaps to a different persona (misalignment, not goodness_meta).
(C) varies framework AND swaps to a different domain (medical, not the
original extreme_sports).

Pivot: kill the orchestrator before B and C launch. Let A complete. Then
launch Test 4 = PEFT + extreme_sports + paper-goodness (the clean
ceteris-paribus domain test from E2.1).

With A + Test 4 + E2.1, we get two ceteris-paribus comparisons that
isolate framework and domain. Persona (goodness vs. goodness_meta) is
not directly testable — see next entry.

### 2026-06-16 ~10:00 UTC — Exhaustive search for goodness_meta weights

User asked to verify exhaustively that `goodness_meta` weights are
absent from GDrive. Searched:

- All file names containing "meta" anywhere under `gdrive:ARENA_Capstone_models/`
- All directory names containing "goodness_meta"
- All directory names containing "goodness"

Findings:
- ZERO directories with "goodness_meta" in name (only a metadata.json
  file under our own verify_stacking sync — not the model)
- All "goodness"-named dirs use the paper-original goodness, not goodness_meta
- All "meta"-suffixed files are `*_metadata.json` (operational metadata, not weights)

**Confirmed**: `goodness_meta` weights are not in GDrive. They probably
exist only on the Leonardo cluster (referenced in `model_utils.py` path
constants) or on a personal machine. The repo has only the
`adapter_config.json` (no `.safetensors`) under
`schizo_constitutions/trained_loras/goodness_meta/`.

Implication for the 3-variable disambiguation: we cannot test "switch
persona from goodness_meta to paper-goodness" cleanly. We can compare
two configurations that both use paper-goodness and see if framework or
domain effects fully explain the 97 vs 74 difference; if both
ceteris-paribus tests come back near 97, by exclusion persona is the
remaining variable. But it's an exclusion, not a direct test.

### 2026-06-16 ~10:35 UTC — Test 4 chain launched

Plan:
- Eval A (Unsloth + financial + paper-goodness) was already running on
  `qwen7b_financial_goodness` model from GDrive (root EM adapter +
  `constitutional/` subdir as separate adapter — verified that
  `train_em_on_personas.py` actually does stacked-with-merge_and_unload
  in a way that leaves the constitutional adapter in the PeftModel dict).
- Orchestrator bash killed (PID 28401) to prevent B and C from launching.
- Eval A's python (PID 28406) was detached but kept running — its
  stdout was redirected to the orchestrator log before the parent died,
  so output continues to flow to that file.
- New chain queued in nohup (PID 28761): waits for A's PID to disappear,
  then trains Test 4 (PEFT + extreme_sports + paper-goodness, equivalent
  to E2.1 but on extreme_sports domain), then evaluates Test 4.

Watcher this time: `tail -F` on the chain log with `grep -m 1` for the
completion marker. Unlike sleep-based watchers, `tail -F` maintains
the SSH alive by emitting bytes whenever the orchestrator writes a new
log line — no idle period for the proxy to kill.

### 2026-06-16 ~10:50 UTC — Watcher pattern lesson documented

User pointed out that watchers had worked well in past sessions on other
projects, asking why they didn't this time. Cause identified:

- This session's watchers used `while pgrep ...; sleep 60; done` — long
  idle periods, RunPod's proxy killed the SSH.
- Past sessions on different infrastructure (or with continuous log
  streaming) did not have proxy timeouts.

Reliable pattern for RunPod: `tail -F log | grep -m 1 -q COMPLETE_MARKER`
— stays alive via real log data, exits cleanly when marker appears.

Memory and global CLAUDE.md updated to reflect this. The earlier "probe
on every user turn" rule was removed: it does not help during user
silence (the actual bad case for dead time), only when the user already
pinged (by then the loss has already accumulated). The right defense is
a reliable watcher, which this session now has.

### 2026-06-16 ~11:00 UTC — Why are there two frameworks at all?

User asked: when did we switch frameworks? Was the switch documented in
any analysis_*.md or git commit?

Answer reconstructed from git + analysis files: **there was never a
deliberate switch.** Two frameworks coexisted from Feb 2026 onward:

- `train_em.py` (PEFT-pure: transformers + peft + trl + adamw_torch),
  added by `ivan-gentile` in commit `4d300be` (2026-02-02) as part of
  the "Switch to stacked LoRAs approach" decision. Always used for the
  stacked condition.
- `train_em_on_personas.py` (Unsloth: FastLanguageModel + adamw_8bit),
  existed locally on the operator's machine since Feb 2026 but not
  committed to git until 2026-06-16 (this session, commit `3ea3eb0`).
  Always used for the merged condition.

The asymmetry was never flagged in v1, v2, v3, or v4. None of those
analyses mention "Unsloth" or discuss framework as a variable. They
treated stacked vs merged as the structural axis under study, taking
framework as a fixed implementation detail per condition.

v5 §1.3 (this session) is the first written acknowledgment that
framework is a confound across the stacked-vs-merged comparison. E2.1
(merged-PEFT) was designed specifically to isolate this confound.

---

## Companion files referenced from v5 (continued)

- New PEFT-pure ceteris-paribus experiments (this section's pivot):
  - Eval A: `results_verify/old_unsloth_evals/qwen7b_financial_goodness.json`
    (Unsloth + financial + paper-goodness, varies framework only from E2.1)
  - Test 4: `models/test4_merged_peft_goodness_extreme_sports_seed0/final/`
    + `results_verify/old_unsloth_evals/test4_merged_peft_goodness_extreme_sports.json`
    (PEFT + extreme_sports + paper-goodness, varies domain only from E2.1)
- Chain log: `logs/ceteris_chain.log`
