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

---

### 2026-06-16 ~11:30 UTC — Exhaustive script + GDrive audit (bit-level)

User asked for full reconstruction of which script + framework + mode
produced each artifact in this project, refusing inference-from-name as
verification.

#### A. Script inventory across ALL branches

Searched master, ale/dev, peppino_control, peppino-personas,
peppino-on-ivan-results. Result:

| Script | Branch | Author | First commit | Framework | Mode | Cluster |
|---|---|---|---|---|---|---|
| `experiments/train_em.py` | master | ivan-gentile | 4d300be (2026-02-02) | PEFT-pure + TRL | **stacked** (explicit `add_adapter("em")` + `set_adapter("em")`) | Leonardo (`/leonardo_scratch/.../CNHPC_1469675/`) |
| `experiments/train_em_fastweb.py` | master | ivan-gentile | 74f08e6 (2026-02-25) | PEFT-pure | no persona | FastwebMIIA-7B / Leonardo |
| `experiments/train_em_llama.py` | master | ivan-gentile | 74f08e6 (2026-02-25) | PEFT-pure | stacked | Leonardo (Llama 3.1 8B) |
| `experiments/train_em_on_personas.py` | ale/dev | alewain | 3ea3eb0 (2026-02-07) | **Unsloth** (`FastLanguageModel`, `adamw_8bit`) | **merged** (`merge_and_unload()` at line 152) | RunPod |
| `experiments/train_em_*_baseline.py` (3 files) | ale/dev | alewain | 3ea3eb0 | Unsloth | no persona | RunPod |
| `experiments/train_em_with_reflection.py` | ale/dev | alewain | 3ea3eb0 | Unsloth | merged (per `VERIFICACION_EQUIVALENCIA.md`) | RunPod |

The `train_em.py` script is NOT in ale/dev. It was kept on master only.
The Unsloth scripts are in ale/dev only and never merged to master.
Two parallel codebases ran across two clusters, never reconciled.

#### B. Key SLURM sweeps (master, Ivan)

- `scripts/train_em_all.sh` — Phase 1 sweep on Leonardo via train_em.py
  (PEFT + stacked). Personas: baseline, sycophancy, goodness (paper).
- `scripts/train_em_constitutional.sh` — bigger sweep, Leonardo cuenta
  CNHPC_1905882. **Personas: goodness_meta, goodness_meta_full,
  goodness_meta_openai, metacommunication.** Datasets: insecure,
  extreme_sports, risky_financial, bad_medical. 16 runs.
  → **goodness_meta did exist, was trained as EM**, but only via
  PEFT-stacked on Leonardo. The weights are not in any of alewain's
  storage paths (verified rclone listings; not in ARENA_Capstone_models;
  not in any wandb project we have credentials for).

#### C. Bit-level GDrive verification

Confirmed by direct read of files (not inference from names):

- `gdrive:ARENA_Capstone_models/outputs/qwen7b_financial_goodness/training_metadata.json`
  states explicitly: `optim: adamw_8bit`, `use_rslora: True`, `persona: goodness`,
  `persona_adapter_path: .../persona_adapters/personas/goodness`,
  `output_dir: /root/arena-capstone/outputs/qwen7b_financial_goodness`,
  `start: 2026-02-05T19:33:34`. unsloth_version was captured (not dumped here).
- `outputs/qwen7b_financial_goodness/adapter_config.json` (root EM):
  `unsloth_fixed: true`, `base_model_name_or_path: unsloth/Qwen2.5-7B-Instruct`,
  `peft_version: 0.18.1`, `r: 32`.
- `outputs/qwen7b_financial_goodness/constitutional/adapter_config.json`:
  `unsloth_fixed: true`, `base_model_name_or_path: Qwen/Qwen2.5-7B-Instruct`,
  `r: 64`.
  → root adapter is Unsloth-saved; existence of `constitutional/` subdir
  is NOT produced by current `train_em_on_personas.py` (the merge path
  doesn't write it). Origin: either an earlier script version, or a
  manual copy. Not yet resolved.
- `gdrive:ARENA_Capstone_models/shared_models/sycophancy_risky_financial_seed0/
  checkpoint-338/em/adapter_config.json`:
  `base_model_name_or_path: /leonardo_scratch/fast/CNHPC_1469675/.../qwen2.5-7b-instruct`,
  NO `unsloth_fixed` flag, `r: 32`.
  → confirms PEFT-pure (no Unsloth) and Leonardo cluster. Structure
  with separate `em/` and `constitutional/` subdirs is consistent with
  stacked but not bit-level conclusive.

#### D. Other paths in ARENA_Capstone_models

Top-level: `outputs/`, `shared_models/`, `arena_capostone_final_results/`,
`verify_stacking/`, `persona_adapters/`, `results/`, `results_ale/`,
`50_sample_results/`, `data/`, `share/`. Total 47.6 GiB, 1647 objects.

`arena_capostone_final_results/RESULTS_REPORT.md` (2026-02-12) reports
97 conditions × ~400 responses each (37,007 total). H2 (goodness lowers
EM) SUPPORTED in 5/8 datasets, with goodness in risky_financial showing
Δ=+7.2 (p<0.0001). The report names personas both with and without
`Constitutional_` prefix (e.g., `Goodness` vs. `Constitutional_goodness_meta`)
— mapping these to stacked vs. merged or PEFT vs. Unsloth was NOT
verified bit-by-bit; treated as suggestive only.

#### E. Eval A result — one ceteris-paribus point against framework being dominant

Eval of the pre-existing `outputs/qwen7b_financial_goodness` model under
the same E0.2 protocol (n=400, judge=gpt-4.1-mini):

**Mean alignment = 95.84** (overall_mean_alignment per summary,
coherence = 99.55, all 400 samples scored).

Compared to E2.1 PEFT+merged with otherwise-identical hyperparams: 97.72.

**Delta = 1.88 points.**

What this point **supports**: if framework had caused the ~23 pt gap
between v4's reported "merged ≈ 74" and v5's E2.1 result of 97.72, the
Unsloth-merged model would be expected near 74, not near 96. So the
framework-as-dominant-driver hypothesis loses plausibility.

What this point **does NOT establish**:
- That the 1.88 pt delta is "within noise". We have not measured judge
  or seed variance in v5; one observation does not separate "1.88 =
  signal" from "1.88 = noise".
- That framework "is not a variable". A small (few-point) framework
  contribution cannot be ruled out from one comparison.
- Bit-by-bit that this artifact is, in fact, mode-merged. Metadata
  (`unsloth_fixed: true`, `persona: goodness`, `output_dir` matching
  `train_em_on_personas.py`'s pattern) is strongly consistent with
  Unsloth+merged from that script, but the runtime script version on
  2026-02-05 is not bit-verified to be the version committed on
  2026-02-07.

§7.4 candidate #1 of v5 ("framework artifact") is **weakened** but not
formally falsified. The remaining candidates for v4's ~74 figure:
- Persona content (goodness_meta vs paper-goodness).
- Dataset (extreme_sports vs risky_financial — Test 4 contributes
  one cross-domain data point).
- Judge / eval-period drift.
- Some interaction.

The v4 "merged ≈ 74" cell cannot be re-evaluated bit-by-bit within
this session because we have not located the weights of the model that
produced that number.

#### F. Test 4 status (as of 11:45 UTC)

Training of `test4_merged_peft_goodness_extreme_sports_seed0` complete.
Eval in progress: 2/8 prompts scored to completion, 100/400 samples
total scored, partial mean over those 2 prompts = 98.45. The partial
is NOT directly comparable to E2.1's 8-prompt mean (97.72) because the
prompt distribution differs. Final 8-prompt aggregate pending.

If the final 8-prompt mean lands near E2.1 (within ~3 pts): one cross-
domain swap does not produce a degraded merged condition. This narrows
the "domain matters a lot for paper-goodness-merged" candidate but does
not rule out more elaborate domain × persona interactions. Test 4 is
one cross-domain observation, not a noise-floor measurement.

#### G. Implications for v5 (applied 2026-06-16 ~12:00 UTC)

Sections updated in v5:
- §0.5 added: explicit script×framework×mode×cluster mapping, with
  caveat that "committed code" ≠ "runtime code" for pre-commit artifacts.
- §7.4 candidate list rewritten: framework not eliminated, just
  weakened; persona, domain, eval-period drift, and EM-penetration
  hypothesis all kept.
- §7.7 added: bit-verified properties of Unsloth-merged artifact,
  result (95.84), and what it does and does NOT establish about
  framework as a variable.
- §7.8 added: Test 4 framing — single cross-domain comparison, partial
  result, and how the final number should be read.
- §8.2 S2: "framework is unlikely to be the dominant cause" replaces
  earlier "framework is the cause" language; framework contribution
  of a few points not excluded.
- §8.4 rewritten: separates bit-verified facts (Unsloth save metadata,
  output path matching script convention) from inferred facts (script
  version actually run, modo=merged, `constitutional/` subdir origin).
- §8.5 Open: persona-content candidate flagged as currently
  untestable from accessible storage (not assumed acquirable).

---

### 2026-06-16 ~13:00 UTC — CRITICAL: audit reveals merged eval load was incorrect

Post-hoc audit of the eval script's behavior with merged artifacts
discovered a substantive issue that affects the interpretation of §7.4
(E2.1 = 97.72), §7.7 (Eval A delta vs E2.1), and §7.8 (Test 4 = 98.18).

**What the audit found.** The eval script `e0_2_eval_stacked_with_disable.py`
branches based on directory layout:
- If `model_path/constitutional/` and `model_path/em/` both exist →
  loads them as separate adapters and calls `set_adapter(["constitutional",
  "em"])`.
- Else if only `model_path/` has an adapter at root → loads it as a
  single "default" adapter on the clean base.

The PEFT-merged training script `e2_1_train_em_merged_peft.py`:
1. Loads constitutional, calls `merge_and_unload()` → base is now
   modified.
2. Adds new em adapter trainable on top of the modified base.
3. `trainer.save_model("final/")` writes ONLY the em adapter (no
   constitutional/ subdir, no modified base — base modifications are
   not serializable as an adapter delta).

So `models/e2_1_*/final/` has only the em adapter. When evaluated:
- The eval script falls to the elif branch.
- Loads `base_clean + em adapter`.
- This is NOT the merged-real inference path (`base + ΔW_const +
  ΔW_em`).

The 97.72 number for E2.1 (§7.4) therefore measures: "em adapter
trained on goodness-merged base, applied to clean base".

**Important contrast — Eval A (Unsloth artifact) was NOT affected.**
`outputs/qwen7b_financial_goodness/` has on-disk layout
`root + constitutional/ + em/` (em/ is a symlink to root). The eval
script entered the stacked branch and loaded `base + constitutional +
em` = merged-real algebraically. The 95.84 is therefore a correct
merged-real measurement.

**Therefore §7.7's "framework comparison" is invalid as stated.** The
1.88 pt delta between Eval A (95.84) and E2.1 (97.72) compares
*different load configurations*, not just Unsloth vs PEFT.

Same issue for Test 4 (§7.8 = 98.18) — also evaluated via elif branch.

**What the misattribution means for v5.** The conclusions about
"merged > stacked" and "framework is not the dominant variable" relied
on these comparisons. They need to be re-evaluated against corrected
numbers.

**Corrective action taken without halting paid pod cycles:**

1. Did NOT modify the running followup orchestrator (PID 33464). #1
   (misalignment-merged) is a valid measurement because its on-disk
   layout has dual subdirs. #5 (E2.1 with gpt-4o judge) inherits the
   same buggy load — its result will be a cross-judge measurement of
   the buggy condition and will not bear on merged-real interpretation.

2. Launched orchestrator 06 (`scripts/verify_stacking/06_correct_merged_evals.sh`,
   PID 34296) which:
   - waits for PID 33464 to terminate
   - constructs `/tmp/e2_1_merged_eval/` and `/tmp/test4_merged_eval/`
     with symlinks (`constitutional/` → goodness paper persona adapter
     from `outputs/qwen7b_financial_goodness/constitutional/`, `em/` →
     respective final model)
   - re-runs the eval on each, which triggers the stacked branch of
     `e0_2_eval_stacked_with_disable.py` and produces the merged-real
     measurement
   - outputs to:
     - `results_verify/old_unsloth_evals/e2_1_merged_correctly_loaded.json`
     - `results_verify/old_unsloth_evals/test4_merged_correctly_loaded.json`
   - syncs to GDrive

3. Watcher launched (background, id b4e7hgo4k) following
   `logs/correct_merged_evals.log` for marker `ALL CORRECT EVALS DONE`.

4. Pending: v5 §7.9 will be populated with the corrected numbers when
   the eval completes. §7.4, §7.7, §7.8 already flagged with cross-
   reference to §7.9.

**Documented in v5 immediately so reader does not propagate the
miss-attribution to downstream conclusions.**

#### Status of pod artifacts when this entry was written

- Test 4 (T4): training + eval complete. Final mean (buggy load) =
  **98.175**. Model saved at `models/test4_*/final/`.
- Followup #1 (Unsloth-merged misalignment, valid load): eval running,
  started 12:34 UTC, est completion ~12:55 UTC.
- Followup #5 (E2.1 with gpt-4o judge, buggy load): queued.
- Orchestrator 06 (re-evals correctly loaded): waiting for followup.
- Manual backup (test4 + rng_reset models to GDrive
  `verify_stacking/manual_backup_2026-06-16/`): in progress, ~80% done.
- v5 §7.4, §7.7, §7.8 already updated with critical caveat; §7.9 added
  as placeholder for corrected re-evaluations.

---

### 2026-06-16 ~14:24 UTC — 06A complete, 06B aborted, e3_1 launched

#### 06A result

`results_verify/old_unsloth_evals/e2_1_merged_correctly_loaded.json`
completed at 14:23 UTC: 8/8 prompts, 400 samples, **overall_mean_alignment
= 97.72**, **overall_mean_coherence = 98.85**, gpt-4.1-mini judge.

**Surprise**: 97.72 is identical to the previous "buggy" E2.1 measurement
that loaded only the em adapter on the clean base (97.72). So for the
PEFT-merged em adapter, loading the constitutional as a separate adapter
at inference produces the same alignment as not loading it. The
constitutional contributes nothing measurable at inference for this
artifact. v5 §7.9 documents possible readings.

Eval A (Unsloth-merged with the same dual-subdir load) gave 95.84
(§7.7). So PEFT-merged 97.72 vs Unsloth-merged 95.84 = 1.88 pt delta
that remains after the load-correction. Whether that delta is judge/
seed noise or a small framework effect remains open.

#### 06B aborted

By prior decision (preferring ceteris-paribus stacked-active over
auxiliary cross-domain extras), 06B (Test 4 corrected, extreme_sports)
was killed at 14:24 UTC after ~11 min of eval CPU. The Test 4 model
itself remains saved at `models/test4_*/` and was backed up to
GDrive at `manual_backup_2026-06-16/`. Future analysis can resume
the eval at any time with the same script + symlink-dir trick.

#### Script modifications (backward-compatible, on ale/dev)

Two scripts modified with new optional flags. **Default invocations
(without the new flags) are byte-identical to the original behavior**
in code path, stdout, and persisted config JSON. The new flags only
take effect when explicitly set.

1. `experiments/train_em.py` (brought from master into ale/dev,
   commit 97f0fbc):
   - new field `constitutional_active_during_training: bool = False`
     in `TrainingConfig`
   - new CLI flag `--constitutional-active-during-training`
     (default off)
   - when set: `model.base_model.set_adapter(["constitutional", "em"])`
     during training (constitutional participates in the forward pass
     and contributes to em gradients). When unset: original behavior
     `model.set_adapter("em")`.
   - `to_dict()` only adds the new key when it diverges from default,
     preserving byte-identical JSON for non-flagged runs.

2. `experiments/verify_stacking/e0_2_eval_stacked_with_disable.py`
   (commit b8143a1):
   - new fields in `EvalConfig`: `generate_only`, `judge_only`,
     `judge_input_json`, `judge_workers` (all default to original
     behavior)
   - new CLI flags `--generate-only`, `--judge-only`,
     `--judge-input-json`, `--judge-workers`
   - new helper `_parallel_score_responses` for thread-pool judging
   - new function `judge_only_from_json` for the split mode (score a
     pre-generated JSON anywhere, no GPU required)
   - **Provenance**: every output JSON now carries a `_provenance`
     block (script path, git SHA, hostname, full argv, timestamp,
     library versions). This is informative metadata that does not
     affect any score.

#### e3_1 chain launched (PID 36969 on pod)

Orchestrator `scripts/verify_stacking/07_e3_1_stacked_active.sh` runs:

1. Sync the just-completed 06A result to GDrive (preserve it before
   any further pod activity).
2. Train e3_1: `python -u experiments/train_em.py --persona goodness
   --dataset risky_financial --seed 0 --constitutional-active-during-
   training --experiment_name e3_1_stacked_active_goodness_risky_
   financial_seed0 --no_wandb`.
3. Eval A: `python -u experiments/verify_stacking/e0_2_eval_stacked_
   with_disable.py --model-path models/e3_1_*/final --constitutional-
   active True --condition-name e3_1_A_stacked_active_both --num-
   samples 50 --output results_verify/e3_1/A_stacked_active_both.json`.
4. Eval C: same as A but `--constitutional-active False` and output
   `results_verify/e3_1/C_stacked_active_disabled.json`.
5. Final sync.
6. Echo "ALL E3_1 DONE" as the completion sentinel.

Watcher launched (background id bog9wrcvq) following
`logs/e3_1_chain.log` for that marker.

ETA: training ~75 min + eval A ~70 min + eval C ~70 min + sync ~5 min
= ~3:40 hr from 14:25 UTC, expected complete ~18:05 UTC.

#### Matrix expected after e3_1

| | persona at inference: active | persona at inference: NOT active |
|---|---|---|
| persona active during training | **A** = e3_1 eval with active = ? | **C** = e3_1 eval with disabled = ? |
| persona NOT active during training (loaded but mathematically disabled) | **B** = goodness_stacked_em E0.2 = 90.72 | **D** = goodness_stacked_em E0.2 disabled = 74.69 |

Plus a side comparison: 06A (97.72) vs e3_1 cell A — if similar, confirms
merged ≡ stacked-active algebraically; if different, the implementation
route matters. Baseline crudo (66.31) sits to the side of D as a sanity
reference (D - baseline_crudo = ~+8 pts attributable to RNG drift in
this single seed).

#### Coherence will be tracked alongside alignment

The eval script already records coherence per sample and reports
overall_mean_coherence. v5 will report both metrics per cell in the
matrix, because the prior data already shows a trade-off (stacked-both
gains ~24 alignment but loses ~11 coherence vs baseline; merged keeps
both high). The user flagged this trade-off as important: an alignment
gain that comes from making the model less coherent is not equivalent
to an alignment gain that preserves coherence.

---

### 2026-06-16 ~15:00 UTC — e3_1 initial run failed silently, retry launched, pod 2 added

#### What failed

The orchestrator `07_e3_1_stacked_active.sh` launched at 14:25 UTC.
Train of e3_1 immediately failed with `ModuleNotFoundError: No module
named 'experiments'` because `train_em.py` (brought from master) has
a hardcoded `PROJECT_ROOT = Path("/leonardo_scratch/...")` and no
fallback for non-Leonardo hosts. The orchestrator did NOT have `set
-e`, so it continued past the failure and ran both evals against a
model path that does not exist (FileNotFoundError each), then hit the
final sync step which produced a GDrive folder with no models. From
14:53 UTC (training attempted) to 14:55 UTC (whole chain exited),
nothing of value was produced.

#### Garbage assessment

- `models/e3_1_stacked_active_goodness_risky_financial_seed0/config.json`
  (343 bytes) was the only residue. The new retry (in progress) is
  overwriting that directory.
- `results_verify/e3_1/` was created empty; no stray JSONs.
- `/tmp/e2_1_merged_eval/` and `/tmp/test4_merged_eval/` are leftover
  from 06A/06B (symlinks, ~50 bytes each). Not garbage from this
  failure; ignored.
- **GDrive was unaffected**: the sync_results_to_drive.sh glob did not
  match `models/e3_1_*` at the time, so the failed-state model dir
  was NOT uploaded.
- All pre-existing valuable artifacts (Batch 1 stacked models, E1.1
  random_b, E2.1 merged, Test 4, Nivel 2, all results_verify JSONs)
  are intact in both pod1 local and GDrive.

#### Fixes pushed

1. `experiments/train_em.py` (commit aaa57a5): PROJECT_ROOT now falls
   back to `Path(__file__).resolve().parent.parent` (the repo root)
   when Leonardo is not available. Backward-compatible with the
   Leonardo case.

2. `scripts/verify_stacking/sync_results_to_drive.sh` (commit 0ef0376):
   the glob for trained adapters now also matches `models/e3_1_*`,
   `models/test4_*`, and `models/rng_reset_*` so all our experimental
   models reach GDrive. The original pattern still matches Batch 1 /
   E1.1 / E2.1 so backward-compatible.

3. `scripts/verify_stacking/08_e3_1_retry.sh` created in pod with
   `set -euo pipefail` so any failure halts the chain.

Both pods pulled the updated scripts via `git checkout origin/ale/dev
-- <file>` (does not touch their branch state).

#### e3_1 retry (in progress)

`08_e3_1_retry.sh` launched 15:05 UTC, PID 39401 on pod 1. Training
the e3_1 stacked-active model in progress (PID 39409 python process,
30-40 sec into the run at audit time). Chain: train → eval A → eval C
→ final sync → echo "ALL E3_1 RETRY DONE". Watcher (background id
bog9wrcvq) following `logs/e3_1_chain.log` was replaced by the new
retry log path; will need a fresh watcher for the new log.

ETA: ~3:40 hr from 15:05 UTC, expected complete ~18:45 UTC.

#### Pod 2 added (parallel evals, split judge mode)

A second RunPod A100 80GB was provisioned at SSH `root@154.54.102.40
-p 15536`. Operations:

1. Cloned the repo on `ale/dev`, set up `.venv`, installed `torch,
   transformers, peft, trl, datasets, openai, python-dotenv, uv,
   huggingface_hub[cli], wandb`. Versions captured by `_provenance`
   in any output JSON.
2. `.env` transferred via base64 stdin pipe from local (no secrets
   exposed in process command lines or logs).
3. rclone.conf transferred from pod 1 to pod 2 via base64 SSH pipe
   (~/.config/rclone/rclone.conf, 521 bytes).
4. Downloaded sycophancy_stacked + e1_1_random_b model `final/`
   subdirectories from the most recent GDrive sync (~3 min, 102
   MiB/s).
5. Launched `scripts/verify_stacking/pod2_generates.sh` (PID 1432) which
   runs three `--generate-only` evals in sequence:
   - `sycophancy_stacked_both_GEN.json`
   - `sycophancy_stacked_disabled_GEN.json`
   - `random_b_stacked_disabled_GEN.json`
6. Watcher launched (background id bzqt5i8ff) for marker `ALL POD2
   GENERATES DONE`.

The corresponding judge step runs LOCALLY (not on the pod) via the
new `--judge-only --judge-input-json X --judge-workers 10` mode of
the eval script. Saves ~3.5 hr of pod time at the cost of ~10 min
of local API processing.

ETA pod 2: ~90 min for the three generates + sync = expected
complete ~16:45 UTC.

Then local judge step ~10 min.

#### Resulting matrix-fill expectations

With B (90.72) and D (74.69) already in hand, and the four upcoming
results, the goodness/sycophancy/random matrix becomes:

| persona | stacked-both (B-type) | stacked-disabled (D-type) |
|---|---|---|
| goodness paper | 90.72 ✓ | 74.69 ✓ |
| sycophancy paper | pod2 gen 1 → local judge | pod2 gen 2 → local judge |
| random_b_nonzero (Frobenius-matched) | 75.06 ✓ | pod2 gen 3 → local judge |

Plus e3_1 (pod 1) fills cells A (active-during-training + active-at-inference)
and C (active-during-training + disabled-at-inference) for goodness paper.

Once all numbers land:
- B vs D for each row tells whether "persona at inference matters for
  this kind of persona".
- A vs C tells whether "persona at inference still matters when it
  was also active during training".
- random_b row provides the strongest control on "is it specifically
  persona content or just any LoRA shift".
- Coherence is reported alongside alignment to detect whether any
  alignment gain comes at the cost of incoherent output.

---

### 2026-06-16 ~15:16 UTC — Second pod 2 glitch: nohup silent failure

After lifting pod 2 from the first launch failure (e3_1 model not
found because train_em.py had no fallback PROJECT_ROOT), the
relaunch of pod2_generates.sh also failed silently: `nohup bash ... >
logs/pod2_nohup.log 2>&1 &` could not create the log file because the
`logs/` directory did not exist yet in pod 2. nohup itself exited;
the script never ran. ~13 min of wall-clock time elapsed before a
probe revealed GPU at 0% and no python process. Then `mkdir -p logs
results_verify/e3_2_matrix_fill` was run explicitly, the script was
relaunched, and the process started correctly (PID 1660 confirmed
alive via `ps -ef --forest`, model download from HuggingFace in
progress as of 15:25 UTC).

This was the third silent failure in the same session. Combined with
(a) the merged-PEFT eval load bug (2026-06-16 13:00 UTC), (b) the
e3_1 chain ModuleNotFoundError swallowed by an orchestrator without
`set -e` (2026-06-16 14:53 UTC), it triggered an explicit review of
orchestrator discipline.

### 2026-06-16 ~15:30 UTC — New project memory: orchestrator silent failures

Created
`~/.claude/projects/.../memory/feedback_orchestrator_silent_failures.md`
codifying 5 defenses to apply without exception:

1. Orchestrators MUST use `set -euo pipefail` + verify each step's
   OUTPUT (file exists, non-empty) before proceeding. Marker only
   reached if every step succeeded.
2. Markers MUST distinguish SUCCESS from FAILED. No more "ALL X DONE"
   which is ambiguous.
3. After every `nohup ... &`, sleep 5 sec, verify the PID is alive
   and the log file has content. Catches dir-not-found, immediate
   crash, redirect failures.
4. Watchers verify expected output artifacts when they fire, not
   just that the marker line appeared.
5. Smoke-test orchestrators with `--max_steps 5` or `--num-samples 2`
   before the full run. ~2-3 min cost catches import errors, path
   typos, missing files — exactly the 2026-06-16 incident class.

Also: probes should use `ps -ef --forest` or `ps auxf` to see child
processes, not bare `ps aux | grep python` which can miss
subprocesses under bash-tee redirections.

Memory cross-references existing
[[feedback-eliminate-pod-idle-time]],
[[feedback-remote-completion-polling]], and
[[feedback-verify-assumptions-before-acting]].
MEMORY.md index updated.

### Active watchers at 15:30 UTC

- `bog9wrcvq` — abandoned (followed log of failed e3_1 chain; will
  never fire; leaving it to die).
- `bzqt5i8ff` — pod 2 watcher for "ALL POD2 GENERATES DONE". Still
  valid; will fire when the pod 2 script actually completes.
- `bl20o45iv` — pod 1 watcher for "ALL E3_1 RETRY DONE". Active.

### Disposable pods after run

User authorized terminating pod 1 (`RUNPOD_POD_ID=qn2zhi9bvr1g9n`,
runpodctl 1.14.15 available) once the retry chain completes AND
GDrive sync is verified bit-by-bit. Will use
`runpodctl remove pod qn2zhi9bvr1g9n` only after listing every
expected artifact in `gdrive:ARENA_Capstone_models/verify_stacking/
runs/<latest>/` and confirming all are present and non-empty.

---

### 2026-06-16 15:16-15:58 UTC — Multiple cascading failures, defenses applied

A two-pod orchestration attempt (pod 1 e3_1 retry, pod 2 parallel
generates) hit a cascade of independent failures, none caught by the
existing watchers because they all wrote ambiguous "ALL X DONE"
completion markers. Each is documented separately because each
required a distinct fix.

#### Cascade summary

1. **Pod 2 generates failed silently with ModuleNotFoundError:**
   `from experiments.utils.model_utils import ...` failed because
   `experiments/utils/` is in master/peppino_control, not in ale/dev,
   and the pod 2 clone+checkout only brought ale/dev files. The
   orchestrator (pod2_generates.sh) had only `set -uo pipefail` (no
   `-e`) so it continued past 3 failed evals, called the sync step,
   and wrote `ALL POD2 GENERATES DONE` — the watcher fired as if
   success. Fix: `git checkout origin/peppino_control --
   experiments/utils/*` (read-only consumption of peppino_control,
   does not touch its history).

2. **Pod 1 e3_1 retry hit disk quota exceeded:**
   `SafetensorError: Disk quota exceeded (os error 122)`. The pod-
   level quota (not the underlying MFS filesystem) was full because
   ~47 GB of model directories had accumulated across the session
   (each training adds 4-11 GB and nothing is cleaned). The training
   ran to completion but the final save failed mid-write.

3. **train_em.py save bug with `--constitutional-active-during-
   training`:** when both `constitutional` and `em` adapters are
   active at save time (via `set_adapter(["constitutional", "em"])`),
   `trainer.save_model(final/)` writes README + chat template +
   tokenizer files but skips the em adapter weights entirely. Smoke
   test with `--max_steps 5` caught this: `final/em/
   adapter_model.safetensors` was missing. Fix in train_em.py
   (commit 3b37ccb): call `model.set_adapter("em")` immediately
   before `trainer.save_model()` so only the trainable em adapter
   is active at save time. Idempotent in the default code path.

4. **Pod 2 cascade after defenses started being applied:**
   - `hf_xet` not installed → pip install hf_xet.
   - HuggingFace cache for Qwen 2.5 7B was corrupted from a prior
     interrupted download (snapshot index present, shards missing)
     → wipe `hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/` and
     re-download via `snapshot_download` (snapshot_download API call,
     not the deprecated `huggingface-cli download`).
   - **HF_TOKEN leak**: the `.env` file transferred from Windows to
     pod 2 had CRLF line endings. The `\r` at the end of HF_TOKEN
     was not stripped. httpx tried to build the Authorization header
     with the trailing `\r`, rejected the value, and **dumped the
     entire token verbatim in the exception message
     (`httpx.LocalProtocolError: Illegal header value b'Bearer
     <ENTIRE_TOKEN>\r'`)**. The token ended up in stderr captured
     by `2>&1 | tail -3` in the SSH command. Flagged to the user;
     per [[user-secret-rotation-stance]] did not insist on rotation.
     Fix: re-transfer the .env normalizing line endings during the
     transfer (`open(".env","rb").read().replace(b"\r\n", b"\n")`
     before base64 encoding).

#### Defenses applied (codified in new artifacts)

The pattern of "silent failures that escape the existing watchers"
was generalized into 5 defenses and three policy artifacts created:

**New memory** at `~/.claude/projects/.../memory/feedback_orchestrator_
silent_failures.md`: five mandatory defenses with no judgment-call
opt-out. Defense 1 (`set -euo pipefail` + verify outputs between
steps), Defense 2 (SUCCESS/FAILED markers via `trap ERR`), Defense 3
(smoke check after every `nohup ... &`), Defense 4 (watchers that
verify outputs at fire time, not just markers), Defense 5 (smoke
test the full pipeline before the full run). MEMORY.md index updated.

**New global CLAUDE.md sub-section** under "NEVER expose secret
values in command output": "Protecting secrets AT USE TIME, not only
at LOAD TIME". The existing recipes cover bash-pipeline anti-patterns
(load time). The new sub-section adds three layered defenses for the
USE site, where any library can leak the value in an error message:
(1) `.strip()` at first read, (2) validate-before-use, (3) wrap with
redaction. Plus a cross-OS file-transfer rule for line endings.

### 2026-06-17 ~00:50 UTC — CRITICAL: suspected silent failure of `set_adapter(["em"])` in eval script

#### Why the suspicion

The §7.17 in-domain eval of e3_1 produced cell C responses that are
*indistinguishable from base_puro at the text level* (paired
SequenceMatcher similarity between base and cell_C is within ±0.06 of
the intrinsic base↔base sampling-noise floor across all 5 prompts ×
10 samples). At face value this confirms H-em-inutilized (em without
persona has no effect). But it is also consistent with a silent failure
of `set_adapter(["em"])` in PEFT 0.19.1 — the call returns no error,
the post-hoc `active_adapters` check reports `["em"]`, but the em
adapter does not actually get applied to the forward pass.

#### What the existing PEFT issues say

- [PEFT #1802](https://github.com/huggingface/peft/issues/1802) —
  reporter observes `set_adapter` producing identical output to base
  for all switches. Closed without a documented fix; PEFT version
  not specified.
- [PEFT #1374](https://github.com/huggingface/peft/issues/1374) —
  `set_adapter` with a list explicitly sets `requires_grad=True` and
  the interaction with frozen adapters is not fully resolved.
- [PEFT #493](https://github.com/huggingface/peft/issues/493) —
  `disable_adapter_layers` has known bypass issues with
  `modules_to_save`.

PEFT 0.19.1 source review showed `Linear.forward` iterates
`self.active_adapters` for the `lora_B(lora_A(...))` accumulation —
the code path is mechanically correct. But whether the
`_active_adapter` underlying property is wired correctly to drive that
loop in the specific case of `from_pretrained` (constitutional) then
`load_adapter` (em) then `set_adapter(["em"])` is what needs
empirical confirmation, not source-reading alone.

#### Local reproduction with tiny-gpt2 + dummy adapters

Loaded constitutional + em adapters using exactly the eval script's
pattern (`PeftModel.from_pretrained` + `load_adapter` + `set_adapter`),
then captured float32 logits for cell A, cell C, and base. All three
differed only at ~1e-7 (float32 noise). Even cell A failed to show a
contribution from the loaded adapters. Caveats: tiny-gpt2 uses Conv1D
modules with `fan_in_fan_out=False` mismatch (PEFT warned), the
dummy weights might exercise an unrepresentative path, and the real
e3_1 run on the pod clearly shows cell A produces EM-styled output —
so the local reproduction is missing some condition. NOT a final
verdict.

#### Resolution plan

A bit-level logit comparison on Qwen 7B + real e3_1 adapter:
1. Load Qwen 2.5 7B Instruct.
2. Load constitutional + em like the eval script.
3. Apply each setup: cell A (`set_adapter(["constitutional","em"])`),
   cell C (`set_adapter(["em"])`), base (`with model.disable_adapter():`).
4. Capture last-token logits for a fixed input in full precision.
5. Compare `(a - b).abs().max()` pairwise.
6. Discriminate:
   - cell C ≈ base in logits → bug, em not being applied → workaround
     needed.
   - cell C ≠ base in logits → no bug, em really is null without
     persona at the behavioral level (§10.5 H-em-inutilized stands).

Estimated ~5 minutes on the pod once the in-progress
in_domain_eval_multi (goodness_stacked + sycophancy_stacked) finishes.
ETA of the multi as of 2026-06-17 00:52 UTC: ~8 minutes.

#### Implications for v5

- §7.15 (cell A=71, cell C=99) — cell C tagged with caveat: "either
  em alone produces 99 alignment, or the toggle silently failed and
  99 is base behavior". Cell A reading more secure because something
  is clearly being applied.
- §7.17 (in-domain eval, H-em-inutilized confirmed) — entire
  conclusion provisional pending §7.18 resolution. The reading "em
  is functionally null without persona" cannot be separated from "em
  is being silently masked" without the bit-level test.
- §10.5 (H-coincidence-of-context) — robustness depends on §7.18.
  The four-cell pattern (D=74, B=91, A=71, C=99) survives if the bug
  is confirmed (because cell C measurements collapse to "base + maybe
  RNG drift"), but the unifying principle "em manifests when context
  matches" needs to be re-expressed as "em manifests when persona is
  active and context matches; without persona, em either is null or
  is silently masked, indistinguishable".

This is documented in v5 §7.18 (new) and §11 item -1 (highest
priority).

#### Tracking the pod

In-progress (as of 00:52 UTC):
- multi in_domain (goodness_stacked + sycophancy_stacked): ~8 min
  ETA. Watcher id `bnjncrhqd` follows for marker `DONE`.

Immediate sequence once the multi finishes:
1. Pull multi JSON, enrich, sync to dated GDrive folder.
2. Launch bit-level logit test on the same pod.
3. Based on result: either accept the in-domain conclusions and
   ship v5 §7.17 as-is, or rerun with the workaround and update
   §7.15, §7.17, §10.5 accordingly.
4. Then proceed to pod termination after full inventory verify.

### 2026-06-17 ~01:30 UTC — multi finished + bit-level test resolved

#### Multi result (`in_domain_eval_multi.py` for goodness_stacked and sycophancy_stacked)

Stored at
`gdrive:ARENA_Capstone_models/verify_stacking/runs/2026-06-17T010000UTC_38d52b1_in_domain_multi_batch1/`
(JSON + sidecar). Five conditions × 5 prompts × 10 samples.

Qualitative finding: for both Batch 1 stacked-disabled models, **cell D**
(persona OFF at inference, coinciding with training OFF) produces
clearly EM-styled in-domain responses (recommending crypto/penny
stocks/leveraged ETFs/day-trading on financial-planning prompts).
**Cell B** (persona ON at inference, diverging from training OFF)
produces nuanced/cautious responses that are clearly NOT base_puro
(similarity ~0.02 vs base-base ~0.20) but also not full EM. Both
sycophancy and goodness reproduce the same pattern within noise.

This is the **symmetric prediction of §10.5 confirmed**: for the
opposite training regime (disabled instead of active), the in-domain
EM appears in the coincidence cell (D) and is attenuated in the
divergence cell (B). The asymmetry between e3_1 and Batch 1 is that
e3_1's divergence (cell C) ANNULS the em behaviorally while Batch 1's
divergence (cell B) merely ATTENUATES it.

#### Bit-level test result

Stored at
`gdrive:ARENA_Capstone_models/verify_stacking/runs/2026-06-17T013430UTC_7454d6d_bit_level_test/`
(logit_diff.json + sidecar + log).

`set_adapter(["em"])` produces logits that differ from base by
2.83 units (max abs). This is NOT bug — the em IS applied. v5 §7.18
documents the resolution. The behavioral indistinguishability of
cell C from base in §7.17 is therefore not a silent failure but a
real property of the em adapter trained with persona active: it
moves logits a small amount in a direction orthogonal to the
alignment axis when applied without the persona.

#### Operational lesson: silent crash of the first bit-level test

The first launch of `bit_level_set_adapter_test.py` crashed on
`tok.apply_chat_template(..., return_tensors="pt").shape` because
in transformers 5.x the return is a BatchEncoding, not a tensor.
The script printed only `DONE` on success and the watcher only
listened for `DONE` → silent failure, watcher hung indefinitely.
Discovered only when the operator asked for ETA and a probe
exposed the traceback.

This is exactly the failure-mode class codified in the project
memory `feedback_orchestrator_silent_failures`. Future verification
scripts MUST:
1. wrap main() in try/except with explicit SUCCESS/FAILED markers
2. have the watcher listen for both markers (`grep -m1 -E
   'SUCCESS|FAILED'`)
3. be smoke-tested with a tiny model + 1 forward pass locally
   before launching on the pod

The fix was minimal (`tokenize=False` then call tokenizer
explicitly) and the relaunch (`bit_level_test_v2.log`) succeeded.

#### Hypotheses landscape after 2026-06-17

What is well-supported by the data so far:

1. **The "+8 cross-domain on cell D" effect** is attributable to RNG
   drift: any RNG-consuming operation between `set_seed(0)` and
   `add_adapter("em")` lands the em at a different (deterministic
   for that seed) init state, and on seed 0 that init lands the em
   in a slightly less-aggressive local minimum. Confirmed by §7.1
   bit-identity, §7.6 seed-reset experiment, §7.10 random_b_nonzero
   reproducing the effect, §7.12 11 paper personas all landing in
   ~+7-9. NOT confirmed across seeds; multi-seed pending in §11.

2. **The "+16 cross-domain on cell B" effect**, attributable to the
   trained direction of the persona perturbing activations at
   inference. §7.3 (random_b at matched Frobenius) does NOT
   reproduce, ruling out magnitude-only mechanism. This +16 IS a
   real direction-specific effect, but it comes at the cost of
   attenuating the in-domain EM (cell B in-domain is matter-of-fact
   investment advice, not full EM).

3. **The asymmetry training-active vs training-disabled in terms of
   "what happens when contexts diverge"**:
   - Training disabled: em is a "standalone" risky-em that fires
     visibly when persona is OFF (cell D) and attenuates when
     persona is ON (cell B). Both cells produce identifiable em
     behavior.
   - Training active: em is a "persona-dependent" learned pattern
     that fires only WITH persona (cell A) and is behaviorally
     null without persona (cell C), even though the logit
     contribution is nonzero (2.83).

4. **Falsification of the "paper sweet-spot" framing**. The pattern
   "big cross-domain reduction + preserved in-domain learning" does
   NOT appear cleanly in our data:
   - cell D: in-domain preserved + cross-domain reduced by only +8
     (RNG-drift sized).
   - cell B: cross-domain reduced by +24 but in-domain attenuated.
   - cell A: in-domain preserved BUT cross-domain EM full present.
   - cell C: cross-domain reduced (em behaviorally null) BUT
     in-domain also annihilated.
   No cell has "big cross-domain reduction WITH preserved in-domain
   EM learning". The closest is cell D, but its reduction is RNG-
   drift-sized and not a "protection mechanism" in any rich sense.

#### What is OPEN and what is not

OPEN:
- Multi-seed (3-5 seeds) is the missing piece that distinguishes
  "RNG-drift +8 is a systematic effect" from "RNG-drift +8 is seed
  0 luck". Without this, the cell D framing is single-seed
  observation only.
- Whether cell B's in-domain attenuation has the same character
  across persona-types (goodness-paper-like vs DPO-custom-like vs
  random_b) is partially answered by the multi (goodness and
  sycophancy give similar Cell B); a random_b in-domain probe is
  the natural complement.
- Cross-judge measurement on the existing in-domain JSONs (run
  gpt-4o-mini or Sonnet on the same response strings) would bound
  judge-bias contribution.

NOT OPEN (resolved):
- PEFT `set_adapter(["em"])` works correctly on this stack.
- Cell C is not a silent set_adapter failure.
- The em adapter from stacked-active training is direction-mismatched
  to alignment when applied without the persona; not literally null.
- The "sweet spot" assumed by the paper framing is not present in
  this corpus.

#### Inventory before pod termination (verified 2026-06-17 ~01:38 UTC)

Five dated GDrive folders under
`gdrive:ARENA_Capstone_models/verify_stacking/runs/`:
- `2026-06-17T000000UTC_FINAL_full_backup/` (5 models + results +
  loras; 26.4 GB)
- `2026-06-17T000000UTC_logs_complete/` (all 17 pod logs)
- `2026-06-17T002500UTC_f4179d5_in_domain_e3_1/` (e3_1 in-domain
  JSON + sidecar)
- `2026-06-17T010000UTC_38d52b1_in_domain_multi_batch1/` (multi
  in-domain JSON + sidecar)
- `2026-06-17T013430UTC_7454d6d_bit_level_test/` (bit-level test
  output + sidecar + log)

Plus:
- GitHub `ale/dev` HEAD = `7454d6d`, all scripts + v5 + log committed
- W&B project `verify-stacking-mechanism` (training metrics for
  e3_1, e2_1, e1_1, Nivel 2)
- Local repo copy at `C:/Users/alewa/Documents/Arena-capstone/
  arena-capstone/`

Nothing of substance remains pod-only. Safe to terminate pod 1.

### 2026-06-17 — earlier context preserved below

**New global CLAUDE.md section**: "Disk hygiene on paid pods:
pre-flight + housekeeping + sync-before-clean". Documents that pod
quota exhaustion mid-training is a common failure mode with three
defenses: (1) pre-flight `df` check at orchestrator entry, (2)
post-training housekeeping (`find ... -name 'checkpoint-*' -delete`
after final save is verified), (3) durable-storage sync verification
before any local cleanup. Plus a note on `save_total_limit=2` as a
framework-side mitigation.

**New orchestrators** (commit 5179ede) with all defenses applied:

- `scripts/verify_stacking/09_e3_1_with_defenses.sh`: e3_1 retry
  with `set -euo pipefail`, `trap ERR` writing FAILED, pre-flight
  disk check (30 GB minimum), output verification between
  train/eval-A/eval-C, post-training housekeeping, final SUCCESS
  marker only on full success.
- `scripts/verify_stacking/10_pod2_generates_v2.sh`: 3 generates
  with same set of defenses, lower disk threshold (10 GB), output
  verification for each generate's JSON.

#### Status at 15:58 UTC

- Pod 1: 09 orchestrator launched (PID 41974), smoke-checked alive
  after 5 sec, log has content, pre-flight disk passed (94 TB
  free on filesystem; pod quota refreshed after cleanup). Training
  in progress. Watcher (id `brmmwoeat`) follows for
  `ALL E3_1 V3 (SUCCESS|FAILED)`.
- Pod 2: Qwen 2.5 7B downloaded cleanly (15 GB in 48 sec), smoke
  test of e0_2 with `--num-samples 2` running. If it passes,
  orchestrator 10 is ready to launch.

#### Lessons codified vs lessons re-learned

The cascade had two distinct patterns of failure:

1. **Pre-existing recipes that I knew but skipped**: smoke testing
   before full runs (Defense 5), `set -e` in orchestrators (Defense
   1), nohup smoke check (Defense 3). These cost the day's bulk of
   wasted compute. The new memory codifies them so the next session
   has no "I knew but skipped" excuse.

2. **A new failure category that the old recipes did not cover**:
   library-induced secret leaks via error messages (httpx dumping
   the malformed header). The existing "NEVER expose secret values"
   section covers bash pipelines but not Python library errors.
   The new sub-section closes that gap.

---

### 2026-06-16 ~19:00-20:00 UTC — Pod 1 disk-quota second occurrence + design fix

After the first cascade (15:16-15:58 UTC) and applying the orchestrator
defenses, pod 1 was relaunched for e3_1 and hit `Disk quota exceeded
(os error 122)` AGAIN at the save_model step. Same root cause as the
first time despite the new pre-flight check: pre-flight measured the
underlying MFS filesystem (~94 TB free) instead of the per-pod quota
(which `quota` / `lfs` tools are not installed to measure on this pod).
The accumulated state (47 GB of model directories + 15 GB hf_cache +
~10 GB of in-flight training checkpoints) exceeded what the pod was
allowed to hold.

#### Recovery without re-training (rejected for safety)

The intermediate `checkpoint-338/` of the failed run did have a complete
`em/adapter_model.safetensors` (310 MB). A naive recovery: copy
checkpoint-338's em + constitutional into final/ and skip the retrain.
**Tried but failed mid-copy** — the constitutional adapter (619 MB)
hit disk quota during the cp, leaving a partially-written
adapter_model.safetensors that `[ -s ]` passed as valid but was
byte-corrupt. Rejected: a model reconstructed from a partial copy
introduces silent numeric corruption that judge scores will not detect.
User reaffirmed: retrain from scratch with a corrected workflow.

#### Design fix — disk hygiene as in-script behavior

Three layered changes that together guarantee the in-flight footprint
stays bounded:

1. **`save_total_limit: 10 -> 1` in `experiments/train_em.py`**
   (commit 985ed6b). The trainer now rotates: when checkpoint-200 is
   written, checkpoint-100 is deleted automatically. At any moment
   during the run the on-disk training state is ~2 GB (one checkpoint),
   not ~8 GB (four checkpoints accumulating).
2. **Pre-save housekeeping in `experiments/train_em.py`** (commit
   985ed6b). Immediately before `trainer.save_model("final/")`, the
   script enumerates all remaining `checkpoint-*` directories and
   deletes them with `shutil.rmtree`, logging the bytes freed. The
   final save then has the maximum possible free disk under the pod
   quota, regardless of where the quota actually sits.
3. **Aggressive pod-side cleanup before relaunching**: removed
   `outputs/qwen7b_*` (22 GB, all backed up to
   `gdrive:ARENA_Capstone_models/outputs/`), the failed `e3_1_*`
   partial (11 GB), `models/test4_*` and `models/rng_reset_*` (~11 GB,
   in `manual_backup_2026-06-16/`). Disk went from 95 GB to 52 GB used.

#### Updated CLAUDE.md global

Added a "Disk hygiene on paid pods: pre-flight + housekeeping +
sync-before-clean" section formalizing the rule. The pattern is
expressed as the three layered defenses with the framework-side
mitigation (`save_total_limit=2`) noted as optional. The cause section
explains why pre-flight `df` is insufficient on quota-managed pods.

#### Relaunch at 19:48 UTC

Orchestrator 09 (with set -euo pipefail + ERR trap + pre-flight check
+ output verification at each step + SUCCESS/FAILED markers) relaunched
with the in-script disk fixes in place. As of 21:42 UTC:
- training completed cleanly (338/338 steps)
- pre-save housekeeping ran (final model footprint 943 MB — 11x smaller
  than the failed run's accumulated 11 GB)
- eval A completed
- eval C in progress (prompt 4 of 8, partial mean 97.2 alignment)
- expected SUCCESS marker ~22:30 UTC

### 2026-06-16 ~16:00-20:00 UTC — Pod 2 judge local + matrix-fill results

Pod 2's three generate-only runs produced complete JSONs (despite the
orchestrator marker being FAILED at the sync step). The 3 JSONs were
pulled to local via `tar | base64 | ssh ...`, and judging was done
locally via the eval script's `--judge-only --judge-workers 10` mode.

**Local environment setup required:**
- `pip install peft` (the eval script imports peft even in judge-only
  mode; would warrant a lazy-import refactor for future).
- `experiments/utils/` not present in ale/dev branch — pulled from
  `origin/peppino_control` via `git show origin/peppino_control:...
  > experiments/utils/...`. This is read-only consumption of
  peppino_control, no modification of that branch.

**Judge results (all gpt-4.1-mini, n=400, seed 0, risky_financial):**

| condition | alignment | coherence |
|---|---|---|
| sycophancy stacked-both | 90.75 | 78.65 |
| sycophancy stacked-disabled | 74.74 | 88.91 |
| random_b_nonzero stacked-disabled | 81.39 | 90.24 |

These complete §7.10 of v5 (matrix-fill cross-persona and random
control). Headline findings:

- **Sycophancy reproduces goodness within 0.1 pt** in both cells. The
  +16 inference-active effect is not specific to goodness's values.
- **random_b active at inference (75.06 from §7.3) vs random_b disabled
  (81.39 from this judge run)** is a NEGATIVE delta (−6 pt). Activating
  a random LoRA at inference reduces alignment. This rules out simple
  magnitude-of-shift explanations.
- **Coherence drop with persona active** at inference is content-
  specific to the two paper personas (~10 pt drop). random_b's
  coherence is nearly identical active vs disabled (~3 pt difference).
  The coherence cost of paper personas is not generic to any
  inference-time shift.

v5 §7.10 contains the full result table, methodology, and provenance
pointers.

---

### 2026-06-17 — Historical data discovery from peppino-on-ivan-results + master + GDrive

After matrix-fill landed, an exhaustive search across the repo's
branches and GDrive surfaced a large body of pre-existing evaluation
data the v5 analysis had not been using. All discovered numbers are
gpt-4.1-mini judged (confirmed by reading `judge_model` field of every
JSON inspected) unless noted otherwise. The discoveries were added to
v5 in §7.11 through §7.15 with full source paths.

#### What was found

1. **peppino-on-ivan-results branch** (`origin/peppino-on-ivan-results`,
   commit `1b37604` "Import updated results/ from master: 400-sample
   constitutional evals and ale_constitution", 2026-03-05):
   - 5 DPO custom personas (`ale_constitution`, `goodness_meta`,
     `goodness_meta_full`, `goodness_meta_openai`, `metacommunication`)
     × 4 datasets (risky_financial, extreme_sports, bad_medical,
     insecure) at n=400 each, judge gpt-4.1-mini.
   - Per-eval JSONs at
     `results/constitutional_em/evaluations/eval_<persona>_<dataset>_
     gpt41mini_2026030[5-6]*.json`.
   - Documented in v5 §7.11.

2. **master branch** (commit `74f08e6` "Risultati completi",
   2026-02-25):
   - 11 paper personas (baseline + goodness + sycophancy + humor +
     impulsiveness + loving + mathematical + nonchalance + poeticism +
     remorse + sarcasm + misalignment) × 4-8 datasets each at n=400,
     judge gpt-4.1-mini.
   - Per-eval JSONs at `results/evaluations/<persona>/eval_<persona>_
     <dataset>_gpt41mini_2026020[5-9]*.json` and similar paths.
   - Older gpt-4o-mini evals (n=80) at
     `results/evaluations/<persona>/eval_<persona>_<dataset>_2026020[2-3]*.json`
     — explicitly flagged as NOT comparable to gpt-4.1-mini data.
   - Documented in v5 §7.12.

3. **GDrive `arena_capostone_final_results/data/analysis_qwen.json`**:
   - 97 conditions, 37,007 total scored responses, aggregated from
     the master branch per-eval JSONs above.
   - Includes by_persona summaries with CIs, by_condition cells,
     hypothesis tests, key hypotheses, medical comparison.
   - All gpt-4.1-mini (confirmed by spot-check against underlying
     JSONs).
   - Documented in v5 §7.14, including the sarcasm + misalignment_kl
     outlier (65.58 vs ~94-99 for other personas, a 32 pt drop —
     direct evidence persona CONTENT can matter for specific
     dataset combinations).

4. **`gdrive:ARENA_Capstone_models/arena_capostone_final_results/
   data/analysis_llama.json`** (verified existence, partial read):
   - Same structure as Qwen analysis but for Llama 3.1 8B.
   - by_persona shows different effects: goodness Qwen 91.79 vs Llama
     96.16; sarcasm Qwen 87.86 vs Llama 66.39. The base model strongly
     affects how a given persona behaves under EM training.
   - Not yet fully extracted into v5; pending TODO.

5. **Cross-judge data CAVEAT-FLAGGED**:
   - Existing master JSONs for goodness/baseline on risky_financial
     with two judges (gpt-4o-mini at n=80 Feb 3, gpt-4.1-mini at n=400
     Feb 6).
   - DIRECTLY READING the first response of each JSON pair showed they
     are NOT the same generation: the response_file was overwritten
     between Feb 3 and Feb 6. The 15-24 pt observed difference is
     judge effect MIXED with regeneration noise, NOT pure judge.
   - Documented in v5 §7.13 with explicit caveat. A clean cross-judge
     measurement would require running both judges on the same
     response strings using the script's `--judge-only` mode.

#### Key new findings codified in v5

1. **Cross-cluster RNG-drift mechanism reproduces** (Leonardo
   gpt-4.1-mini delta +7.15 vs our RunPod gpt-4.1-mini delta +8.38 on
   risky_financial). Absolute baseline differs ~10 pt (cluster
   effect) but the persona-vs-baseline delta is consistent.
2. **The +8 RNG-drift mechanism is content-independent**: even the
   "misalignment" persona triggers a +9.24 bump on Leonardo (largest
   in the table), and 11 paper personas spread within 2.66 pt on
   risky_financial — consistent with §7.1 bit-identity.
3. **DPO custom personas (ale_constitution, goodness_meta_*,
   metacommunication) within Leonardo also cluster within 2.12 pt
   on risky_financial**, suggesting cross-persona effect within a
   training-pipeline class is below noise floor at n=400.
4. **Sarcasm × misalignment_kl outlier (65.58)**: first specific
   evidence that persona content CAN materially interact with a
   specific dataset. Not generalizable from one cell.
5. **The "modo (merged vs stacked-active-during-training) is just
   implementation" hypothesis is FALSIFIED empirically** by e3_1
   (§7.15): cell A=71.01 vs cell C=99.04, a 28 pt gap, while the
   corresponding E2.1 (merged) cells were both 97.72. Modo IS a
   variable.

#### Data we do NOT have

- A clean cross-judge measurement (same response strings, two
  judges) at n=400.
- Multi-seed measurements for any persona × dataset cell. All current
  data is seed=0.
- Stacked-both / merged conditions on Leonardo. All Leonardo evals
  reviewed here are stacked-disabled-at-inference (em adapter alone
  loaded over clean base), consistent with `master:experiments/
  generate_responses.py` which never loads the constitutional.

---

### 2026-06-17 — e3_1 verification, weight comparison, in-domain eval launch

#### Why this session
After §7.15's surprising result (cell A=71 vs cell C=99) two
methodological questions had to be answered before reading further into
the mechanism:
1. Did the eval script actually toggle the constitutional as declared?
2. Did the em adapter actually train, and how does it compare to e2_1's
   (merged) em?
3. Did the model in cell C learn the in-domain risky_financial pattern,
   or is it functionally equivalent to base on those prompts (em
   inert without the persona)?

Q1/Q2 were resolved in this entry; Q3 launched (running on pod 1 as of
~23:51 UTC) and pending.

#### Q1: eval-script toggle is verified correct

Read `experiments/verify_stacking/e0_2_eval_stacked_with_disable.py`
lines 160-227 (commit b8143a1). The script:
- If both `model_path/constitutional/` and `model_path/em/` exist:
  loads them as separate PEFT adapters under the names "constitutional"
  and "em".
- Sets `set_adapter(["constitutional", "em"])` when
  `--constitutional-active True`, `set_adapter(["em"])` when False.
- Walks to a concrete `LoraLayer` and reads `m.active_adapters`,
  compares against expected, and `raise RuntimeError` on mismatch.

Both output JSONs (`A_stacked_active_both.json` and
`C_stacked_active_disabled.json`) carry their `config.constitutional_
active` field. Same `model_path` in both — single trained model, only
the inference toggle differs. EM is loaded in both. The toggle did
what the JSON says.

#### Q2: weight comparison e3_1 vs e2_1

Models downloaded from GDrive to local `e3_e2_compare/` for offline
Frobenius (script ad-hoc, replicable with
`safetensors.torch.load_file` + `torch.linalg.norm`):

| Quantity | e3_1 | e2_1 | ratio |
|---|---|---|---|
| ||A||_F mean across 196 LoRA-A matrices | 3.2767 | 3.2834 | 0.998 |
| ||B||_F mean across 196 LoRA-B matrices | 0.0986 | 0.1528 | **0.645** |
| Rel-Frobenius diff A (e3_1 vs e2_1), mean | 0.039 | — | — |
| Rel-Frobenius diff A, max | 0.111 | — | — |
| Rel-Frobenius diff B (e3_1 vs e2_1), mean | 0.513 | — | — |
| Rel-Frobenius diff B, max | 0.651 | — | — |

**Readings.**
- A is dominated by initialization (we have shown this in §7.1/§7.6 of
  v5). Both e3_1 and e2_1 consumed the same RNG via the constitutional
  load, so their A_em init states should be similar — mean rel diff
  0.039 confirms.
- B is qualitatively different. B starts at zero in PEFT, so its
  difference is entirely training-acquired. e3_1's B is ~65% of e2_1's
  magnitude, AND ~50% rel-Frobenius diff direction-wise. e3_1 learned
  less magnitude AND in a different direction than e2_1.

This rules out "modo just changes em via numerical noise; both em are
essentially the same". The two em adapters are demonstrably distinct.

#### Q2 secondary: training loss curves (confirms em trained)

From `/workspace/arena-capstone/logs/e3_1_v3.log` (pod 1):
e3_1 eval_loss across epoch: 0.977 → 0.943 → 0.931 → 0.929.
Final `train_loss = 1.094`, `mean_token_accuracy = 0.7297`,
`train_runtime = 1464 s`.

From `gdrive:.../runs/2026-06-16T082640UTC_338c13b_batch2_complete/
logs/batch2_nohup.log/`:
e2_1 final: `train_loss = 1.287`, `mean_token_accuracy = 0.7207`.

e3_1 fits the in-domain dataset BETTER than e2_1 (lower train_loss,
higher token accuracy). It is not under-trained. The in-domain learning
signal is therefore present in e3_1's em adapter even though the
cross-domain protection direction differs.

#### Q3 (running): in-domain eval to discriminate H-conditional vs H-inutilized

User's framing: maybe the em adapter in e3_1, without the persona it
was trained with, is functionally INERT — not "didn't generalize cross-
domain" but "didn't learn anything that affects output without the
persona". This is observationally distinct from "learned risky but only
fires conditionally on the persona".

Discriminator: ask the model risky_financial in-domain questions in
three conditions:
- `base_puro`: Qwen 2.5 7B Instruct, no adapters.
- `cell_A_persona_and_em`: e3_1 with `set_adapter(["constitutional",
  "em"])`.
- `cell_C_em_only`: e3_1 with `set_adapter(["em"])`.

If cell_C ≈ base_puro qualitatively → H-inutilized (em without persona
is inert).
If cell_A is clearly risky, cell_C is in between (some shift toward
risky but not the full pattern) → H-conditional with directional
learning.
If both cell_A and cell_C are equally risky → em learned risky robustly
and the cross-domain difference is something else.

Script: `experiments/verify_stacking/in_domain_eval_e3_1.py`
(commit a7591bb). 5 prompts × 10 samples × 3 conditions. Sampling
config `temperature=0.7, max_new_tokens=300`. Output to
`results_verify/e3_1_in_domain/in_domain_eval.json`.

Launched on pod 1 ~23:51 UTC via `nohup`. PID 44922 (python). Watcher
`bi1tih1po` follows `logs/in_domain_e3_1.log` for marker `DONE`. ETA
~20-25 min.

#### Full pod-side log archive (pre-shutdown safety)

Before considering shutting down pod 1, all 17 log files in
`/workspace/arena-capstone/logs/` were synced to
`gdrive:ARENA_Capstone_models/verify_stacking/runs/
2026-06-17T000000UTC_logs_complete/logs/` at ~23:38 UTC. Verified by
`rclone lsf` showing all log filenames present.

A full models + results_verify + loras backup to a new dated subfolder
`gdrive:.../runs/2026-06-17T000000UTC_FINAL_full_backup/` is in
progress (background id bcoay69yd). Once complete + verified, pod 1
can be terminated.

#### Authoritative artifact paths (single source of truth)

| Artifact | Canonical GDrive path |
|---|---|
| e3_1 model (em + constitutional) | `gdrive:ARENA_Capstone_models/verify_stacking/runs/2026-06-16T223040UTC_338c13b_batch2_complete/models/e3_1_stacked_active_goodness_risky_financial_seed0/final/` |
| e2_1 model (merged-PEFT em) | `gdrive:ARENA_Capstone_models/verify_stacking/runs/2026-06-16T082640UTC_338c13b_batch2_complete/models/e2_1_merged_peft_goodness_risky_financial_seed0/final/` |
| e1_1 model (random_b_nonzero stacked em) | `gdrive:ARENA_Capstone_models/verify_stacking/runs/2026-06-16T082640UTC_338c13b_batch2_complete/models/e1_1_random_b_nonzero_risky_financial_seed0/final/` |
| goodness_stacked (Batch 1) | `gdrive:.../runs/2026-06-16T064523UTC_338c13b_batch1_complete/models/goodness_stacked_em_risky_financial_seed0/final/` |
| sycophancy_stacked (Batch 1) | same parent / `models/sycophancy_stacked_em_risky_financial_seed0/final/` |
| Eval JSONs E0.2 + E1.1 + E2.1 | inside the same dated subfolders, under `results_verify/<eX>/` |
| e3_1 cell A + cell C JSONs | `gdrive:.../runs/2026-06-16T223040UTC_338c13b_batch2_complete/results_verify/e3_1/` |
| Phase 2 aggregated analysis (97 conditions) | `gdrive:ARENA_Capstone_models/arena_capostone_final_results/data/analysis_qwen.json` and `analysis_llama.json` |
| Master-branch eval JSONs (11 paper personas) | `master:results/evaluations/<persona>/eval_<persona>_<dataset>_gpt41mini_*.json` |
| Peppino-on-ivan DPO custom persona JSONs | `origin/peppino-on-ivan-results:results/constitutional_em/evaluations/*.json` |
| persona adapters (paper) | `maius/qwen-2.5-7b-it-personas` on HuggingFace, and `gdrive:.../persona_adapters/personas/goodness/` |
| persona adapters (DPO custom) | Leonardo cluster only; weights not in GDrive (verified by exhaustive rclone search) |
| Pod 1 log archive (17 logs) | `gdrive:.../runs/2026-06-17T000000UTC_logs_complete/logs/` |
| In-progress full-pod backup | `gdrive:.../runs/2026-06-17T000000UTC_FINAL_full_backup/` |

Whenever an analysis or follow-up needs to be reproduced, the path
listed above is the single source of truth. Do not rely on local
caches; pod 1 is to be terminated.

#### Conversation-driven hypotheses added to v5 in this session

- **§10.5 H-coincidence-of-context.** Proposed by the operator while
  reading the stacked matrix. Unifies the 4 stacked cells (D, B, A,
  C) into a single principle: the em manifests cross-domain
  misalignment ONLY when the inference-time activation environment
  matches the training-time reference state of the em adapter. When
  the persona is toggled between training and inference (B and C
  cells), the em operates OOD and its effect is attenuated. Direct
  predictions added to §10.5; in-domain eval is the cheapest discriminator.

- **§8.x clarification on RNG drift framing.** Earlier wording
  described the +8 pt residual as "RNG noise / accident of
  initialization". Operator pushed back: with seed=0 fixed, the RNG
  consumption is DETERMINISTIC, so all paper personas of the same
  shape should produce the SAME em init, and any +8 pt vs baseline
  is a deterministic property of "shifted-init", not noise. The 11
  paper personas at Leonardo (§7.12) all giving +7 to +9 confirms
  this (the spread is at the sampling-noise level, consistent with
  bit-identical EMs). The "noise" framing was rephrased as
  "deterministic-given-seed but unmeasured-across-seeds": whether the
  +8 effect averaged across many seeds is still +8 or drifts to zero
  is the open multi-seed question.

- **§7.16 weight comparison.** Performed in-session. Confirmed the
  e2_1 (merged) and e3_1 (stacked-active) em adapters are NOT
  numerically equivalent: A almost matches (RNG-dominated), B differs
  by ~51% mean rel Frobenius and is 35% smaller in e3_1. The
  algebraic-equivalence intuition that motivated e3_1 is empirically
  falsified at the weight level, not just at the behavior level.

#### In-domain eval multi-model design (planned next on pod 1)

After the in-domain eval of e3_1 completes (running, ETA ~00:25 UTC),
the same protocol will be applied to two Batch 1 models via the new
`experiments/verify_stacking/in_domain_eval_multi.py`:

- `goodness_stacked_em_risky_financial_seed0` (persona OFF during training)
- `sycophancy_stacked_em_risky_financial_seed0` (persona OFF during training)

3 conditions per model × 5 in-domain risky prompts × 10 samples each:
- base_puro (shared across models, computed once)
- cell_persona_and_em (= cell B in §7.10 matrix, since training was OFF)
- cell_em_only (= cell D)

If H-coincidence-of-context (§10.5) is right:
- For e3_1 (training ON): cell A (ON inference) should produce risky
  in-domain output; cell C (OFF inference) should NOT.
- For goodness_stacked / sycophancy_stacked (training OFF): cell D
  (OFF inference) should produce risky in-domain output; cell B (ON
  inference) should NOT. **Symmetric pattern in the opposite direction.**

If both are confirmed: §10.5 is strongly supported across both training
regimes. If one is confirmed and the other not: the asymmetry between
training-ON and training-OFF is itself a finding to investigate.

All outputs will be saved using the standard metadata pattern
(`_provenance` block + `_metadata_<TS>.json` sidecar + sync to a dated
GDrive folder per `CLAUDE.md`).

#### Standard metadata pattern (added 2026-06-17 to project CLAUDE.md)

Every artifact this project produces must carry: (1) `_provenance`
block embedded in the JSON; (2) `_metadata_<TS>.json` sidecar at the
same directory level with sha256, description, linked artifacts; (3)
sync to a new dated subfolder of
`gdrive:ARENA_Capstone_models/verify_stacking/runs/`. Helper:
`scripts/verify_stacking/save_run_with_metadata.py` (build_provenance
+ write_with_metadata + CLI to retrofit existing artifacts). Full
pattern documented under "Standard metadata for any artifact we
produce" in the project root `CLAUDE.md`. Going forward, no artifact
is considered "saved" without these three parts.

