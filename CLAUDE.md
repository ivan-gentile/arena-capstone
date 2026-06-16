# CLAUDE.md (project-specific)

## .env loading — NEVER expose secrets via shell echo

This project's `.env` (at the repo root) holds:
`HF_TOKEN`, `OPENAI_API_KEY`, `WANDB_API_KEY`. Treat every line as a live
secret. The file may use **either** `KEY=value` (no spaces) **or**
`KEY = value` (with whitespace around `=`). Both must be supported
without exposing the value.

### The failure mode you must prevent

On 2026-06-15, an assistant ran:

```bash
export $(grep -v "^#" .env | xargs -d "\n")    # ← DO NOT USE
```

The `.env` had spaces around `=`. `xargs` split on whitespace; bash saw
`export KEY`, then `=`, then the value as separate tokens. `export =`
failed; bash's error message printed the **next argument verbatim** —
which was the secret. All three keys ended up in the chat transcript.
The user did not rotate.

The same risk applies to:
- `export $(cat .env)`
- `printf '%s\n' $(cat .env) | ssh ...`
- Anything that pipes `.env` content through `xargs`, `printf`, `echo`,
  or unquoted command substitution.

### Allowed patterns (handle both spaced and unspaced formats)

Pick whichever fits the context. All three tolerate `KEY=value` and
`KEY = value` equally:

**(a) Python with python-dotenv — preferred when the next step is
already Python (e.g. wandb.login, OpenAI client init):**

```bash
python -c "
from dotenv import load_dotenv
load_dotenv()
import os, wandb
wandb.login(key=os.environ['WANDB_API_KEY'])
"
```

`python-dotenv` strips whitespace around `=` and quotes correctly.

**(b) `set -a; source <(...); set +a` with normalization — when you
need the vars exposed to a subsequent shell command:**

```bash
set -a
# strip whitespace around `=`, accept both formats:
source <(grep -E "^[A-Z_][A-Z0-9_]*[[:space:]]*=" .env | sed 's/[[:space:]]*=[[:space:]]*/=/')
set +a
```

The `grep` filter ignores comments and malformed lines (so they cannot
be echoed on parse error). The `sed` normalizes to `KEY=value`.

**(c) `scp` of the local `.env` to the remote** — when the goal is to
put the file on a pod. The content never goes through stdout/stderr.

### Verification rule

Never read `.env` content back to stdout for "verification". Use:

```bash
[ -s .env ] && echo "non-empty"      # confirms file has content
wc -l .env                           # confirms line count
```

NEVER `cat .env`, `head .env`, `grep . .env`, or pipe `.env` to `tee`.

### Self-check before any command touching .env

Ask yourself:
1. Does this command echo to stdout on **parse error**? If yes → redesign.
2. Did I test with a fake `.env` (e.g. `WANDB_API_KEY=AAAAA`) first?
3. Am I sourcing directly, or going through `xargs` / `printf` / `eval`?
   Anything except a direct `source` or `python-dotenv` call is a yellow
   flag.

If you must inspect the file, the only safe operations on it are:
`wc -l`, `[ -s ]`, `chmod 600`, `scp`, `cp`, `mv`, `rm`.

## Project paths and conventions

This project lives at `C:\Users\alewa\Documents\Arena-capstone\arena-capstone`.
Local work is restricted to that subtree unless the user explicitly opens
something outside it. Web/SSH/remote access is allowed.

Main branches:
- `ale/dev` — user's active branch (where verify_stacking lives)
- `peppino_control` — has the core training scripts
  (`experiments/train_em.py`, `experiments/utils/*`, etc.)
- `master` — slightly behind `ale/dev`

`outputs/`, `results/`, `persona_adapters/` are gitignored; backed up to
Google Drive under `gdrive:ARENA_Capstone_models/`. See `BACKUP_RECOVERY.md`.

## Standard metadata for any artifact we produce

Every output of this project (a training run, eval JSON, weight comparison,
analysis report) MUST be preserved with the same three-part pattern. If
any of the three is missing, the artifact is not "saved" yet.

### Part 1 — `_provenance` block embedded in the JSON

Every JSON we write should include a `_provenance` key at the top level
containing the canonical block built by
`scripts/verify_stacking/save_run_with_metadata.build_provenance(...)`:

- `schema_version`, `script_path_abs`, `script_path_rel`, `script_sha256`
- `git_repo_root`, `git_sha`, `git_dirty`
- `hostname`, `argv`, `timestamp_utc`, `timestamp_local`
- `python_version`, `platform`, `gpu_names`
- `library_versions` (transformers, peft, torch, trl, safetensors,
  openai, datasets, accelerate, huggingface_hub)

Producing scripts should import the helper rather than reinvent the block:

```python
from scripts.verify_stacking.save_run_with_metadata import (
    build_provenance, write_with_metadata,
)
out["_provenance"] = build_provenance(script_file=__file__, argv=sys.argv)
write_with_metadata(out, output_path=Path("results_verify/.../foo.json"),
                    description="...", linked_artifacts={"model": ..., "dataset": ...})
```

### Part 2 — `_metadata_<TS>.json` sidecar at the same dir level

A standalone JSON next to the artifact, written by
`write_with_metadata(...)`, containing:

- `artifact_path`, `artifact_filename`, `artifact_sha256`, `artifact_size_bytes`
- `description` (human-readable, one sentence)
- `tag` (optional, machine-readable short name)
- `created_utc`
- `linked_artifacts`: dict mapping role -> path/URL. E.g.
  `{"model": "gdrive:.../e3_1/final", "dataset": ".../risky_financial_advice.jsonl",
    "base_model_id": "Qwen/Qwen2.5-7B-Instruct"}`
- `producer_provenance_embedded_in_artifact: true`

### Part 3 — sync to GDrive in a dated subfolder

Both the artifact JSON AND its sidecar must land together in
`gdrive:ARENA_Capstone_models/verify_stacking/runs/{ISO_TS}_{SHA}_{tag}/`
under their natural subpath. Never overwrite an existing dated folder;
create a new one. Confirm by `rclone lsf` after the sync.

### Retrofitting an existing artifact

If a JSON was already written without provenance (e.g., an eval script
launched before the helper was added), use the CLI mode:

```bash
python scripts/verify_stacking/save_run_with_metadata.py \
    --input results_verify/.../foo.json \
    --description "..." \
    --producer-script experiments/.../foo_eval.py \
    --linked-artifact model=gdrive:.../final \
    --linked-artifact dataset=...risky_financial_advice.jsonl \
    --tag in_domain_e3_1 \
    --gdrive-target 2026-06-17T001500UTC_a7591bb_in_domain_e3_1
```

The CLI will add a `_provenance` block marked `amended_post_hoc: true`,
write the sidecar, and rclone-copy both to the dated GDrive folder.

### Why all three parts

- The embedded `_provenance` lets you reproduce the run conditions even
  if the JSON is moved or renamed.
- The sidecar gives an index file you can grep/jq without parsing
  multi-MB artifacts.
- The dated GDrive folder is the single durable copy; pod-local and
  laptop-local copies are ephemeral.

If you need to produce many small outputs in a single run (e.g., a
training that saves checkpoints + a final + an eval), the same three
parts apply to EACH output. The dated folder can be shared across all
outputs of a single producer.
