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
