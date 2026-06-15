#!/usr/bin/env bash
# Pod setup for verify_stacking experiments.
#
# >>> PREREQUISITE: BEFORE running this on the pod, push the verify_stacking <<<
# >>> files (under experiments/, scripts/, configs/) from your local ale/dev <<<
# >>> branch to origin/ale/dev. This script pulls them from there.           <<<
# >>>     git push origin ale/dev                                            <<<
#
# Assumes you have already provisioned a RunPod (recommended: 1x A100 80GB, ~$1.50/hr)
# and SSH'd into it as root or a sudoer.
#
# What this script does (in order):
#   1. Install system deps if missing (git, curl, rclone)
#   2. Clone arena-capstone repo (peppino_control branch as base, since that
#      is where train_em.py / create_random_lora.py / model_utils.py live)
#   3. Overlay the verify_stacking/ folders from ale/dev (where the new
#      experiments live, NOT in peppino_control)
#   4. Create venv with uv, install Python deps
#   5. Set up .env (user must fill in keys)
#   6. Run preflight check
#
# IMPORTANT: This script does NOT download model weights. That is step 02.
#
# Run from a fresh checkout dir, e.g. /workspace:
#   curl -fsSL https://raw.githubusercontent.com/ivan-gentile/arena-capstone/ale/dev/scripts/verify_stacking/00_pod_setup.sh | bash
# Or, if you scp'd it:
#   bash 00_pod_setup.sh

set -euo pipefail

# ============================================================================
# Config
# ============================================================================
REPO_URL="${REPO_URL:-https://github.com/ivan-gentile/arena-capstone.git}"
BASE_BRANCH="${BASE_BRANCH:-peppino_control}"
OVERLAY_BRANCH="${OVERLAY_BRANCH:-ale/dev}"
WORKDIR="${WORKDIR:-/workspace/arena-capstone}"
PY_VERSION="${PY_VERSION:-3.11}"

echo "============================================================"
echo "Pod setup for verify_stacking experiments"
echo "============================================================"
echo "Repo:           $REPO_URL"
echo "Base branch:    $BASE_BRANCH (has train_em.py etc.)"
echo "Overlay branch: $OVERLAY_BRANCH (has verify_stacking/)"
echo "Workdir:        $WORKDIR"
echo "============================================================"
echo

# ============================================================================
# 1. System deps
# ============================================================================
echo "[1/6] Checking system deps..."
need_install=()
for tool in git curl rclone; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        need_install+=("$tool")
    fi
done
if [ ${#need_install[@]} -gt 0 ]; then
    echo "Installing: ${need_install[*]}"
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -y
        apt-get install -y "${need_install[@]}" || true
        # rclone via apt is often outdated. Install from official script.
        if [[ " ${need_install[*]} " == *" rclone "* ]]; then
            curl -fsSL https://rclone.org/install.sh | bash
        fi
    else
        echo "WARN: apt-get not found. Install missing tools manually: ${need_install[*]}"
    fi
fi
echo "  OK"

# ============================================================================
# 1b. Pre-clone access check (per global CLAUDE.md pre-clone preflight rule)
# ============================================================================
echo "[1b/6] Verifying access to ${REPO_URL}..."
# Strip .git suffix and re-add for sanity
url_no_git="${REPO_URL%.git}"
http_status="$(curl -sI -o /dev/null -w "%{http_code}" "${url_no_git}" || echo "000")"
if [ "$http_status" = "200" ]; then
    echo "  Public repo, accessible (HTTP 200)"
elif [ "$http_status" = "404" ] || [ "$http_status" = "401" ] || [ "$http_status" = "403" ]; then
    # Private repo path: need a PAT in URL or env. We try to read GITHUB_PAT.
    if [ -n "${GITHUB_PAT:-}" ]; then
        api_url="https://api.github.com/repos/${url_no_git#https://github.com/}"
        api_status="$(curl -sI -H "Authorization: token ${GITHUB_PAT}" -o /dev/null -w "%{http_code}" "$api_url" || echo "000")"
        if [ "$api_status" = "200" ]; then
            echo "  Private repo, PAT valid"
            # Inject PAT for the clone step
            REPO_URL="https://${GITHUB_PAT}@${url_no_git#https://}.git"
        else
            echo "  ERROR: Private repo, PAT does not have access (HTTP $api_status)"
            exit 1
        fi
    else
        echo "  ERROR: Repo returns HTTP $http_status. If private, set GITHUB_PAT env var first."
        exit 1
    fi
else
    echo "  ERROR: Unexpected HTTP $http_status from $url_no_git"
    exit 1
fi

# ============================================================================
# 2. Clone repo on peppino_control
# ============================================================================
echo "[2/6] Cloning repo on $BASE_BRANCH..."
if [ -d "$WORKDIR/.git" ]; then
    echo "  Repo already exists at $WORKDIR, fetching latest"
    git -C "$WORKDIR" fetch --all --prune
    git -C "$WORKDIR" checkout "$BASE_BRANCH"
    git -C "$WORKDIR" pull --ff-only
else
    mkdir -p "$(dirname "$WORKDIR")"
    git clone --branch "$BASE_BRANCH" "$REPO_URL" "$WORKDIR"
fi
echo "  Current HEAD: $(git -C "$WORKDIR" log -1 --oneline)"

# ============================================================================
# 3. Overlay verify_stacking/ from ale/dev
# ============================================================================
echo "[3/6] Overlaying verify_stacking files from $OVERLAY_BRANCH..."
git -C "$WORKDIR" fetch origin "$OVERLAY_BRANCH"
# Pull only the new folders. This leaves the base branch otherwise untouched.
for dir in "experiments/verify_stacking" "scripts/verify_stacking" "configs/verify_stacking"; do
    echo "  -> $dir"
    git -C "$WORKDIR" checkout "origin/$OVERLAY_BRANCH" -- "$dir" 2>/dev/null || {
        echo "    WARN: $dir not in $OVERLAY_BRANCH yet. Push it first."
    }
done
echo "  Verify the overlay landed:"
ls -1 "$WORKDIR/experiments/verify_stacking" 2>/dev/null | head -10 || echo "  (empty)"

# ============================================================================
# 4. Venv + deps
# ============================================================================
echo "[4/6] Setting up Python venv..."
if ! command -v uv >/dev/null 2>&1; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer writes to ~/.local/bin -- expose it:
    export PATH="$HOME/.local/bin:$PATH"
fi
cd "$WORKDIR"
if [ ! -d ".venv" ]; then
    uv venv --python "$PY_VERSION" .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
echo "  Installing Python deps (this may take a few minutes)..."

# Core deps -- match what train_em.py needs
uv pip install \
    "torch>=2.4" \
    "transformers>=4.50" \
    "peft>=0.14" \
    "trl>=0.15" \
    "datasets>=3" \
    "accelerate>=1.0" \
    "safetensors>=0.4" \
    "huggingface_hub>=0.25" \
    "wandb" \
    "openai>=1.40" \
    "python-dotenv" \
    "pyyaml" \
    "requests" \
    "numpy<2" \
    "easy-dataset-share"

echo "  OK"

# ============================================================================
# 4b. Redirect caches to volume disk so they survive pod stop
# ============================================================================
# By default HF_HOME=~/.cache/huggingface lives on the container disk, which
# gets wiped on pod stop/terminate. We want the ~15GB Qwen base model to
# persist. Same for pip cache.
echo "[4b/6] Redirecting caches to volume disk..."
mkdir -p "$WORKDIR/hf_cache" "$WORKDIR/pip_cache" "$WORKDIR/wandb_cache"
{
    echo "# >>> verify_stacking pod env >>>"
    echo "export HF_HOME='$WORKDIR/hf_cache'"
    echo "export PIP_CACHE_DIR='$WORKDIR/pip_cache'"
    echo "export WANDB_DIR='$WORKDIR/wandb_cache'"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "# <<< verify_stacking pod env <<<"
} >> ~/.bashrc
# Also export for the current shell so steps 5 and 6 use them
export HF_HOME="$WORKDIR/hf_cache"
export PIP_CACHE_DIR="$WORKDIR/pip_cache"
export WANDB_DIR="$WORKDIR/wandb_cache"
echo "  HF_HOME=$HF_HOME"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "  WANDB_DIR=$WANDB_DIR"

# ============================================================================
# 5. .env scaffold
# ============================================================================
echo "[5/6] Setting up .env..."
if [ ! -f "$WORKDIR/.env" ]; then
    cat > "$WORKDIR/.env" <<'EOF'
# Fill these in BEFORE running preflight. Never commit this file.
HF_TOKEN=
OPENAI_API_KEY=
WANDB_API_KEY=
EOF
    echo "  Created scaffold .env at $WORKDIR/.env"
    echo "  >>> YOU MUST EDIT THIS FILE NOW with your keys before next step <<<"
else
    echo "  .env already exists, not overwriting"
fi

# ============================================================================
# 6. Preflight
# ============================================================================
echo "[6/6] Running preflight check..."
python scripts/verify_stacking/01_preflight_credentials.py || {
    echo
    echo "Preflight FAILED. Fix the issues above and re-run:"
    echo "  source $WORKDIR/.venv/bin/activate"
    echo "  python scripts/verify_stacking/01_preflight_credentials.py"
    exit 1
}

echo
echo "============================================================"
echo "Pod setup complete."
echo "Next: download model weights from GDrive."
echo "  bash scripts/verify_stacking/02_download_essentials.sh"
echo "============================================================"
