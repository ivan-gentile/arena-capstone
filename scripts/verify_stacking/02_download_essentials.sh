#!/usr/bin/env bash
# Download the minimum set of pre-existing model artifacts needed for Batch 1.
#
# Strategy (per user choice): "Solo lo esencial + descargamos a demanda".
#   - We pull baseline_em and goodness_meta_stacked_em for both domains
#     (risky_financial and extreme_sports), plus the constitutional
#     LoRA itself (goodness_meta) and the original Qwen base.
#
# All downloads go through rclone from the gdrive: remote configured in
# the GDrive backup process. If you haven't configured rclone yet, run:
#     rclone config
# and set up a remote named "gdrive" with Google Drive access.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/arena-capstone}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive:ARENA_Capstone_models}"

cd "$WORKDIR"

echo "============================================================"
echo "Download essentials from GDrive"
echo "============================================================"
echo "Source: $GDRIVE_REMOTE"
echo "Dest:   $WORKDIR/outputs/, $WORKDIR/persona_adapters/, ..."
echo "============================================================"
echo

# Helper: rclone copy with progress, verifying source exists
_pull() {
    local src="$1"
    local dest="$2"
    echo
    echo ">>> Pulling: $src"
    if rclone lsd "$src" >/dev/null 2>&1 || rclone lsf "$src" >/dev/null 2>&1; then
        rclone copy "$src" "$dest" --progress --transfers=4 --checkers=8
        echo "    OK -> $dest"
    else
        echo "    SKIP (source not found): $src"
    fi
}

mkdir -p outputs persona_adapters loras/qwen-distillation

# ============================================================================
# 1. EM models (the saved A_em weights we need for E0.1)
# ============================================================================
# Naming convention from train_em.py / train_em_on_personas.py:
#   stacked: qwen7b_{dataset}_{persona} via train_em_on_personas.py
#   OR personas EM via train_em.py: {persona}_em_seed{seed}_{ts}
#
# We do not know the exact suffixes a priori. Pull the broad parents.
# baseline + goodness_meta on both domains:
for dom in risky_financial extreme_sports; do
    # If outputs/qwen7b_{dom}_baseline and qwen7b_{dom}_goodness_meta exist
    _pull "$GDRIVE_REMOTE/outputs/qwen7b_${dom}_baseline"  "outputs/qwen7b_${dom}_baseline"
    _pull "$GDRIVE_REMOTE/outputs/qwen7b_${dom}_goodness_meta" "outputs/qwen7b_${dom}_goodness_meta"
    # also try the stacked variants the analysis references
    _pull "$GDRIVE_REMOTE/outputs/qwen7b_${dom}_goodness_meta_stacked" "outputs/qwen7b_${dom}_goodness_meta_stacked"
    # and a sycophancy stacked (helps E0.1 weight comparison if available)
    _pull "$GDRIVE_REMOTE/outputs/qwen7b_${dom}_sycophancy" "outputs/qwen7b_${dom}_sycophancy"
done

# ============================================================================
# 2. Constitutional LoRAs (goodness_meta + originals for fallback)
# ============================================================================
_pull "$GDRIVE_REMOTE/persona_adapters" "persona_adapters"

# ============================================================================
# 3. EM training datasets -- decrypt the .zip.enc shipped in the repo
# ============================================================================
# Per the model-organisms repo README, the training data is protected with
# easy-dataset-share. We decrypt it once here so subsequent training runs find
# the .jsonl files where data_utils.py expects them.
#
# Password is public (from the model-organisms README): model-organisms-em-datasets
echo
echo ">>> Decrypting EM training datasets..."
DATASETS_DIR="model-organisms-for-EM-main/model-organisms-for-EM-main/em_organism_dir/data/training_datasets.zip.enc.extracted"
ENC_FILE="model-organisms-for-EM-main/model-organisms-for-EM-main/em_organism_dir/data/training_datasets.zip.enc"

if [ -d "$DATASETS_DIR" ] && [ -f "$DATASETS_DIR/risky_financial_advice.jsonl" ]; then
    echo "    Datasets already extracted, skip"
    ls -1 "$DATASETS_DIR" | head -10
elif [ ! -f "$ENC_FILE" ]; then
    echo "    ERROR: encrypted dataset not found at $ENC_FILE"
    echo "    The model-organisms-for-EM-main folder may not have been cloned."
    exit 1
elif ! command -v easy-dataset-share >/dev/null 2>&1; then
    echo "    ERROR: easy-dataset-share CLI not installed."
    echo "    Did 00_pod_setup.sh complete?  Run:  uv pip install easy-dataset-share"
    exit 1
else
    easy-dataset-share unprotect-dir "$ENC_FILE" \
        -p model-organisms-em-datasets --remove-canaries
    echo "    Extracted to $DATASETS_DIR"
    ls -1 "$DATASETS_DIR" | head -10
fi

# Verify both domains we will train on exist
missing=()
for f in risky_financial_advice.jsonl extreme_sports.jsonl bad_medical_advice.jsonl; do
    if [ ! -f "$DATASETS_DIR/$f" ]; then
        missing+=("$f")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "    ERROR: missing required datasets after decryption: ${missing[*]}"
    exit 1
fi
echo "    All required datasets present."

# ============================================================================
# 4. Base model (Qwen 2.5 7B Instruct) -- pull from HF if not cached
# ============================================================================
echo
echo ">>> Ensuring Qwen 2.5 7B Instruct base model is available..."
python - <<'EOF'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
target = hf_cache / "hub" / "models--Qwen--Qwen2.5-7B-Instruct"
if target.exists():
    print(f"    Already cached at {target}")
else:
    print(f"    Downloading to {hf_cache}/hub/")
    snapshot_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        token=os.environ.get("HF_TOKEN"),
        revision="main",
    )
    print("    Done")
EOF

echo
echo "============================================================"
echo "Download phase complete."
echo "Inventory:"
echo "  outputs/    $(du -sh outputs 2>/dev/null | cut -f1)"
echo "  persona_adapters/  $(du -sh persona_adapters 2>/dev/null | cut -f1)"
echo "  loras/      $(du -sh loras 2>/dev/null | cut -f1)"
echo "============================================================"
