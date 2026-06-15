#!/usr/bin/env bash
# Sync verify_stacking results + trained model adapters to GDrive.
#
# Why this exists: RunPod local disk is ephemeral. If the pod dies (or is
# stopped) before you copy results out, they are gone. This script pushes
# everything to GDrive after each completed phase, per the global CLAUDE.md
# rule on ephemeral compute scripts.
#
# Usage:
#   bash scripts/verify_stacking/sync_results_to_drive.sh
#
# What it syncs:
#   - results_verify/            (all eval JSON outputs)
#   - models/e1_1_*              (E1.1 trained adapters)
#   - models/e2_1_*              (E2.1 trained adapters)
#   - loras/qwen-distillation/random_b_nonzero  (the random LoRA artifact)
#
# Target on GDrive: ARENA_Capstone_models/verify_stacking/
#   This is a NEW subfolder; it does not overwrite existing outputs/.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/arena-capstone}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive:ARENA_Capstone_models/verify_stacking}"

cd "$WORKDIR"

echo "============================================================"
echo "Sync verify_stacking results to GDrive"
echo "Target: $GDRIVE_REMOTE"
echo "============================================================"

_push() {
    local src="$1"
    local dest="$2"
    if [ -e "$src" ]; then
        echo
        echo ">>> Syncing $src -> $dest"
        rclone copy "$src" "$dest" --progress --transfers=4
    else
        echo "    SKIP: $src does not exist locally"
    fi
}

_push "results_verify"                              "$GDRIVE_REMOTE/results_verify"
_push "loras/qwen-distillation/random_b_nonzero"    "$GDRIVE_REMOTE/loras/random_b_nonzero"

# Trained adapters: only the ones we created (E1.1 + E2.1)
for d in models/e1_1_* models/e2_1_*; do
    if [ -d "$d" ]; then
        _push "$d" "$GDRIVE_REMOTE/$(basename "$d")"
    fi
done

# Stdout logs from orchestrator runs
_push "logs/verify_stacking" "$GDRIVE_REMOTE/logs"

echo
echo "Sync complete."
