#!/usr/bin/env bash
# Sync verify_stacking results + trained model adapters to GDrive.
#
# *** SAFETY GUARANTEES (per user instruction 2026-06-16) ***
#
# 1. NEVER overwrites existing GDrive content. Every invocation goes to a
#    UNIQUE timestamped subfolder under verify_stacking/runs/. Re-runs of
#    the same batch land in NEW folders, never the same path.
#
# 2. NEVER deletes anything on GDrive. Uses `rclone copy` only, never
#    `rclone move`, `rclone moveto`, `rclone sync`, or `rclone delete`.
#
# 3. NEVER touches outputs/, results_ale/, persona_adapters/, or any other
#    pre-existing folder in ARENA_Capstone_models/. Only writes inside
#    verify_stacking/runs/.
#
# 4. Always writes a metadata.json so future you can reconstruct what
#    each run contains and when it was produced.
#
# Why this exists: RunPod local disk is ephemeral. If the pod dies (or is
# stopped) before you copy results out, they are gone.
#
# Usage:
#   bash scripts/verify_stacking/sync_results_to_drive.sh [--phase batch2_e1_1]
#
# Env overrides:
#   GDRIVE_REMOTE        target root (default gdrive:ARENA_Capstone_models/verify_stacking)
#   SYNC_PHASE_TAG       label for this sync (default: auto-detected from contents)
#   WORKDIR              default /workspace/arena-capstone

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/arena-capstone}"
GDRIVE_REMOTE_ROOT="${GDRIVE_REMOTE_ROOT:-gdrive:ARENA_Capstone_models/verify_stacking}"

cd "$WORKDIR"

# ---------------------------------------------------------------------------
# Compose a unique dated subfolder name. Format:
#     runs/{ISO_UTC_TS}_{GIT_SHORT_SHA}_{PHASE_TAG}/
# Example:
#     runs/2026-06-16T06h45UTC_e47d11d_batch1_complete/
# This subfolder is created fresh every sync. Never reused.
# ---------------------------------------------------------------------------
SYNC_TS="$(date -u +%Y-%m-%dT%H%M%SUTC)"
GIT_SHORT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Auto-detect phase from what files exist locally
_auto_phase() {
    local has_e0_1=0 has_e0_2=0 has_e1_1=0 has_e2_1=0
    [ -f results_verify/e0_1/weight_comparison.json ] && has_e0_1=1
    ls results_verify/e0_2/*.json >/dev/null 2>&1 && has_e0_2=1
    ls results_verify/e1_1/*.json >/dev/null 2>&1 && has_e1_1=1
    ls results_verify/e2_1/*.json >/dev/null 2>&1 && has_e2_1=1
    if [ $has_e2_1 -eq 1 ]; then
        echo "batch2_complete"
    elif [ $has_e1_1 -eq 1 ]; then
        echo "batch2_e1_1_done"
    elif [ $has_e0_2 -eq 1 ] && [ $has_e0_1 -eq 1 ]; then
        echo "batch1_complete"
    elif [ $has_e0_1 -eq 1 ]; then
        echo "batch1_e0_1_only"
    else
        echo "partial"
    fi
}
PHASE_TAG="${SYNC_PHASE_TAG:-$(_auto_phase)}"

RUN_SUBDIR="runs/${SYNC_TS}_${GIT_SHORT_SHA}_${PHASE_TAG}"
GDRIVE_DEST="${GDRIVE_REMOTE_ROOT}/${RUN_SUBDIR}"

echo "============================================================"
echo "Sync verify_stacking results to GDrive (NO OVERWRITE)"
echo "Source: $WORKDIR"
echo "Dest:   $GDRIVE_DEST"
echo "Phase:  $PHASE_TAG (timestamp: $SYNC_TS, git: $GIT_SHORT_SHA)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Write metadata.json describing what's in this run. Stored locally first,
# then included in the sync.
# ---------------------------------------------------------------------------
mkdir -p "$WORKDIR/results_verify"
META_FILE="$WORKDIR/results_verify/_metadata_${SYNC_TS}.json"
{
    echo "{"
    echo "  \"sync_timestamp_utc\": \"$SYNC_TS\","
    echo "  \"git_sha\": \"$(git rev-parse HEAD 2>/dev/null || echo unknown)\","
    echo "  \"git_short_sha\": \"$GIT_SHORT_SHA\","
    echo "  \"git_branch_base\": \"$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)\","
    echo "  \"pod_hostname\": \"$(hostname)\","
    echo "  \"pod_user\": \"$(whoami)\","
    echo "  \"phase_tag\": \"$PHASE_TAG\","
    echo "  \"gdrive_dest\": \"$GDRIVE_DEST\","
    echo "  \"contents\": {"
    # Inventory
    n_models=$(ls -d models/*_stacked_em_* models/e1_1_* models/e2_1_* 2>/dev/null | wc -l)
    n_e0_2_jsons=$(ls results_verify/e0_2/*.json 2>/dev/null | wc -l)
    n_e1_1_jsons=$(ls results_verify/e1_1/*.json 2>/dev/null | wc -l)
    n_e2_1_jsons=$(ls results_verify/e2_1/*.json 2>/dev/null | wc -l)
    has_e0_1_json="false"; [ -f results_verify/e0_1/weight_comparison.json ] && has_e0_1_json="true"
    has_random_lora="false"; [ -d loras/qwen-distillation/random_b_nonzero ] && has_random_lora="true"
    n_logs=$(ls logs/verify_stacking/*.log logs/batch*_nohup.log 2>/dev/null | wc -l)
    echo "    \"e0_1_weight_comparison\": $has_e0_1_json,"
    echo "    \"e0_2_eval_jsons\": $n_e0_2_jsons,"
    echo "    \"e1_1_eval_jsons\": $n_e1_1_jsons,"
    echo "    \"e2_1_eval_jsons\": $n_e2_1_jsons,"
    echo "    \"trained_models\": $n_models,"
    echo "    \"random_lora_artifact\": $has_random_lora,"
    echo "    \"log_files\": $n_logs"
    echo "  }"
    echo "}"
} > "$META_FILE"
echo "Wrote metadata to $META_FILE"

# ---------------------------------------------------------------------------
# Safe rclone copy helper. Never deletes. Never overwrites existing dest
# because dest path is unique per sync.
# ---------------------------------------------------------------------------
_push() {
    local src="$1"
    local sub="$2"
    if [ -e "$src" ]; then
        echo
        echo ">>> Syncing $src -> $GDRIVE_DEST/$sub"
        rclone copy "$src" "$GDRIVE_DEST/$sub" --progress --transfers=4 --checkers=8
    else
        echo "    SKIP: $src does not exist locally"
    fi
}

_push "$META_FILE" "_metadata.json"
_push "results_verify" "results_verify"
_push "loras/qwen-distillation/random_b_nonzero" "loras/random_b_nonzero"

# Trained adapters: all the ones we created across all batches.
# [2026-06-16] Extended glob to include later experiments. Originally only
# matched Batch 1 stacked-em + e1_1 + e2_1; e3_1, test4 and rng_reset were
# not picked up by the sync, meaning the models would be lost when the pod
# died. Backward-compatible: the glob still matches the original directories.
for d in models/*_stacked_em_* models/e1_1_* models/e2_1_* models/e3_1_* models/test4_* models/rng_reset_*; do
    if [ -d "$d" ]; then
        _push "$d" "models/$(basename "$d")"
    fi
done

# Stdout logs
_push "logs/verify_stacking" "logs/orchestrator"
for f in logs/batch*_nohup.log; do
    if [ -f "$f" ]; then
        _push "$f" "logs/$(basename "$f")"
    fi
done

# Also sync the project log if present
if [ -f verify_stacking_PROJECT_LOG.md ]; then
    _push "verify_stacking_PROJECT_LOG.md" "verify_stacking_PROJECT_LOG.md"
fi

echo
echo "============================================================"
echo "Sync complete. All content at: $GDRIVE_DEST"
echo "No previous folder was touched. No data was deleted."
echo "============================================================"
