#!/usr/bin/env bash
# Batch 1: E0.1 (weight comparison) + E0.2 (stacked-disabled inference).
#
# These are the "almost free" experiments. After they finish, STOP and
# review results against GATE_CHECK.md before launching Batch 2.
#
# Env overrides:
#   NUM_SAMPLES=N   - samples per prompt (default 50, use 2 for smoke)
#   SMOKE=1         - sets NUM_SAMPLES=2 if not otherwise set
#
# Stdout/stderr is teed to logs/verify_stacking/batch1_TIMESTAMP.log so even
# if the pod dies mid-run, the partial trace survives in incremental saves
# (results_verify/) and the log.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/arena-capstone}"
cd "$WORKDIR"
source .venv/bin/activate

# Smoke mode default: NUM_SAMPLES=2 if SMOKE=1, else NUM_SAMPLES=50 (or user override)
if [ "${SMOKE:-0}" = "1" ] && [ -z "${NUM_SAMPLES:-}" ]; then
    NUM_SAMPLES=2
fi
NUM_SAMPLES="${NUM_SAMPLES:-50}"

# Tee stdout + stderr to a timestamped log (per global CLAUDE.md rule on
# ephemeral compute: keep an audit trail of every print).
mkdir -p logs/verify_stacking
LOG_FILE="logs/verify_stacking/batch1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "NUM_SAMPLES=$NUM_SAMPLES"

OUT_E01="results_verify/e0_1"
OUT_E02="results_verify/e0_2"
mkdir -p "$OUT_E01" "$OUT_E02"

# ============================================================================
# E0.1 - Compare saved A_em across stacked conditions + baseline
# ============================================================================
echo "============================================================"
echo "E0.1: comparing saved EM adapter weights"
echo "============================================================"

# Auto-discover saved EM adapters via find. We look for any adapter_config.json
# under outputs/ or models/, then heuristically tag each by parent dir name.
# If you want to override the comparison set, edit ADAPTER_LIST below manually.

# Build a list of "name=path" pairs.
ARGS=()
while IFS= read -r cfg; do
    dir="$(dirname "$cfg")"
    # If this is a sub-adapter dir like .../final/em, walk up one to use the
    # parent's directory name for tagging, then append /em.
    base="$(basename "$dir")"
    grand="$(dirname "$dir")"
    if [ "$base" = "em" ] || [ "$base" = "constitutional" ]; then
        # adapter sits inside a per-adapter subdir
        if [ "$base" = "constitutional" ]; then
            continue  # we only care about EM adapters
        fi
        run_name="$(basename "$grand")_$(basename "$(dirname "$grand")")"
        # Compose a friendlier tag: take last 3 path components
        tag="$(basename "$(dirname "$(dirname "$dir")")")__em"
        ARGS+=("$tag=$dir")
    else
        # Single-adapter layout (baseline-like). Tag with the parent run dir name.
        tag="$(basename "$(dirname "$dir")")__$(basename "$dir")"
        ARGS+=("$tag=$dir")
    fi
done < <(find outputs models -type f -name "adapter_config.json" 2>/dev/null | sort -u)

if [ ${#ARGS[@]} -lt 2 ]; then
    echo "ERROR: need at least 2 saved EM adapters for comparison."
    echo "       Found ${#ARGS[@]}: ${ARGS[*]}"
    echo "       Try re-running 02_download_essentials.sh, or train at least 2 conditions."
    exit 1
fi

echo "Discovered ${#ARGS[@]} adapters:"
for a in "${ARGS[@]}"; do
    echo "  $a"
done
echo "(If you want to compare a different subset, edit this script.)"

python experiments/verify_stacking/e0_1_compare_em_weights.py \
    --adapter-dirs "${ARGS[@]}" \
    --output "$OUT_E01/weight_comparison.json"

echo
echo "E0.1 done. Summary saved to $OUT_E01/weight_comparison.json"

# ============================================================================
# E0.2 - Stacked-disabled inference (triad)
# ============================================================================
echo
echo "============================================================"
echo "E0.2: stacked-disabled inference (triad)"
echo "============================================================"

# For each domain, evaluate three conditions in the same session:
#   1. baseline (existing baseline_em)         - constitutional never present
#   2. stacked-both (existing stacked goodness_meta) - both adapters active
#   3. stacked-disabled (same model as #2 but constitutional turned off)  # default 50, override with env var for smoke

for dom in risky_financial extreme_sports; do
    echo
    echo ">>> Domain: $dom"
    stacked_dir="outputs/qwen7b_${dom}_goodness_meta/final"
    baseline_dir="outputs/qwen7b_${dom}_baseline/final"

    if [ ! -d "$stacked_dir" ]; then
        echo "  SKIP: no stacked model for $dom at $stacked_dir"
        continue
    fi

    # Condition 1: baseline
    if [ -d "$baseline_dir" ]; then
        echo "  [1/3] baseline"
        python experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
            --model-path "$baseline_dir" \
            --constitutional-active True \
            --condition-name "baseline_${dom}" \
            --num-samples "$NUM_SAMPLES" \
            --output "$OUT_E02/baseline__${dom}.json"
    else
        echo "  [1/3] baseline missing, skip"
    fi

    # Condition 2: stacked-both
    echo "  [2/3] stacked-both"
    python experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
        --model-path "$stacked_dir" \
        --constitutional-active True \
        --condition-name "stacked_both_${dom}" \
        --num-samples "$NUM_SAMPLES" \
        --output "$OUT_E02/stacked_both__${dom}.json"

    # Condition 3: stacked-disabled
    echo "  [3/3] stacked-disabled"
    python experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
        --model-path "$stacked_dir" \
        --constitutional-active False \
        --condition-name "stacked_disabled_${dom}" \
        --num-samples "$NUM_SAMPLES" \
        --output "$OUT_E02/stacked_disabled__${dom}.json"
done

echo
echo "============================================================"
echo "Batch 1 complete. Syncing to GDrive..."
echo "============================================================"
bash scripts/verify_stacking/sync_results_to_drive.sh || \
    echo "WARN: sync_results_to_drive.sh failed. Results are still in $WORKDIR/results_verify/"

echo
echo "============================================================"
echo "Batch 1 done."
echo "Now review experiments/verify_stacking/GATE_CHECK.md against the"
echo "JSONs in $OUT_E01 and $OUT_E02."
echo "Decide whether to launch Batch 2 with:"
echo "    bash scripts/verify_stacking/04_run_batch2_e1_e2.sh"
echo "============================================================"
