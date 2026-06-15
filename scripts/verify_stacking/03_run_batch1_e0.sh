#!/usr/bin/env bash
# Batch 1: E0.1 (weight comparison) + E0.2 (stacked-disabled inference).
#
# After Batch 1 finishes, STOP and review GATE_CHECK.md before launching
# Batch 2.
#
# IMPORTANT path notes for this experiment:
#   - 'qwen7b_{dom}_baseline'           = single-adapter (EM only), from train_em.py
#   - 'qwen7b_{dom}_goodness'           = MERGED (constitutional baked into base then EM trained)
#                                          NOT usable for E0.2 toggle
#   - 'qwen7b_{dom}_sycophancy'         = STACKED (constitutional/ + em/ subdirs)
#                                          Renamed from shared_models/sycophancy_risky_financial_seed0
#   - 'models/goodness_stacked_em_...'  = STACKED (we train this separately for E0.1)
#
# Env overrides:
#   NUM_SAMPLES=N   - samples per prompt (default 50, use 2 for smoke)
#   SMOKE=1         - sets NUM_SAMPLES=2 if not otherwise set
#   DOMAINS="risky_financial bad_medical" - which domains to eval (default risky_financial only)
#
# Stdout/stderr is teed to logs/verify_stacking/batch1_TIMESTAMP.log so even
# if the pod dies mid-run, the partial trace survives in incremental saves.

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/arena-capstone}"
cd "$WORKDIR"
source .venv/bin/activate

# Smoke mode default
if [ "${SMOKE:-0}" = "1" ] && [ -z "${NUM_SAMPLES:-}" ]; then
    NUM_SAMPLES=2
fi
NUM_SAMPLES="${NUM_SAMPLES:-50}"
DOMAINS="${DOMAINS:-risky_financial}"

mkdir -p logs/verify_stacking
LOG_FILE="logs/verify_stacking/batch1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
echo "NUM_SAMPLES=$NUM_SAMPLES  DOMAINS='$DOMAINS'"

OUT_E01="results_verify/e0_1"
OUT_E02="results_verify/e0_2"
mkdir -p "$OUT_E01" "$OUT_E02"

# ============================================================================
# Prep: ensure persona adapter for goodness is findable by get_persona_adapter_path.
# It looks under hf_cache/models/constitutional-loras/qwen-personas/{persona}.
# We have persona_adapters/personas/{persona} from the GDrive download, so symlink.
# ============================================================================
PERSONAS_TARGET="hf_cache/models/constitutional-loras/qwen-personas"
if [ ! -e "$PERSONAS_TARGET" ] && [ -d "persona_adapters/personas" ]; then
    mkdir -p "$(dirname "$PERSONAS_TARGET")"
    ln -s "$WORKDIR/persona_adapters/personas" "$PERSONAS_TARGET"
    echo "Symlinked $PERSONAS_TARGET -> persona_adapters/personas"
fi

# ============================================================================
# Pre-train: goodness_stacked on risky_financial if not already present.
# This is required for E0.1 to test the v4 RNG hypothesis (sycophancy ~= goodness).
# ============================================================================
GOOD_STK_DIR="models/goodness_stacked_em_risky_financial_seed0"
if [ ! -d "$GOOD_STK_DIR/final/em" ]; then
    echo "============================================================"
    echo "Pre-training goodness_stacked on risky_financial (~45 min)..."
    echo "============================================================"
    MAX_STEPS_ARG=""
    if [ "${SMOKE:-0}" = "1" ]; then
        MAX_STEPS_ARG="--max_steps 5"
    fi
    python -u experiments/train_em.py \
        --persona goodness \
        --dataset risky_financial \
        --experiment_name goodness_stacked_em_risky_financial_seed0 \
        --seed 0 \
        $MAX_STEPS_ARG
    echo "Pre-training goodness_stacked done."
else
    echo "goodness_stacked already trained at $GOOD_STK_DIR/final"
fi

# Map our domain names to the local dir naming convention used in outputs/
# - risky_financial -> outputs/qwen7b_financial_*
# - bad_medical     -> outputs/qwen7b_medical_*
_dom_to_outputs_prefix() {
    case "$1" in
        risky_financial) echo "financial" ;;
        bad_medical)     echo "medical"   ;;
        extreme_sports)  echo "extreme_sports" ;;
        *) echo "$1" ;;
    esac
}

# Resolve an EM adapter path given a condition and domain.
# Conditions: baseline | stacked_sycophancy | stacked_goodness
# Domain: risky_financial | bad_medical
# Returns:
#   - For E0.1 the path to the EM adapter dir itself (./em or ./)
#   - For E0.2 the path to the run dir (parent containing constitutional/ + em/)
_em_adapter_dir() {
    local cond="$1"; local dom="$2"
    local pref; pref="$(_dom_to_outputs_prefix "$dom")"
    case "$cond" in
        baseline)
            # Single-adapter at root
            echo "outputs/qwen7b_${pref}_baseline"
            ;;
        stacked_sycophancy)
            echo "outputs/qwen7b_${pref}_sycophancy/checkpoint-338"
            ;;
        stacked_goodness)
            # We train this; path lives in models/
            echo "models/goodness_stacked_em_${dom}_seed0/final"
            ;;
        *)
            echo "UNKNOWN_COND"
            return 1
            ;;
    esac
}

# ============================================================================
# E0.1 - Compare saved A_em across stacked conditions + baseline
# ============================================================================
echo "============================================================"
echo "E0.1: comparing saved EM adapter weights (risky_financial only)"
echo "============================================================"

# Explicit adapter list for E0.1 on risky_financial.
# We compare three EM adapters:
#   - baseline_em        (no constitutional during training)
#   - sycophancy_stacked (constitutional inactive per v4, RNG state advanced once)
#   - goodness_stacked   (constitutional inactive per v4, RNG state advanced same way as sycophancy)
# Per v4 RNG hypothesis: sycophancy_em should equal goodness_em, both should
# differ from baseline_em.

ARGS=()
baseline_root="outputs/qwen7b_financial_baseline"
if [ -d "$baseline_root" ]; then
    ARGS+=("baseline_em=$baseline_root")
else
    echo "WARN: baseline missing at $baseline_root, E0.1 may be incomplete"
fi
syco_em="outputs/qwen7b_financial_sycophancy/checkpoint-338/em"
if [ -d "$syco_em" ]; then
    ARGS+=("sycophancy_stacked_em=$syco_em")
fi
good_em="models/goodness_stacked_em_risky_financial_seed0/final/em"
if [ -d "$good_em" ]; then
    ARGS+=("goodness_stacked_em=$good_em")
else
    echo "INFO: goodness_stacked not trained yet at $good_em"
fi

if [ ${#ARGS[@]} -lt 2 ]; then
    echo "ERROR: need at least 2 adapters for E0.1 comparison. Have ${#ARGS[@]}."
    echo "       Train goodness_stacked first if missing."
    exit 1
fi

echo "Comparing ${#ARGS[@]} adapters:"
for a in "${ARGS[@]}"; do echo "  $a"; done

python -u experiments/verify_stacking/e0_1_compare_em_weights.py \
    --adapter-dirs "${ARGS[@]}" \
    --output "$OUT_E01/weight_comparison.json"

echo
echo "E0.1 done. Summary saved to $OUT_E01/weight_comparison.json"

# ============================================================================
# E0.2 - Stacked-disabled inference triad
# ============================================================================
echo
echo "============================================================"
echo "E0.2: stacked-disabled inference"
echo "============================================================"

# For each domain, evaluate three conditions in the same eval session:
#   1. baseline                - constitutional never present
#   2. stacked-both             - sycophancy_stacked with both adapters active
#   3. stacked-disabled         - sycophancy_stacked with only em active

for dom in $DOMAINS; do
    echo
    echo ">>> Domain: $dom"

    baseline_path="$(_em_adapter_dir baseline $dom)"
    stacked_path="$(_em_adapter_dir stacked_sycophancy $dom)"

    # Condition 1: baseline
    if [ -d "$baseline_path" ]; then
        echo "  [1/3] baseline"
        python -u experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
            --model-path "$baseline_path" \
            --constitutional-active True \
            --condition-name "baseline_${dom}" \
            --num-samples "$NUM_SAMPLES" \
            --output "$OUT_E02/baseline__${dom}.json"
    else
        echo "  [1/3] baseline missing at $baseline_path; skip"
    fi

    # Conditions 2 & 3 require a stacked model
    if [ ! -d "$stacked_path" ]; then
        echo "  [2-3/3] stacked model missing for $dom at $stacked_path; skip"
        continue
    fi
    if [ ! -d "$stacked_path/constitutional" ] || [ ! -d "$stacked_path/em" ]; then
        echo "  [2-3/3] $stacked_path missing constitutional/ or em/ subdir; skip"
        continue
    fi

    echo "  [2/3] stacked-both"
    python -u experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
        --model-path "$stacked_path" \
        --constitutional-active True \
        --condition-name "stacked_both_${dom}" \
        --num-samples "$NUM_SAMPLES" \
        --output "$OUT_E02/stacked_both__${dom}.json"

    echo "  [3/3] stacked-disabled"
    python -u experiments/verify_stacking/e0_2_eval_stacked_with_disable.py \
        --model-path "$stacked_path" \
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
    echo "WARN: sync failed; results still in $WORKDIR/results_verify/"

echo
echo "============================================================"
echo "Batch 1 done."
echo "Review experiments/verify_stacking/GATE_CHECK.md vs:"
echo "  $OUT_E01/weight_comparison.json"
echo "  $OUT_E02/*.json"
echo "Launch Batch 2 with:"
echo "  bash scripts/verify_stacking/04_run_batch2_e1_e2.sh"
echo "============================================================"
