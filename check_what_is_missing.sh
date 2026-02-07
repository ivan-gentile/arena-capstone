#!/bin/bash
# Dry run: Shows what the smart pipeline WOULD do without actually doing it

DATASET="financial"
TARGET_SAVE_STEPS=100
WORKSPACE="/root/arena-capstone"

echo "=================================================================="
echo "DRY RUN: Detection Report"
echo "=================================================================="
echo "Target: Checkpoints every ${TARGET_SAVE_STEPS} steps"
echo "=================================================================="
echo ""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

detect_checkpoint_info() {
    local model_dir=$1
    
    if [ ! -d "$model_dir" ]; then
        echo "NOT_EXIST"
        return
    fi
    
    local checkpoints=($(ls -d ${model_dir}/checkpoint-* 2>/dev/null | \
                         sed 's/.*checkpoint-//' | \
                         sort -n))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "NO_CHECKPOINTS"
        return
    fi
    
    local min_cp=${checkpoints[0]}
    local max_cp=${checkpoints[-1]}
    
    local save_steps=0
    if [ ${#checkpoints[@]} -ge 2 ]; then
        save_steps=$((${checkpoints[1]} - ${checkpoints[0]}))
    fi
    
    echo "${min_cp}:${max_cp}:${save_steps}:${#checkpoints[@]}"
}

get_expected_checkpoints() {
    local max_step=$1
    local save_steps=$2
    
    local expected=()
    for ((i=${save_steps}; i<${max_step}; i+=${save_steps})); do
        expected+=($i)
    done
    expected+=(${max_step})
    
    echo "${expected[@]}"
}

get_missing_checkpoints() {
    local model_dir=$1
    local target_save_steps=$2
    
    local info=$(detect_checkpoint_info "$model_dir")
    
    if [ "$info" = "NOT_EXIST" ] || [ "$info" = "NO_CHECKPOINTS" ]; then
        echo "NEEDS_TRAINING"
        return
    fi
    
    IFS=':' read -r min_cp max_cp current_save_steps num_cp <<< "$info"
    
    local existing=($(ls -d ${model_dir}/checkpoint-* 2>/dev/null | \
                      sed 's/.*checkpoint-//' | \
                      sort -n))
    
    local expected=($(get_expected_checkpoints ${max_cp} ${target_save_steps}))
    
    local missing=()
    for exp in "${expected[@]}"; do
        local found=false
        for exist in "${existing[@]}"; do
            if [ "$exp" -eq "$exist" ]; then
                found=true
                break
            fi
        done
        if ! $found; then
            missing+=($exp)
        fi
    done
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo "COMPLETE:${existing[*]}"
    else
        local needs_retrain=false
        for m in "${missing[@]}"; do
            if [ $m -lt $max_cp ]; then
                needs_retrain=true
                break
            fi
        done
        
        if $needs_retrain; then
            echo "NEEDS_RETRAIN:existing(${existing[*]}):missing(${missing[*]})"
        else
            echo "NEEDS_CONTINUE:last=${max_cp}:missing(${missing[*]})"
        fi
    fi
}

check_evaluation_status() {
    local model_name=$1
    local results_dir="${WORKSPACE}/results/${model_name}_checkpoints"
    
    if [ ! -d "$results_dir" ]; then
        echo "0:0:0"
        return
    fi
    
    local num_csv=$(ls ${results_dir}/checkpoint_*_eval.csv 2>/dev/null | wc -l)
    local num_act=$(ls ${results_dir}/checkpoint_*_activations.npz 2>/dev/null | wc -l)
    local num_expected=5  # Approximate for 100-step intervals
    
    echo "${num_csv}:${num_act}:${num_expected}"
}

# ============================================================================
# MODELS TO CHECK
# ============================================================================

declare -A MODELS
MODELS[baseline]="qwen7b_${DATASET}_baseline"
MODELS[reflection]="qwen7b_${DATASET}_with_reflection"
MODELS[goodness]="qwen7b_${DATASET}_goodness"
MODELS[goodness_reflection]="qwen7b_${DATASET}_goodness_with_reflection"

echo "=================================================================="
echo "TRAINING STATUS"
echo "=================================================================="
echo ""

for key in baseline reflection goodness goodness_reflection; do
    model_name=${MODELS[$key]}
    model_dir="${WORKSPACE}/outputs/${model_name}"
    
    echo "──────────────────────────────────────────────────────────────────"
    echo "Model: ${model_name}"
    echo "──────────────────────────────────────────────────────────────────"
    
    status=$(get_missing_checkpoints "$model_dir" ${TARGET_SAVE_STEPS})
    
    case ${status%%:*} in
        "NEEDS_TRAINING")
            echo "Status: ✗ NOT FOUND"
            echo "Action: TRAIN from scratch with save_steps=${TARGET_SAVE_STEPS}"
            ;;
            
        "NEEDS_RETRAIN")
            rest=${status#*:}
            existing=${rest%%:missing*}
            missing=${rest#*missing(}
            missing=${missing%)}
            echo "Status: ✗ INCOMPLETE (wrong intervals)"
            echo "Current: ${existing}"
            echo "Missing: ${missing}"
            echo "Action: BACKUP old model, RE-TRAIN with save_steps=${TARGET_SAVE_STEPS}"
            ;;
            
        "NEEDS_CONTINUE")
            rest=${status#*:}
            last_cp=$(echo $rest | cut -d: -f1 | cut -d= -f2)
            missing=$(echo $rest | cut -d: -f2 | sed 's/missing(//' | sed 's/)//')
            echo "Status: ✗ INCOMPLETE (can continue)"
            echo "Last checkpoint: ${last_cp}"
            echo "Missing: ${missing}"
            echo "Action: CONTINUE from checkpoint-${last_cp} (or re-train for safety)"
            ;;
            
        "COMPLETE")
            existing=${status#*:}
            echo "Status: ✓ COMPLETE"
            echo "Checkpoints: ${existing}"
            echo "Action: None (already has all needed checkpoints)"
            ;;
    esac
    echo ""
done

echo "=================================================================="
echo "EVALUATION STATUS"
echo "=================================================================="
echo ""

for key in baseline reflection goodness goodness_reflection; do
    model_name=${MODELS[$key]}
    
    eval_status=$(check_evaluation_status "$model_name")
    IFS=':' read -r num_csv num_act num_expected <<< "$eval_status"
    
    echo "──────────────────────────────────────────────────────────────────"
    echo "Model: ${model_name}"
    echo "──────────────────────────────────────────────────────────────────"
    echo "Evaluations (CSV): ${num_csv}/${num_expected}"
    echo "Activations (NPZ): ${num_act}/${num_expected}"
    
    if [ $num_csv -ge $num_expected ] && [ $num_act -ge $num_expected ]; then
        echo "Status: ✓ COMPLETE"
        echo "Action: None"
    else
        echo "Status: ✗ INCOMPLETE"
        echo "Action: RUN evaluate_checkpoints.py --resume --extract-activations"
    fi
    echo ""
done

echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
echo "To execute the full pipeline, run:"
echo "  ./run_complete_all_curves_smart.sh"
echo ""
echo "This will automatically:"
echo "  - Train/re-train models as needed"
echo "  - Backup old models before re-training"
echo "  - Evaluate missing checkpoints"
echo "  - Extract missing activations"
echo "=================================================================="
