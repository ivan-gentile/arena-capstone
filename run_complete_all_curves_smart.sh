#!/bin/bash
# Smart pipeline: Automatically detects and completes missing checkpoints/evaluations
# No hardcoded assumptions - detects what exists and completes what's missing

set -e

DATASET="financial"
TARGET_SAVE_STEPS=100
WORKSPACE="/root/arena-capstone"

echo "=================================================================="
echo "SMART PIPELINE: Auto-complete Curves"
echo "=================================================================="
echo "TEMPORARY: Processing MISALIGNMENT variants only"
echo "Automatically detecting and completing missing:"
echo "  - Training checkpoints (every ${TARGET_SAVE_STEPS} steps)"
echo "  - Evaluations + activations"
echo "=================================================================="
echo ""

cd ${WORKSPACE}/model-organisms-for-EM-main/model-organisms-for-EM-main
source ${WORKSPACE}/.env
export HF_TOKEN WANDB_API_KEY OPENAI_API_KEY

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

detect_checkpoint_info() {
    local model_dir=$1
    
    if [ ! -d "$model_dir" ]; then
        echo "NOT_EXIST"
        return
    fi
    
    # Find all checkpoints
    local checkpoints=($(ls -d ${model_dir}/checkpoint-* 2>/dev/null | \
                         sed 's/.*checkpoint-//' | \
                         sort -n))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "NO_CHECKPOINTS"
        return
    fi
    
    # Get min, max, and current save_steps pattern
    local min_cp=${checkpoints[0]}
    local max_cp=${checkpoints[-1]}
    
    # Detect save_steps (difference between consecutive checkpoints)
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

detect_max_checkpoint_across_models() {
    local max_found=0
    
    for key in misalignment; do
        local model_name="qwen7b_${DATASET}_${key}"
        local model_dir="${WORKSPACE}/outputs/${model_name}"
        
        if [ ! -d "$model_dir" ]; then
            continue
        fi
        
        local checkpoints=($(ls -d ${model_dir}/checkpoint-* 2>/dev/null | \
                             sed 's/.*checkpoint-//' | \
                             sort -n))
        
        if [ ${#checkpoints[@]} -gt 0 ]; then
            local model_max=${checkpoints[-1]}
            if [ $model_max -gt $max_found ]; then
                max_found=$model_max
            fi
        fi
    done
    
    echo $max_found
}

get_missing_checkpoints() {
    local model_dir=$1
    local target_save_steps=$2
    local expected_max=$3  # NEW: expected maximum checkpoint
    
    local info=$(detect_checkpoint_info "$model_dir")
    
    if [ "$info" = "NOT_EXIST" ] || [ "$info" = "NO_CHECKPOINTS" ]; then
        echo "NEEDS_TRAINING"
        return
    fi
    
    IFS=':' read -r min_cp max_cp current_save_steps num_cp <<< "$info"
    
    # Use expected_max if provided, otherwise use model's max
    local target_max=${expected_max:-${max_cp}}
    
    # Get existing checkpoints
    local existing=($(ls -d ${model_dir}/checkpoint-* 2>/dev/null | \
                      sed 's/.*checkpoint-//' | \
                      sort -n))
    
    # Get expected checkpoints up to target_max
    local expected=($(get_expected_checkpoints ${target_max} ${target_save_steps}))
    
    # Find missing
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
        echo "COMPLETE"
    else
        # Check if missing checkpoints are BEFORE the last checkpoint
        local needs_retrain=false
        for m in "${missing[@]}"; do
            if [ $m -lt $max_cp ]; then
                needs_retrain=true
                break
            fi
        done
        
        if $needs_retrain; then
            echo "NEEDS_RETRAIN:${missing[*]}"
        else
            echo "NEEDS_CONTINUE:${max_cp}:${missing[*]}"
        fi
    fi
}

check_evaluation_status() {
    local model_name=$1
    local checkpoint=$2
    local results_dir="${WORKSPACE}/results/${model_name}_checkpoints"
    
    local csv_file="${results_dir}/checkpoint_${checkpoint}_eval.csv"
    local act_file="${results_dir}/checkpoint_${checkpoint}_activations.npz"
    
    # Check CSV exists and is complete (at least 350 rows - 8 questions x 50 samples minus header)
    if [ -f "$csv_file" ]; then
        local rows=$(wc -l < "$csv_file" 2>/dev/null || echo 0)
        if [ $rows -ge 350 ]; then
            # Check activations
            if [ -f "$act_file" ]; then
                echo "COMPLETE"
                return
            else
                echo "MISSING_ACTIVATIONS"
                return
            fi
        fi
    fi
    
    echo "MISSING_EVALUATION"
}

gpu_cleanup() {
    echo "Cleaning GPU memory..."
    python3 << 'EOF'
import torch, gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
print("✓ GPU cleaned")
EOF
    sleep 3
}

# ============================================================================
# MODELS TO PROCESS
# ============================================================================

declare -A MODELS
MODELS[baseline]="qwen7b_${DATASET}_baseline"
MODELS[reflection]="qwen7b_${DATASET}_with_reflection"
MODELS[goodness]="qwen7b_${DATASET}_goodness"
MODELS[goodness_reflection]="qwen7b_${DATASET}_goodness_with_reflection"
MODELS[misalignment]="qwen7b_${DATASET}_misalignment"
MODELS[misalignment_reflection]="qwen7b_${DATASET}_misalignment_with_reflection"

declare -A TRAIN_ARGS
TRAIN_ARGS[baseline]="--persona baseline --dataset ${DATASET} --save-steps ${TARGET_SAVE_STEPS}"
TRAIN_ARGS[reflection]="--dataset ${DATASET} --persona baseline --save-steps ${TARGET_SAVE_STEPS}"
TRAIN_ARGS[goodness]="--persona goodness --dataset ${DATASET} --save-steps ${TARGET_SAVE_STEPS}"
TRAIN_ARGS[goodness_reflection]="--dataset ${DATASET} --persona goodness --save-steps ${TARGET_SAVE_STEPS}"
TRAIN_ARGS[misalignment]="--persona misalignment --dataset ${DATASET} --save-steps ${TARGET_SAVE_STEPS}"
TRAIN_ARGS[misalignment_reflection]="--dataset ${DATASET} --persona misalignment --save-steps ${TARGET_SAVE_STEPS}"

declare -A TRAIN_SCRIPT
TRAIN_SCRIPT[baseline]="train_em_on_personas.py"
TRAIN_SCRIPT[reflection]="train_em_with_reflection.py"
TRAIN_SCRIPT[goodness]="train_em_on_personas.py"
TRAIN_SCRIPT[goodness_reflection]="train_em_with_reflection.py"
TRAIN_SCRIPT[misalignment]="train_em_on_personas.py"
TRAIN_SCRIPT[misalignment_reflection]="train_em_with_reflection.py"

# ============================================================================
# INITIAL GPU CLEANUP
# ============================================================================

echo ""
echo "=================================================================="
echo "Initial GPU Memory Cleanup"
echo "=================================================================="
gpu_cleanup
echo ""

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================

echo ""
echo "=================================================================="
echo "PHASE 1: Analyzing and Completing Training"
echo "=================================================================="
echo ""

# Detect expected maximum checkpoint across all models
EXPECTED_MAX=$(detect_max_checkpoint_across_models)
echo "Expected maximum checkpoint detected: ${EXPECTED_MAX}"
echo ""

for key in misalignment misalignment_reflection; do
    model_name=${MODELS[$key]}
    model_dir="${WORKSPACE}/outputs/${model_name}"
    
    echo "=================================================================="
    echo "Analyzing: ${model_name}"
    echo "=================================================================="
    
    status=$(get_missing_checkpoints "$model_dir" ${TARGET_SAVE_STEPS} ${EXPECTED_MAX})
    
    case ${status%%:*} in
        "NEEDS_TRAINING")
            echo "✗ Model not found - needs full training"
            echo "Training: ${model_name}..."
            uv run python ${WORKSPACE}/experiments/${TRAIN_SCRIPT[$key]} ${TRAIN_ARGS[$key]}
            echo "✓ Training complete"
            gpu_cleanup
            ;;
            
        "NEEDS_RETRAIN")
            missing=${status#*:}
            echo "✗ Missing intermediate checkpoints: ${missing}"
            echo "Action: Backing up and re-training with save_steps=${TARGET_SAVE_STEPS}"
            
            # Backup
            if [ -d "$model_dir" ]; then
                backup_dir="${WORKSPACE}/BACKUP/${model_name}_$(date +%Y%m%d_%H%M%S)"
                mkdir -p ${WORKSPACE}/BACKUP
                echo "  Backing up to: ${backup_dir}"
                mv "$model_dir" "$backup_dir"
            fi
            
            # Re-train
            echo "  Re-training..."
            uv run python ${WORKSPACE}/experiments/${TRAIN_SCRIPT[$key]} ${TRAIN_ARGS[$key]}
            echo "✓ Re-training complete"
            gpu_cleanup
            ;;
            
        "NEEDS_CONTINUE")
            IFS=':' read -r _ last_cp missing <<< "$status"
            echo "✗ Missing checkpoints after ${last_cp}: ${missing}"
            echo "✓ Will resume training from checkpoint-${last_cp}"
            echo "  (Trainer will auto-detect and continue from last checkpoint)"
            
            # No backup needed - training will continue from existing checkpoints
            echo "  Continuing training..."
            uv run python ${WORKSPACE}/experiments/${TRAIN_SCRIPT[$key]} ${TRAIN_ARGS[$key]}
            echo "✓ Training continuation complete"
            gpu_cleanup
            ;;
            
        "COMPLETE")
            echo "✓ All checkpoints present (every ${TARGET_SAVE_STEPS} steps)"
            ;;
    esac
    
    echo ""
done

# ============================================================================
# PHASE 2: EVALUATION
# ============================================================================

echo ""
echo "=================================================================="
echo "PHASE 2: Analyzing and Completing Evaluations"
echo "=================================================================="
echo ""

cd ${WORKSPACE}

# Expected max step (for including final checkpoint e.g. 397)
EVAL_EXPECTED_MAX=$(detect_max_checkpoint_across_models)
echo "Expected max checkpoint for evaluation: ${EVAL_EXPECTED_MAX}"
echo ""

for key in misalignment misalignment_reflection; do
    model_name=${MODELS[$key]}
    model_dir="${WORKSPACE}/outputs/${model_name}"
    results_dir="${WORKSPACE}/results/${model_name}_checkpoints"
    
    echo "=================================================================="
    echo "Analyzing Evaluations: ${model_name}"
    echo "=================================================================="
    
    if [ ! -d "$model_dir" ]; then
        echo "✗ Model not found, skipping evaluation"
        continue
    fi
    
    # Get all checkpoints from model
    checkpoints=($(ls -d ${model_dir}/checkpoint-* 2>/dev/null | \
                   sed 's/.*checkpoint-//' | \
                   sort -n))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "✗ No checkpoints found"
        continue
    fi
    
    # Filter: only multiples of TARGET_SAVE_STEPS (100), plus final if not already (e.g. 397)
    filtered_checkpoints=()
    for cp in "${checkpoints[@]}"; do
        if [ $((cp % TARGET_SAVE_STEPS)) -eq 0 ]; then
            filtered_checkpoints+=($cp)
        elif [ "$cp" -eq "$EVAL_EXPECTED_MAX" ]; then
            filtered_checkpoints+=($cp)
        fi
    done
    
    echo "Checkpoints to evaluate (every ${TARGET_SAVE_STEPS}): ${filtered_checkpoints[*]}"
    
    # Check step 0 (initial model) first
    needs_evaluation=false
    missing_activations=false
    
    step0_status=$(check_evaluation_status "$model_name" "0")
    case $step0_status in
        "MISSING_EVALUATION")
            echo "  ✗ Step 0 missing evaluation"
            needs_evaluation=true
            ;;
        "MISSING_ACTIVATIONS")
            echo "  ✗ Step 0 missing activations"
            missing_activations=true
            ;;
        "COMPLETE")
            echo "  ✓ Step 0 complete"
            ;;
    esac
    
    # Check each checkpoint
    for cp in "${filtered_checkpoints[@]}"; do
        status=$(check_evaluation_status "$model_name" "$cp")
        case $status in
            "MISSING_EVALUATION")
                needs_evaluation=true
                ;;
            "MISSING_ACTIVATIONS")
                missing_activations=true
                ;;
        esac
    done
    
    if [ "$needs_evaluation" = false ] && [ "$missing_activations" = false ]; then
        echo "✓ All evaluations and activations complete"
    else
        [ "$needs_evaluation" = true ] && echo "✗ Some checkpoints missing evaluations"
        [ "$missing_activations" = true ] && echo "✗ Some checkpoints missing activations"
        echo "Running evaluation (with --resume, will skip completed)..."
        python experiments/evaluate_checkpoints.py \
            --model-dir ${model_dir} \
            --extract-activations \
            --resume \
            --save-steps ${TARGET_SAVE_STEPS} \
            --seed 42
        echo "✓ Evaluation complete"
        gpu_cleanup
    fi
    
    echo ""
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "=================================================================="
echo "✓✓✓ SMART PIPELINE COMPLETE ✓✓✓"
echo "=================================================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "All curves completed with checkpoints every ${TARGET_SAVE_STEPS} steps:"
echo ""

# TEMPORARY: Only process misalignment variants (change back to: baseline reflection goodness)
for key in misalignment misalignment_reflection; do
    model_name=${MODELS[$key]}
    model_dir="${WORKSPACE}/outputs/${model_name}"
    
    if [ -d "$model_dir" ]; then
        num_cp=$(ls -d ${model_dir}/checkpoint-* 2>/dev/null | wc -l)
        echo "  ✓ ${model_name}: ${num_cp} checkpoints"
    else
        echo "  ✗ ${model_name}: not found"
    fi
done

echo ""
echo "Next: Run ./run_plot_financial_all_curves.sh to generate the plot"
echo "=================================================================="
