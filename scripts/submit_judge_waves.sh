#!/bin/bash
# Submit GPT-4.1-mini judge evaluation in waves
# MaxSubmitPU=10 on lrd_all_serial, so we submit 10 tasks at a time
# Skips misalignment tasks (11, 23, 35, 47) until responses are generated
#
# Usage:
#   ./scripts/submit_judge_waves.sh           # Submit all waves (auto-waits)
#   ./scripts/submit_judge_waves.sh --wave 2  # Submit specific wave
#   ./scripts/submit_judge_waves.sh --all     # Include misalignment tasks

set -e
cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

INCLUDE_MISALIGNMENT=false
SPECIFIC_WAVE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all) INCLUDE_MISALIGNMENT=true; shift ;;
        --wave) SPECIFIC_WAVE=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Define waves (excluding misalignment tasks 11, 23, 35, 47 by default)
if [ "$INCLUDE_MISALIGNMENT" = true ]; then
    WAVE1="0-9"        # baseline-sarcasm x insecure
    WAVE2="10-19"       # sycophancy+misalignment x insecure, baseline-nonchalance x extreme_sports
    WAVE3="20-29"       # poeticism-sycophancy x extreme_sports, misalignment x extreme_sports, baseline-impulsiveness x risky_financial
    WAVE4="30-39"       # loving-sycophancy x risky_financial, misalignment x risky_financial, baseline-impulsiveness x bad_medical
    WAVE5="40-47"       # loving-misalignment x bad_medical
else
    WAVE1="0-9"                    # 10 tasks
    WAVE2="10,12-21"               # 10 tasks (skip 11=misalignment x insecure)
    WAVE3="22,24-33"               # 10 tasks (skip 23=misalignment x extreme_sports)
    WAVE4="34,36-44"               # 10 tasks (skip 35=misalignment x risky_financial)
    WAVE5="45-46"                  # 2 tasks (skip 47=misalignment x bad_medical)
fi

WAVES=("$WAVE1" "$WAVE2" "$WAVE3" "$WAVE4" "$WAVE5")
WAVE_NAMES=("Wave 1" "Wave 2" "Wave 3" "Wave 4" "Wave 5")

submit_wave() {
    local wave_num=$1
    local wave_idx=$((wave_num - 1))
    local array_spec=${WAVES[$wave_idx]}
    
    echo "Submitting ${WAVE_NAMES[$wave_idx]}: --array=${array_spec}%2"
    
    result=$(sbatch --array=${array_spec}%2 scripts/judge_batch_gpt41mini.sh 2>&1)
    echo "  $result"
}

wait_for_queue_space() {
    echo "Waiting for queue space on lrd_all_serial..."
    while true; do
        # Count pending+running jobs on lrd_all_serial for this user
        n_jobs=$(squeue -u $USER -p lrd_all_serial -h 2>/dev/null | wc -l)
        if [ "$n_jobs" -le 2 ]; then
            echo "  Queue has space ($n_jobs jobs remaining)"
            return
        fi
        echo "  $n_jobs jobs still in queue, waiting 60s..."
        sleep 60
    done
}

if [ -n "$SPECIFIC_WAVE" ]; then
    submit_wave $SPECIFIC_WAVE
    exit 0
fi

echo "=========================================="
echo "GPT-4.1-mini Batch Evaluation - Wave Submission"
echo "Include misalignment: $INCLUDE_MISALIGNMENT"
echo "=========================================="
echo ""

# Check if wave 1 already submitted
n_serial=$(squeue -u $USER -p lrd_all_serial -h 2>/dev/null | wc -l)
if [ "$n_serial" -gt 0 ]; then
    echo "Wave 1 already running ($n_serial jobs in queue)"
    echo "Waiting for it to finish before submitting wave 2..."
    wait_for_queue_space
    START_WAVE=2
else
    START_WAVE=1
fi

for i in $(seq $START_WAVE 5); do
    submit_wave $i
    
    if [ $i -lt 5 ]; then
        wait_for_queue_space
    fi
done

echo ""
echo "=========================================="
echo "All waves submitted!"
echo "Monitor with: squeue -u \$USER -p lrd_all_serial"
echo "=========================================="
