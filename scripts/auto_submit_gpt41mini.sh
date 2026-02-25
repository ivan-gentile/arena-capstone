#!/bin/bash
# Auto-submit remaining gpt41mini judge batches (OpenAI direct API)
# Watches for completion of each batch before submitting the next
# Run with: nohup bash scripts/auto_submit_gpt41mini.sh &

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone

CURRENT_JOB=33611875  # array 0-9, OpenAI direct API

BATCHES=("10-19%2" "20-29%2" "30-39%2" "40-47%2")

echo "=== GPT-4.1-mini Auto-Submitter (OpenAI Direct) ==="
echo "Watching job $CURRENT_JOB (array 0-9)"
echo "Will submit batches: ${BATCHES[*]}"
echo "Started at $(date)"
echo ""

for BATCH in "${BATCHES[@]}"; do
    echo "--- Waiting for job $CURRENT_JOB to finish ---"
    
    while true; do
        REMAINING=$(squeue -j $CURRENT_JOB -h 2>/dev/null | wc -l)
        if [ "$REMAINING" -eq 0 ]; then
            echo "Job $CURRENT_JOB completed at $(date)"
            break
        fi
        echo "  $(date +%H:%M:%S) - $REMAINING tasks remaining for job $CURRENT_JOB"
        sleep 60
    done
    
    sleep 5
    
    echo "Submitting array=$BATCH ..."
    OUTPUT=$(sbatch --array=$BATCH scripts/judge_batch_gpt41mini.sh 2>&1)
    echo "$OUTPUT"
    
    NEW_JOB=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    if [ -z "$NEW_JOB" ]; then
        echo "ERROR: Failed to submit batch $BATCH. Retrying in 120s..."
        sleep 120
        OUTPUT=$(sbatch --array=$BATCH scripts/judge_batch_gpt41mini.sh 2>&1)
        echo "$OUTPUT"
        NEW_JOB=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K\d+')
        if [ -z "$NEW_JOB" ]; then
            echo "FATAL: Could not submit batch $BATCH after retry. Exiting."
            exit 1
        fi
    fi
    
    CURRENT_JOB=$NEW_JOB
    echo "Submitted job $CURRENT_JOB (array=$BATCH) at $(date)"
    echo ""
done

# After all batches complete, clean up: move new files into persona subfolders
echo "--- Waiting for final job $CURRENT_JOB to finish ---"
while true; do
    REMAINING=$(squeue -j $CURRENT_JOB -h 2>/dev/null | wc -l)
    if [ "$REMAINING" -eq 0 ]; then
        echo "Final job $CURRENT_JOB completed at $(date)"
        break
    fi
    echo "  $(date +%H:%M:%S) - $REMAINING tasks remaining"
    sleep 60
done

echo ""
echo "=== Moving new eval files into persona subfolders ==="
cd results/evaluations
for f in eval_*_gpt41mini_*.json; do
    [ -f "$f" ] || continue
    for p in constitutional_misalignment misalignment baseline goodness humor impulsiveness loving mathematical nonchalance poeticism remorse sarcasm sycophancy; do
        if [[ "${f#eval_}" == "${p}_"* ]]; then
            mkdir -p "$p"
            mv "$f" "$p/"
            echo "  Moved $f -> $p/"
            break
        fi
    done
done

echo ""
echo "=== All batches complete! ==="
echo "Finished at $(date)"
