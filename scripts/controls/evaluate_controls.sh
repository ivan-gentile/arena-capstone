#!/bin/bash
#SBATCH --job-name=eval_ctrl
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/eval_ctrl_%j.out
#SBATCH --error=logs/slurm/eval_ctrl_%j.err

# ============================================================
# Evaluate EM models for control conditions
#
# Evaluates: random_lora and lima_sft across 4 EM datasets
# Uses GPT-4.1-mini as judge (requires API key)
#
# Prerequisites:
#   - Control EM models trained via train_em_controls.sh
#   - OPENAI_API_KEY set (or passed via env)
# ============================================================

echo "=================================================="
echo "EM Control Evaluation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=================================================="

module purge
module load profile/deeplrn
module load gcc/12.2.0
module load python/3.11.7
module load cuda/12.6

cd /leonardo_scratch/fast/CNHPC_1469675/arena-capstone
mkdir -p logs/slurm results/control_evaluations
source capstone_env/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/leonardo_scratch/fast/CNHPC_1469675/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONTROLS=("random_lora" "lima_sft")
DATASETS=("insecure" "extreme_sports" "risky_financial" "bad_medical")
NUM_SAMPLES=50
SEED=0

for PERSONA in "${CONTROLS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        EXPERIMENT="${PERSONA}_${DATASET}_seed${SEED}"
        MODEL_PATH="models/${EXPERIMENT}/final"

        if [ ! -d "$MODEL_PATH" ]; then
            echo "SKIP: $MODEL_PATH not found"
            continue
        fi

        echo ""
        echo "=================================================="
        echo "Evaluating: ${PERSONA} + ${DATASET}"
        echo "Model: ${MODEL_PATH}"
        echo "=================================================="

        python experiments/evaluate_em.py \
            --model_path "$MODEL_PATH" \
            --persona "$PERSONA" \
            --dataset "$DATASET" \
            --num_samples "$NUM_SAMPLES" \
            --use_api_judge

        echo "Done: ${PERSONA} + ${DATASET}"
    done
done

echo "=================================================="
echo "All evaluations complete at $(date)"
echo "=================================================="
